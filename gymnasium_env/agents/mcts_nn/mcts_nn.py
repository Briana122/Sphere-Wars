import math
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium_env.agents.base_agent import BaseAgent
from gymnasium_env.utils.constants import MAX_RESOURCES, MAX_STEPS, SPAWN_COST
from gymnasium_env.game.Game import Game
from gymnasium_env.game.Piece import Piece
from gymnasium_env.game.Tile import Tile


class PolicyValueNet(nn.Module):
    """
    Simple MLP that maps encoded observation -> (policy_logits, value).
    - policy_logits: shape [batch, action_size]
    - value: shape [batch, 1], in [-1, 1] via tanh
    """

    def __init__(self, obs_dim: int, action_size: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, obs_dim]
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))  # [-1, 1]
        return policy_logits, value.squeeze(-1)


class MCTSNode:
    """
    Tree node used by MCTS.
    Stores:
      - prior: P(s, a) from the policy network
      - N, W, Q: visit count, total value, mean value
      - children: dict[action_tuple] -> child_node
      - player: player to move at this node
      - is_terminal / terminal_value: for terminal positions
    """

    __slots__ = (
        "player",
        "prior",
        "N",
        "W",
        "Q",
        "children",
        "is_terminal",
        "terminal_value",
    )

    def __init__(self, player, prior: float = 0.0):
        self.player = player
        self.prior = float(prior)
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children = {}  # (pid, dest, action_type) -> MCTSNode
        self.is_terminal = False
        self.terminal_value = 0.0


class MCTSNNAgent(BaseAgent):
    """
    AlphaZero-style MCTS + neural network agent.

    Usage:
      - Call set_env(env) once so the agent can deep copy the Game state.
      - select_action(obs, legal_actions) runs MCTS from the current Game state
        in env.game (no environment stepping inside the search).
    """

    def __init__(
        self,
        action_space,
        observation_space,
        num_simulations: int = 128,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: str = None,
    ):
        super().__init__(action_space, observation_space)

        self.name = "mcts_nn"

        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

        # Action flattening: (pid, dest, action_type) -> flat index
        self.max_pieces = action_space.spaces[0].n
        self.num_tiles = action_space.spaces[1].n
        self.num_action_types = action_space.spaces[2].n
        self.action_size = self.max_pieces * self.num_tiles * self.num_action_types

        # Observation encoding size
        num_tiles = observation_space["ownership"].shape[0]
        num_players = observation_space["resources"].shape[0]
        # ownership (num_tiles), piece_owner (num_tiles),
        # resources (num_players), plus 3 scalars
        self.obs_dim = 2 * num_tiles + num_players + 3

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PolicyValueNet(self.obs_dim, self.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        # env reference (must be set externally)
        self.env = None

        # very basic replay buffer 
        self.replay_buffer = []


    def set_env(self, env):
        """
        Attach the environment so we can access env.game.
        Call this once after creating the agent.
        """
        self.env = env


    def action_to_index(self, action):
        """
        Map (pid, dest, action_type) to a flat index in [0, action_size).
        """
        pid, dest, a_type = action
        return pid * (self.num_tiles * self.num_action_types) + dest * self.num_action_types + a_type

    def index_to_action(self, idx):
        """
        Inverse of action_to_index. Mostly useful if you ever
        want to sample directly from the policy vector.
        """
        pid_block = self.num_tiles * self.num_action_types
        pid = idx // pid_block
        rem = idx % pid_block
        dest = rem // self.num_action_types
        a_type = rem % self.num_action_types
        return (int(pid), int(dest), int(a_type))

    def _encode_obs(self, obs):
        """
        Encode env-style observation (dict) -> flat torch tensor [1, obs_dim].
        """
        ownership = obs["ownership"].astype(np.float32)
        piece_owner = obs["piece_owner"].astype(np.float32)
        resources = obs["resources"].astype(np.float32)

        current_player = float(obs["current_player"])
        step_count = float(obs["step_count"])
        tiles_to_win = float(obs["tiles_to_win"])

        num_players = resources.shape[0]
        max_player_id = max(1.0, num_players - 1.0)

        ownership_norm = ownership / max_player_id
        piece_owner_norm = piece_owner / max_player_id
        resources_norm = resources / max(1.0, float(MAX_RESOURCES))
        step_norm = step_count / max(1.0, float(MAX_STEPS))
        tiles_norm = tiles_to_win / max(1.0, float(self.num_tiles))

        vec = np.concatenate(
            [
                ownership_norm,
                piece_owner_norm,
                resources_norm,
                np.array([current_player, step_norm, tiles_norm], dtype=np.float32),
            ],
            dtype=np.float32,
        )

        x = torch.from_numpy(vec).unsqueeze(0).to(self.device)
        return x

    def _build_obs_from_game(self, game, base_step_count=0, base_tiles_to_win=None):
        """
        Recreate an observation dict from a Game instance.
        Matches GameEnv._get_obs as closely as possible.
        """
        ownership = np.full((self.num_tiles,), -1, dtype=np.int8)
        piece_owner = np.full((self.num_tiles,), -1, dtype=np.int8)

        for tid, tile in game.tiles.items():
            if tile.owner is not None:
                ownership[tid] = tile.owner
            if tile.piece is not None:
                piece_owner[tid] = tile.piece[0]

        resources = np.array(
            [game.resources[p] for p in range(game.players)],
            dtype=np.int32,
        )

        if base_tiles_to_win is None:
            base_tiles_to_win = int(math.ceil(self.num_tiles / 2.0))

        obs = {
            "ownership": ownership,
            "piece_owner": piece_owner,
            "resources": resources,
            "current_player": game.current_player,
            "step_count": np.int32(base_step_count),
            "tiles_to_win": np.int32(base_tiles_to_win),
        }
        return obs

    def _get_legal_actions_for_game(self, game):
        """
        Version of get_legal_actions that works directly on a Game instance.
        Returns list of (pid, dest, action_type).
        """
        current = game.current_player
        legal = []

        can_spawn = game.resources[current] >= SPAWN_COST

        all_keys = list(game.pieces.keys())
        for (agent, pid) in all_keys:
            if agent != current:
                continue
            if game.moved_flags[current].get(pid, False):
                continue

            piece = game.pieces[(agent, pid)]
            moves = game.legal_moves(piece)

            # Move actions
            for dest in moves:
                legal.append((pid, dest, 0))

            # Spawn action 
            if can_spawn:
                legal.append((pid, piece.tile_id, 1))

        return legal

    def _apply_action_inplace(self, game, action):
        """
        Apply (pid, dest, action_type) directly to a Game instance.
        Returns:
           - done (bool),
           - winner (int or None)
        Game.current_player and moved_flags are updated accordingly.
        """
        pid, dest, action_type = action
        current = game.current_player

        piece_key = (current, pid)
        if piece_key not in game.pieces:
            # Should never happen if we respect legal moves
            return False, None

        piece = game.pieces[piece_key]

        # Spawn
        if action_type == 1:
            spawned = False
            if game.resources[current] >= SPAWN_COST:
                spawned = game.spawn_piece(current, cost=SPAWN_COST)

            if not spawned:
                result = game.move(piece, dest)
                if isinstance(result, tuple):
                    moved, _ = result
                else:
                    moved = result  

        else:
            result = game.move(piece, dest)
            if isinstance(result, tuple):
                moved, _ = result
            else:
                moved = result  


        # Victory check
        if game.check_victory(last_agent=current):
            return True, game.winner

        # If no more legal actions for this player, end their turn
        remaining = self._get_legal_actions_for_game(game)
        if not remaining:
            game.end_turn()

        return False, None

    def _run_mcts(self, root_obs, root_game, root_legal_actions):
        """
        Run MCTS from the given root position.

        root_obs: observation dict for the root (as returned by env._get_obs())
        root_game: deep copy of env.game
        root_legal_actions: list of legal (pid, dest, action_type) at root

        Returns:
            policy_vector over legal actions (dict action -> visit_count),
            and the selected action.
        """
        if self.env is None:
            raise RuntimeError("MCTSNNAgent.env is not set. Call agent.set_env(env) once before using.")

        current_player = root_game.current_player

        # Build root node
        root = MCTSNode(player=current_player, prior=1.0)
        root.children = {}

        # Evaluate root to get priors & value, and expand it
        with torch.no_grad():
            x = self._encode_obs(root_obs)
            logits, value = self.net(x)
            logits = logits[0].cpu().numpy()

        # Build prior distribution only over legal actions
        priors = {}
        raw_ps = []
        for a in root_legal_actions:
            idx = self.action_to_index(a)
            p = math.exp(logits[idx])  # unnormalized softmax
            priors[a] = p
            raw_ps.append(p)

        if len(priors) == 0:
            # No legal actions: nothing to search, just return None
            return {}, None

        sum_p = sum(raw_ps)
        if sum_p <= 0:
            # fallback to uniform
            for a in priors:
                priors[a] = 1.0 / len(priors)
        else:
            for a in priors:
                priors[a] /= sum_p

        if self.dirichlet_alpha is not None and self.dirichlet_eps is not None:
            alpha = self.dirichlet_alpha
            noise = np.random.dirichlet([alpha] * len(priors))
            for (a, n) in zip(priors.keys(), noise):
                priors[a] = (1 - self.dirichlet_eps) * priors[a] + self.dirichlet_eps * float(n)

        # Create root children
        for a, p in priors.items():
            child = MCTSNode(player=None, prior=p)
            root.children[a] = child

        if root_game.winner is not None:
            root.is_terminal = True
            root.terminal_value = 1.0 if root_game.winner == current_player else -1.0

        # Run simulations
        for _ in range(self.num_simulations):
            game = copy.deepcopy(root_game)
            node = root
            path = [node]

            # Selection & expansion
            done = game.winner is not None
            winner = game.winner

            while True:
                if node.is_terminal:
                    # terminal node -> use stored terminal value
                    leaf_value = node.terminal_value
                    break

                if not node.children:
                    # Leaf: expand using the network
                    leaf_obs = self._build_obs_from_game(
                        game,
                        base_step_count=int(root_obs["step_count"]),
                        base_tiles_to_win=int(root_obs["tiles_to_win"]),
                    )
                    with torch.no_grad():
                        x_leaf = self._encode_obs(leaf_obs)
                        logits_leaf, value_leaf = self.net(x_leaf)
                        logits_leaf = logits_leaf[0].cpu().numpy()
                        v = float(value_leaf.cpu().item())

                    legal = self._get_legal_actions_for_game(game)
                    if not legal:
                        # No legal moves: treat as terminal draw
                        node.is_terminal = True
                        node.terminal_value = 0.0
                        leaf_value = 0.0
                        break

                    # Build priors for children
                    priors_child = {}
                    raw = []
                    for a in legal:
                        idx = self.action_to_index(a)
                        p = math.exp(logits_leaf[idx])
                        priors_child[a] = p
                        raw.append(p)

                    s = sum(raw)
                    if s <= 0:
                        for a in priors_child:
                            priors_child[a] = 1.0 / len(priors_child)
                    else:
                        for a in priors_child:
                            priors_child[a] /= s

                    # Create children
                    for a, p in priors_child.items():
                        child_player = game.current_player  # player to move after a
                        child = MCTSNode(player=child_player, prior=p)
                        node.children[a] = child

                    leaf_value = v
                    break

                # Selection: choose action that maximizes PUCT
                best_score = -float("inf")
                best_action = None
                best_child = None

                sqrt_N = math.sqrt(max(1, node.N))
                for a, child in node.children.items():
                    U = self.c_puct * child.prior * (sqrt_N / (1 + child.N))
                    score = child.Q + U
                    if score > best_score:
                        best_score = score
                        best_action = a
                        best_child = child

                # Apply chosen action
                done, winner = self._apply_action_inplace(game, best_action)
                # Update child's player (player to move after action)
                best_child.player = game.current_player
                node = best_child
                path.append(node)

                if done:
                    # Terminal
                    if winner is None:
                        leaf_value = 0.0
                    else:
                        # Value is from the viewpoint of the player_to_move at *this* node.
                        # winner == node.player -> +1, else -1
                        leaf_value = 1.0 if winner == node.player else -1.0
                    node.is_terminal = True
                    node.terminal_value = leaf_value
                    break

            # Backup: alternate sign of value up the tree
            value = leaf_value
            for n in reversed(path):
                n.N += 1
                n.W += value
                n.Q = n.W / n.N
                value = -value

        # After simulations, choose action ~ argmax(N(s, a))
        visit_counts = {a: child.N for a, child in root.children.items()}
        best_action = max(visit_counts.items(), key=lambda kv: kv[1])[0]

        return visit_counts, best_action

    def select_action(self, state, legal_actions):
        """
        Called from play_game:
            action = agent.select_action(obs, legal_actions)

        Runs MCTS starting from env.game.
        """
        if self.env is None:
            raise RuntimeError(
                "MCTSNNAgent.env is not set. "
                "In your main script, after creating env and agent, call agent.set_env(env)."
            )

        if not legal_actions:
            return None

        # Use the real Game instance as root; deep-copy for simulations
        root_game = copy.deepcopy(self.env.game)
        root_obs = state  # obs dict from GameEnv._get_obs

        _, chosen_action = self._run_mcts(root_obs, root_game, legal_actions)
        return chosen_action

    def train(self, batch):
        """
        Very basic AlphaZero-style training step (optional).

        Expects batch to be a dict with:
          - "obs": list of observation dicts
          - "policy_target": list of numpy arrays of shape [action_size]
          - "value_target": list/array of scalars

        If you don't use this, you can ignore it.
        """
        obs_list = batch["obs"]
        policy_target = np.stack(batch["policy_target"], axis=0)  # [B, A]
        value_target = np.array(batch["value_target"], dtype=np.float32)  # [B]

        X = torch.cat([self._encode_obs(o) for o in obs_list], dim=0)
        policy_target = torch.from_numpy(policy_target).to(self.device)
        value_target = torch.from_numpy(value_target).to(self.device)

        logits, values = self.net(X)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        # Policy loss: cross-entropy between target distribution and predicted probs
        policy_loss = -(policy_target * log_probs).sum(dim=-1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(values, value_target)

        # Optional entropy regularization
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = policy_loss + value_loss - 1e-4 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    def save_model(self, filepath):
        torch.save(self.net.state_dict(), filepath)

    def load_model(self, filepath):
        state_dict = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
