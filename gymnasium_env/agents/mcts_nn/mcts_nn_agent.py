# mcts_nn_agent.py

import math
from collections import defaultdict, OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium_env.agents.base_agent import BaseAgent
from gymnasium_env.utils.constants import MAX_RESOURCES, MAX_STEPS, SPAWN_COST
from gymnasium_env.game.Game import Game
from gymnasium_env.game.Piece import Piece
from gymnasium_env.game.Tile import Tile
from gymnasium_env.game.Player import Player

def fast_clone_tile(tile: Tile):
    t = Tile.__new__(Tile)
    t.id = tile.id
    t.center3d = tile.center3d
    t.owner = tile.owner
    t.resources = tile.resources
    t.piece = tile.piece 
    return t

def fast_clone_piece(piece: Piece):
    p = Piece.__new__(Piece)
    p.agent = piece.agent
    p.pid = piece.pid
    p.tile_id = piece.tile_id
    return p

def fast_clone_player(player: Player):
    p = Player.__new__(Player)
    p.player_id = player.player_id
    p.tiles_owned = list(player.tiles_owned)
    p.pieces_owned = list(player.pieces_owned)
    p.acqumilated_resources = player.acqumilated_resources
    p.total_moves = player.total_moves
    return p



def fast_clone_game(game: Game):
    g = Game.__new__(Game)

    # shared/static objects
    g.hex = game.hex
    g.players = game.players

    # basic scalar fields
    g.current_player = game.current_player
    g.selected = game.selected
    g.rot = list(game.rot)
    g.winner = game.winner

    # shallow-copied dicts
    g.resources = dict(game.resources)
    g.moved_flags = {a: dict(flags) for a, flags in game.moved_flags.items()}

    # tiles
    g.tiles = {tid: fast_clone_tile(t) for tid, t in game.tiles.items()}

    # pieces
    g.pieces = {key: fast_clone_piece(piece) for key, piece in game.pieces.items()}

    # players
    g.player_objs = [fast_clone_player(p) for p in game.player_objs]

    # logs (not required during MCTS but safe to copy shallow)
    g.episode_log = list(game.episode_log)

    return g


class PolicyValueNet(nn.Module):
    """
    Simple MLP that maps encoded observation -> (policy_logits, value).
    - policy_logits: [B, action_size]
    - value: [B], in [-1, 1]
    """

    def __init__(self, obs_dim: int, action_size: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        return policy_logits, value.squeeze(-1)


class MCTSNode:
    """
    A node in the MCTS tree.
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

    def __init__(self, player, prior: float):
        self.player = player
        self.prior = float(prior)
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children = {}       # action -> MCTSNode
        self.is_terminal = False
        self.terminal_value = 0.0


class MCTSNNAgent(BaseAgent):
    """
    AlphaZero-style MCTS + NN agent.

    Important knobs you can tweak at runtime:
      - self.num_simulations (MCTS rollouts per move)
      - self.use_cache / self.max_cache_size (NN caching)
    """

    def __init__(
        self,
        action_space,
        observation_space,
        num_simulations: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: str = None,
    ):
        super().__init__(action_space, observation_space)

        self.name = "mcts_nn"

        # MCTS hyperparameters
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

        # action flattening: (pid, dest, action_type)
        self.max_pieces = action_space.spaces[0].n
        self.num_tiles = action_space.spaces[1].n
        self.num_action_types = action_space.spaces[2].n
        self.action_size = self.max_pieces * self.num_tiles * self.num_action_types

        # observation encoding size
        num_tiles = observation_space["ownership"].shape[0]
        num_players = observation_space["resources"].shape[0]
        # ownership (num_tiles), piece_owner (num_tiles),
        # resources (num_players), plus 3 scalars
        self.obs_dim = 2 * num_tiles + num_players + 3

        # device / net
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PolicyValueNet(self.obs_dim, self.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        # env reference
        self.env = None

        # simple policy/value cache: obs_key -> (logits_np, value_float)
        self.use_cache = True
        self.max_cache_size = 10000
        self.policy_value_cache = OrderedDict()


    def set_env(self, env):
        """
        Attach the environment so we can access env.game for search.
        """
        self.env = env

    def action_to_index(self, action):
        pid, dest, a_type = action
        return pid * (self.num_tiles * self.num_action_types) + dest * self.num_action_types + a_type

    def index_to_action(self, idx):
        pid_block = self.num_tiles * self.num_action_types
        pid = idx // pid_block
        rem = idx % pid_block
        dest = rem // self.num_action_types
        a_type = rem % self.num_action_types
        return (int(pid), int(dest), int(a_type))

    # -----------------------------------------------------
    # Observation encoding + cache keys
    # -----------------------------------------------------

    def _encode_obs(self, obs):
        """
        Encode env-style observation (dict) -> torch tensor [1, obs_dim].
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
        Matches GameEnv._get_obs reasonably closely for MCTS.
        """
        ownership = np.full((self.num_tiles,), -1, dtype=np.int8)
        piece_owner = np.full((self.num_tiles,), -1, dtype=np.int8)

        for tid, tile in game.tiles.items():
            if tile.owner is not None:
                ownership[tid] = tile.owner
            if tile.piece is not None:
                piece_owner[tid] = tile.piece[0]

        # assume contiguous player ids [0..players-1]
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

    def _obs_to_key(self, obs):
        """
        Build a hashable key for caching from an observation dict.
        """
        return (
            obs["ownership"].tobytes(),
            obs["piece_owner"].tobytes(),
            obs["resources"].tobytes(),
            int(obs["current_player"]),
            int(obs["step_count"]),
            int(obs["tiles_to_win"]),
        )

    def _get_policy_value_cached(self, obs):
        """
        Returns (logits_np, value_float), using cache if enabled.
        """
        if not self.use_cache:
            return self._forward_policy_value(obs)

        key = self._obs_to_key(obs)
        cached = self.policy_value_cache.get(key, None)
        if cached is not None:
            # move to end for LRU
            self.policy_value_cache.move_to_end(key)
            return cached

        logits_np, value = self._forward_policy_value(obs)
        self.policy_value_cache[key] = (logits_np, value)
        if len(self.policy_value_cache) > self.max_cache_size:
            self.policy_value_cache.popitem(last=False)
        return logits_np, value

    def _forward_policy_value(self, obs):
        """
        Single NN forward pass for an obs dict.
        Returns (logits_np, value_float).
        """
        with torch.no_grad():
            x = self._encode_obs(obs)
            logits, value = self.net(x)
            logits_np = logits[0].cpu().numpy()
            v = float(value.cpu().item())
        return logits_np, v

    # ---------------------------------------------------------
    # Game simulation helpers
    # ---------------------------------------------------------

    def _get_legal_actions_for_game(self, game):
        """
        Version of get_legal_actions that works directly on a Game instance.
        Returns list of (pid, dest, action_type).
        """
        current = game.current_player
        legal = []

        can_spawn = game.resources[current] >= SPAWN_COST

        for (agent, pid), piece in game.pieces.items():
            if agent != current:
                continue
            if game.moved_flags[current].get(pid, False):
                continue

            moves = game.legal_moves(piece)
            for dest in moves:
                legal.append((pid, dest, 0))

            if can_spawn:
                legal.append((pid, piece.tile_id, 1))

        return legal

    def _apply_action_inplace(self, game, action):
        """
        Apply (pid, dest, action_type) to a Game clone.
        Returns (done, winner).
        """
        pid, dest, action_type = action
        current = game.current_player
        key = (current, pid)

        if key not in game.pieces:
            return False, None

        piece = game.pieces[key]

        if action_type == 1:
            # SPAWN
            spawned = False
            if game.resources[current] >= SPAWN_COST:
                spawned = game.spawn_piece(current, cost=SPAWN_COST)
            if not spawned:
                game.move(piece, dest)
        else:
            # MOVE
            game.move(piece, dest)

        if game.check_victory(last_agent=current):
            return True, game.winner

        remaining = self._get_legal_actions_for_game(game)
        if not remaining:
            game.end_turn()

        return False, None

    # ---------------------------------------------------------
    # MCTS core
    # ---------------------------------------------------------

    def _run_mcts(self, root_obs, root_game, root_legal_actions):
        """
        Run MCTS from the given root position.
        Returns:
            visit_counts (dict[action] -> int),
            chosen_action
        """
        if self.env is None:
            raise RuntimeError("MCTSNNAgent.env is not set. Call agent.set_env(env) once before using.")

        current_player = root_game.current_player
        root = MCTSNode(player=current_player, prior=1.0)

        # evaluate root for priors/value
        logits, _ = self._get_policy_value_cached(root_obs)

        priors = {}
        raw = []
        for a in root_legal_actions:
            idx = self.action_to_index(a)
            p = math.exp(logits[idx])
            priors[a] = p
            raw.append(p)

        if len(priors) == 0:
            return {}, None

        s = sum(raw)
        if s <= 0:
            for a in priors:
                priors[a] = 1.0 / len(priors)
        else:
            for a in priors:
                priors[a] /= s

        # add Dirichlet noise at root if training
        if self.dirichlet_alpha is not None and self.dirichlet_eps is not None:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
            for (a, n) in zip(priors.keys(), noise):
                priors[a] = (1 - self.dirichlet_eps) * priors[a] + self.dirichlet_eps * float(n)

        for a, p in priors.items():
            root.children[a] = MCTSNode(player=None, prior=p)

        if root_game.winner is not None:
            root.is_terminal = True
            root.terminal_value = 1.0 if root_game.winner == current_player else -1.0

        # ----- MCTS simulations -----
        for _ in range(self.num_simulations):
            game = fast_clone_game(root_game)
            node = root
            path = [node]

            while True:
                if node.is_terminal:
                    leaf_value = node.terminal_value
                    break

                if not node.children:
                    # leaf: expand using network
                    leaf_obs = self._build_obs_from_game(
                        game,
                        base_step_count=int(root_obs["step_count"]),
                        base_tiles_to_win=int(root_obs["tiles_to_win"]),
                    )
                    logits_leaf, v = self._get_policy_value_cached(leaf_obs)

                    legal = self._get_legal_actions_for_game(game)
                    if not legal:
                        node.is_terminal = True
                        node.terminal_value = 0.0
                        leaf_value = 0.0
                        break

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

                    for a, p in priors_child.items():
                        child = MCTSNode(player=game.current_player, prior=p)
                        node.children[a] = child

                    leaf_value = v
                    break

                # selection: PUCT
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

                done, winner = self._apply_action_inplace(game, best_action)
                best_child.player = game.current_player
                node = best_child
                path.append(node)

                if done:
                    if winner is None:
                        leaf_value = 0.0
                    else:
                        leaf_value = 1.0 if winner == node.player else -1.0
                    node.is_terminal = True
                    node.terminal_value = leaf_value
                    break

            # backup
            value = leaf_value
            for n in reversed(path):
                n.N += 1
                n.W += value
                n.Q = n.W / n.N
                value = -value

        visit_counts = {a: child.N for a, child in root.children.items()}
        best_action = max(visit_counts.items(), key=lambda kv: kv[1])[0]
        return visit_counts, best_action

    def select_action(self, state, legal_actions):
        """
        Called from play_game:
            action = agent.select_action(obs, legal_actions)
        """
        if self.env is None:
            raise RuntimeError(
                "MCTSNNAgent.env is not set. "
                "In your main script, after creating env and agent, call agent.set_env(env)."
            )

        if not legal_actions:
            return None

        root_game = fast_clone_game(self.env.game)
        root_obs = state
        _, chosen_action = self._run_mcts(root_obs, root_game, legal_actions)
        return chosen_action


    def train(self, batch):
        """
        Expects:
          - batch["obs"]: list of obs dicts
          - batch["policy_target"]: list of [action_size] numpy arrays
          - batch["value_target"]: list of scalars
        """
        obs_list = batch["obs"]
        policy_target = np.stack(batch["policy_target"], axis=0)
        value_target = np.array(batch["value_target"], dtype=np.float32)

        X = torch.cat([self._encode_obs(o) for o in obs_list], dim=0)
        policy_target = torch.from_numpy(policy_target).to(self.device)
        value_target = torch.from_numpy(value_target).to(self.device)

        logits, values = self.net(X)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        policy_loss = -(policy_target * log_probs).sum(dim=-1).mean()
        value_loss = F.mse_loss(values, value_target)
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
