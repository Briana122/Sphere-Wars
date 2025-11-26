# gymnasium_env/agents/actor_critic/ac_agent.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

from gymnasium_env.agents.base_agent import BaseAgent
from gymnasium_env.utils.constants import MAX_RESOURCES, MAX_STEPS
from .ac_model import ActorCriticNet


class ActorCriticAgent(BaseAgent):
    """
    On-policy Actor–Critic agent for GameEnv.

    - Uses a shared network:
        policy head → logits over flattened action space
        value head  → V(s)
    - Masks illegal actions before sampling.
    """

    def __init__(
        self,
        action_space,
        observation_space,
        lr=3e-4,
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01,
        hidden_dim=256,
        device=None,
    ):
        super().__init__(action_space, observation_space)

        # ----- Env sizes -----
        self.num_tiles = observation_space["ownership"].shape[0]
        self.num_players = observation_space["resources"].shape[0]

        self.max_pieces = int(action_space[0].n)
        self.num_action_tiles = int(action_space[1].n)
        self.num_action_types = int(action_space[2].n)

        # ----- State dimensionality -----
        # ownership:       num_tiles
        # piece_owner:     num_tiles
        # resources:       num_players
        # current_player:  num_players (one-hot)
        # step_count:      1
        # tiles_to_win:    1
        self.state_dim = (2 * self.num_tiles + 2 * self.num_players + 2)

        # Flattened action count
        self.num_actions = (self.max_pieces * self.num_action_tiles * self.num_action_types)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Network + optimizer
        self.model = ActorCriticNet(
            self.state_dim, self.num_actions, hidden_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    # ------------------------------------------------------------
    # State / Action Encoding
    # ------------------------------------------------------------
    def encode_state(self, obs, current_player):
        """
        Encode observation + current player into a flat state vector.

        Uses:
        - ownership (per tile)
        - piece_owner (per tile)
        - resources (per player, normalized)
        - current_player (one-hot)
        - step_count (normalized)
        - tiles_to_win (normalized)
        """
        ownership = obs["ownership"].astype(np.float32)
        piece_owner = obs["piece_owner"].astype(np.float32)
        resources = obs["resources"].astype(np.float32)
        step_count = float(obs["step_count"])
        tiles_to_win = float(obs["tiles_to_win"])

        # Normalize ownership & piece_owner to roughly [-1, 1]
        # ownership, piece_owner ∈ [-1, num_players-1]
        denom = max(self.num_players - 1, 1)
        ownership = ownership / float(denom)
        piece_owner = piece_owner / float(denom)

        # Normalize resources and counts
        resources_norm = resources / float(MAX_RESOURCES)
        step_norm = step_count / float(MAX_STEPS)
        tiles_norm = tiles_to_win / float(self.num_tiles)

        # Current player one-hot
        current_player_oh = np.zeros(self.num_players, dtype=np.float32)
        if 0 <= current_player < self.num_players:
            current_player_oh[current_player] = 1.0

        # Concatenate everything
        state_vec = np.concatenate(
            [
                ownership,              # num_tiles
                piece_owner,            # num_tiles
                resources_norm,         # num_players
                current_player_oh,      # num_players
                np.array([step_norm], dtype=np.float32),   # 1
                np.array([tiles_norm], dtype=np.float32),  # 1
            ]
        ).astype(np.float32)

        assert state_vec.shape[0] == self.state_dim, (
            f"State dim mismatch: got {state_vec.shape[0]}, expected {self.state_dim}"
        )

        return state_vec

    def action_to_index(self, action):
        piece_idx, dest_tile, action_type = action
        return (
            piece_idx * (self.num_action_tiles * self.num_action_types)
            + dest_tile * self.num_action_types
            + action_type
        )

    def index_to_action(self, index):
        base = self.num_action_tiles * self.num_action_types
        piece_idx = index // base
        rem = index % base
        dest_tile = rem // self.num_action_types
        action_type = rem % self.num_action_types
        return int(piece_idx), int(dest_tile), int(action_type)

    def build_legal_mask(self, legal_actions):
        """
        Build a boolean mask over the flattened action space:
        True  = legal
        False = illegal
        """
        mask = np.zeros(self.num_actions, dtype=bool)
        for a in legal_actions:
            idx = self.action_to_index(a)
            if 0 <= idx < self.num_actions:
                mask[idx] = True
        return mask

    # ------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------
    def select_action(
        self,
        obs,
        legal_actions,
        current_player,
        greedy: bool = False,
    ):
        """
        Select an action.

        greedy = False → sample from the masked categorical (for training)
        greedy = True  → argmax over masked logits (for evaluation)
        """
        if not legal_actions:
            return None, {}

        state_vec = self.encode_state(obs, current_player)
        state_t = torch.from_numpy(state_vec).to(self.device).unsqueeze(0)

        legal_mask = self.build_legal_mask(legal_actions)
        mask_t = torch.from_numpy(legal_mask).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(state_t)
            logits = logits.squeeze(0)

            # Kill illegal actions
            masked_logits = logits.clone()
            masked_logits[~mask_t] = -1e9

            if greedy:
                action_index = torch.argmax(masked_logits).item()
            else:
                dist = Categorical(logits=masked_logits)
                action_index = dist.sample().item()

        action_tuple = self.index_to_action(action_index)

        return action_tuple, {
            "state": state_vec,
            "action_index": action_index,
            "mask": legal_mask,
        }

    # ------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------
    def update(self, states, action_indices, rewards, dones, masks):
        """
        states:         [T, state_dim]          (np.float32)
        action_indices: [T]                     (np.int64)
        rewards:        [T]                     (np.float32)
        dones:          [T]                     (np.bool or 0/1)
        masks:          [T, num_actions]        (np.bool) legal-mask per step
        """
        T = len(rewards)

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(action_indices).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.float32)).to(self.device)
        masks_t = torch.from_numpy(masks).to(self.device)

        # Compute discounted returns
        returns = torch.zeros_like(rewards_t)
        next_return = 0.0

        for t in reversed(range(T)):
            not_done = 1.0 - dones_t[t]
            next_return = rewards_t[t] + self.gamma * next_return * not_done
            returns[t] = next_return

        print("update: states_t.shape:", states_t.shape)
        print("update: masks_t.shape:", masks_t.shape)
        
        logits, values = self.model(states_t)

        # Apply action masks to logits
        masked_logits = logits.clone()
        masked_logits[~masks_t] = -1e9

        dist = Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        # Advantage (with normalization)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs * advantages).mean()
        value_loss = torch.mean((returns - values) ** 2)

        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    # ------------------------------------------------------------
    # Saving / Loading
    # ------------------------------------------------------------
    def save_model(self, filepath):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "meta": {
                    "state_dim": self.state_dim,
                    "num_actions": self.num_actions,
                },
            },
            filepath,
        )

    def load_model(self, filepath, map_location=None):
        if map_location is None:
            map_location = "cpu"
        ckpt = torch.load(filepath, map_location=map_location)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
