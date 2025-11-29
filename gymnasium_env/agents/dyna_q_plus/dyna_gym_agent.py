# DYNAQ+ GYM WRAPPER
# agent.py is a generic Dyna-Q+ implementation that doesn't know much of anything about the game environment.
# This adaptor's sole purpose is to make DynaQPlusAgent compatible with the GameEnv and BaseAgent interface.
import numpy as np

from gymnasium_env.agents.base_agent import BaseAgent
from gymnasium_env.utils.constants import MAX_RESOURCES, MAX_STEPS
from .agent import DynaQPlusAgent


class DynaQPlusGymAgent(BaseAgent):
    '''
    Makes DynaQPlusAgent work with GameEnv and BaseAgent.
    ----------------------------------------------
    action_space and observation_space are from GameEnv.
    alpha, gamma, epsilon, plan_n, bonus_c & bonus_mode passed to DynaQPlusAgent.
    seed is for randomness
    ----------------------------------------------
    '''

    def __init__(
        self,
        action_space,
        observation_space,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.9,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.999,
        plan_n: int = 5,
        bonus_c: float = 0.01,
        bonus_mode: str = "sqrt",
        seed: int = 0,
    ):
        super().__init__(action_space, observation_space)

        # Get the information we need from the spaces
        self.num_tiles = observation_space["ownership"].shape[0]
        self.num_players = observation_space["resources"].shape[0]
        self.max_pieces = int(action_space[0].n)
        self.num_action_tiles = int(action_space[1].n)
        self.num_action_types = int(action_space[2].n)

        # Flattened action space size
        self.num_actions = (
            self.max_pieces
            * self.num_action_tiles
            * self.num_action_types
        )

        # Create the DynaQPlusAgent instance
        self.dyna = DynaQPlusAgent(
            action_dim=self.num_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            plan_n=plan_n,
            bonus_c=bonus_c,
            bonus_mode=bonus_mode,
            seed=seed,
        )

    # Get and set epsilon directly on the DynaQPlusAgent from the play_game script
    @property
    def epsilon(self) -> float:
        return self.dyna.epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self.dyna.epsilon = float(value)

    # State encoding: observation dict -> flat float32 vector
    def encode_state(self, obs):
        '''
        Encode the game environment observation dictionary into a flat float32 vector.

        Uses:
        - ownership (per tile)          in [-1, num_players-1], normalized
        - piece_owner (per tile)        same
        - resources (per player)        / MAX_RESOURCES
        - current_player (one-hot)      length = num_players
        - step_count                    / MAX_STEPS
        - tiles_to_win                  / num_tiles
        '''

        ownership = obs["ownership"].astype(np.float32)
        piece_owner = obs["piece_owner"].astype(np.float32)
        resources = obs["resources"].astype(np.float32)
        step_count = float(obs["step_count"])
        tiles_to_win = float(obs["tiles_to_win"])
        current_player = int(obs["current_player"])

        # Normalize ownership & piece_owner to roughly [-1, 1]
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

        return state_vec

    def action_to_index(self, action):
        '''
        Flatten (piece_idx, dest_tile, action_type) -> integer index.
        '''
        piece_idx, dest_tile, action_type = action
        return (
            piece_idx * (self.num_action_tiles * self.num_action_types)
            + dest_tile * self.num_action_types
            + action_type
        )

    def index_to_action(self, index):
        '''
        Inverse of action_to_index.
        '''
        base = self.num_action_tiles * self.num_action_types
        piece_idx = index // base
        rem = index % base
        dest_tile = rem // self.num_action_types
        action_type = rem % self.num_action_types
        return int(piece_idx), int(dest_tile), int(action_type)

    def build_legal_mask(self, legal_actions):
        '''
        Build a boolean mask over the flattened action space:
        True = legal, False = illegal.
        '''
        mask = np.zeros(self.num_actions, dtype=bool)
        for a in legal_actions:
            idx = self.action_to_index(a)
            if 0 <= idx < self.num_actions:
                mask[idx] = True
        return mask

    # Select action according to BaseAgent interface, and then map to action tuple
    def select_action(self, obs, legal_actions):
        '''
        Match BaseAgent interface:
            select_action(obs, legal_actions) -> action_tuple or None
        '''
        if not legal_actions:
            return None

        state_vec = self.encode_state(obs)
        legal_mask = self.build_legal_mask(legal_actions)

        # DynaQPlusAgent returns an integer index in [0, num_actions)
        action_index = self.dyna.select_action(state_vec, legal_mask)
        action_tuple = self.index_to_action(action_index)

        return action_tuple

    # Training interface
    def start_episode(self):
        self.dyna.start_episode()

    def end_episode(self):
        self.dyna.end_episode()

    def train_step(self, prev_obs, action, reward, next_obs, done, next_legal_actions):
        '''
        One Dyna-Q+ update from a real transition, plus planning.
        '''
        s_vec = self.encode_state(prev_obs)
        s_next_vec = self.encode_state(next_obs)
        a_index = self.action_to_index(action)

        if not next_legal_actions:
            # No future actions possible; force terminal
            self.dyna.step(
                s=s_vec,
                a=a_index,
                r=reward,
                s_next=s_next_vec,
                done=True,                # force terminal
                next_legal_mask=None,     # ignored since done=True
            )
            return

        next_mask = self.build_legal_mask(next_legal_actions)

        self.dyna.step(
            s=s_vec,
            a=a_index,
            r=reward,
            s_next=s_next_vec,
            done=done,
            next_legal_mask=next_mask,
        )

    # Save/Load Functions
    def save_model(self, filepath):
        self.dyna.save(filepath)

    def load_model(self, filepath):
        self.dyna.load(filepath)

    # Helper Functions
    def get_epsilon(self):
        return self.dyna.epsilon
