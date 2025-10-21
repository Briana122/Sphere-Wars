import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..game.Game import Game
from ..game.Piece import Piece
from ..game.Tile import Tile


class GameEnv(gym.Env):
    """
    A Gym environment for the custom Game class.
    This lets us to interact with the game using Gym interfaces.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, players=2, pieces_per=1, render_mode=None):
        super().__init__()
        self.game = None
        self.players = players
        self.pieces_per = pieces_per
        self.render_mode = render_mode

        # Change based on # of tiles!!!
        self.action_space = spaces.Discrete(100)

        self.observation_space = spaces.Dict({ # for each player
            "ownership": spaces.Box(low=0, high=players, shape=(100,), dtype=np.int32),
            "resources": spaces.Box(low=0, high=1000, shape=(players,), dtype=np.int32)
        })

        # Stops the rendering looking glitchy
        if self.render_mode == "human":
            import pygame
            pygame.init()
            self.W, self.H = 800, 800
            self.screen = pygame.display.set_mode((self.W, self.H))

    def reset(self, seed=None, options=None):
        """Start a new game and return the initial observation."""

        self.game = Game(self._make_hex(), players=self.players, pieces_per=self.pieces_per)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """
        Execute one step in env.
        """
        piece_id, dest = action
        piece = self.game.pieces[piece_id]

        self.game.move(piece, dest)

        # TODO: define our reward function
        reward = 0

        # See if game ended
        terminated = self.game.winner is not None
        truncated = False

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render current game state
        Note: only works in human mode
        """

        if self.render_mode == "human":
            import pygame
            from ..game.visual_game import render
            render(self.screen, self.game, self.W, self.H)
            pygame.display.flip()

    def _get_obs(self):
        """Return current observation of game"""

        ownership = np.array([
            self.game.tiles[t].owner if self.game.tiles[t].owner is not None else -1
            for t in self.game.tiles
        ], dtype=np.int32)

        resources = np.array([
            self.game.resources[p] for p in range(self.players)
        ], dtype=np.int32)

        return {"ownership": ownership, "resources": resources}

    def _make_hex(self):
        from ..utils.training_game import Hexasphere
        return Hexasphere(subdiv=3)
