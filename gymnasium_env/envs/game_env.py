import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

from ..game.Game import Game
from ..game.Piece import Piece
from ..game.Tile import Tile
from ..utils.training_game import Hexasphere

# UI Import
from ..game.UI import PlayerUI


class GameEnv(gym.Env):
    """
    A Gym environment for the custom Game class.
    This lets us interact with the game using Gym interfaces.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, players=2, pieces_per=1, render_mode=None, subdiv=2):
        super().__init__()
        self.game = None
        self.players = players
        self.pieces_per = pieces_per
        self.render_mode = render_mode
        self.subdiv = subdiv

        temp_hex = Hexasphere(subdiv=self.subdiv)
        self.num_tiles = len(temp_hex.tiles)

        self.action_space = spaces.Discrete(self.num_tiles)

        self.observation_space = spaces.Dict({
            "ownership": spaces.Box(low=-1, high=players, shape=(self.num_tiles,), dtype=np.int32),
            "resources": spaces.Box(low=0, high=1000, shape=(self.players,), dtype=np.int32)
        })

        # rendering setup
        if self.render_mode == "human":
            pygame.init()
            self.W, self.H = 800, 800
            self.screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("Game Environment")

            # camera control state
            self.cam_pitch = 0.0
            self.cam_yaw = 0.0 
            self.zoom = 1.0
            self.dragging = False
            self.last_pos = None

            self.clock = pygame.time.Clock()

            # This array contains a PlayerUI for each player.
            self.stats_ui = [PlayerUI(player_index=p, font_size=18) for p in range(self.players)]

    def reset(self, seed=None, options=None):
        """Start a new game and return the initial observation."""
        hex_map = Hexasphere(subdiv=self.subdiv)
        self.game = Game(hex_map, players=self.players, pieces_per=self.pieces_per)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """Execute one environment step."""
        piece_id, dest = action
        piece = self.game.pieces[piece_id]
        self.game.move(piece, dest)

        reward = 0
        terminated = self.game.winner is not None
        truncated = False

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the current game state."""
        if self.render_mode != "human":
            return

        from ..game.visual_game import render

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                exit()

            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1: 
                    self.dragging = True
                    self.last_pos = e.pos
                elif e.button == 4: 
                    self.zoom = min(self.zoom * 1.1, 5.0)
                elif e.button == 5: 
                    self.zoom = max(self.zoom * 0.9, 0.3)

            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1:
                    self.dragging = False
                    self.last_pos = None

            elif e.type == pygame.MOUSEMOTION and self.dragging:
                dx, dy = e.rel
                self.cam_yaw += dx * 0.01
                self.cam_pitch += dy * 0.01
                self.cam_pitch = max(-math.pi/2, min(math.pi/2, self.cam_pitch))

        render(self.screen, self.game, self.W, self.H, rx=self.cam_pitch, ry=self.cam_yaw, zoom=self.zoom)

        # Draw the ui for each player in the stats_ui array.
        for playerUI in self.stats_ui:
            playerUI.draw_player_stats(self.screen, self.game)

        self.clock.tick(self.metadata["render_fps"])

        # Update the display
        # Note: There was a pygame.display.flip() in visual_game.py's render function.
        # However, I think it's best to call pygame.display.update() here after doing everything that needs to be done.
        # Otherwise I'd have to add all the player UI drawing stuff in visual_game.py which i don't know if that's the best idea.
        # But we can always change it later if needed.
        pygame.display.update()
        

    def _get_obs(self):
        """Return current observation of game."""
        ownership = np.array([
            self.game.tiles[t].owner if self.game.tiles[t].owner is not None else -1
            for t in self.game.tiles
        ], dtype=np.int32)

        resources = np.array([
            self.game.resources[p] for p in range(self.players)
        ], dtype=np.int32)

        return {"ownership": ownership, "resources": resources}

    def _make_hex(self):
        return Hexasphere(subdiv=3)
