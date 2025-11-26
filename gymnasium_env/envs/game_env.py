import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

from gymnasium_env.utils.constants import MAX_RESOURCES, MAX_STEPS, SPAWN_COST, CAPTURE_REWARD, SUBDIV, WIN_REWARD

from ..game.Game import Game
from ..game.Piece import Piece
from ..utils.game_board import Hexasphere

# UI Import
from ..game.PlayerUI import PlayerUI


class GameEnv(gym.Env):
    """
    A Gym environment for the custom Game class.
    This lets us interact with the game using Gym interfaces.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, players=2, pieces_per=1, render_mode=None, subdiv=SUBDIV, max_steps=MAX_STEPS):
        super().__init__()
        self.game = None
        self.players = players
        self.pieces_per = pieces_per
        self.render_mode = render_mode
        self.subdiv = subdiv
        self.max_steps = max_steps

        temp_hex = Hexasphere(subdiv=self.subdiv)
        self.num_tiles = len(temp_hex.tiles)
        self.max_pieces_per_player = math.floor(self.num_tiles / 4)

        self.tiles_to_win = int(math.ceil(self.num_tiles / 2))
        # print(f"Goal is {self.tiles_to_win} tiles, max pieces is {self.max_pieces_per_player}")

        self.action_space = spaces.Tuple((
            spaces.Discrete(self.max_pieces_per_player),
            spaces.Discrete(self.num_tiles),
            spaces.Discrete(2)
        ))

        self.observation_space = spaces.Dict({
            # which player owns which tile
            "ownership": spaces.Box(
                low=-1, high=self.players-1,
                shape=(self.num_tiles,), dtype=np.int8
            ),
            # where each piece is located
            "piece_owner": spaces.Box(
                low=-1, high=self.players-1,
                shape=(self.num_tiles,), dtype=np.int8
            ),
            # number of resources per player
            "resources": spaces.Box(
                low=0, high=MAX_RESOURCES,
                shape=(self.players,), dtype=np.int32
            ),
            # current player
            "current_player": spaces.Discrete(self.players),
            # step number
            "step_count": spaces.Box(
                low=0, high=MAX_STEPS,  # arbitrary cap; you can adjust
                shape=(), dtype=np.int32
            ),
            # number of tiles needed to win
            "tiles_to_win": spaces.Box(
                low=0, high=self.num_tiles,
                shape=(), dtype=np.int32
            ),
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
        self.game.current_player = np.random.randint(self.players)
        self.step_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """Take an action = (piece_id, dest_tile, action_type)."""
        reward = -0.1
        self.step_count += 1 
        terminated = truncated = False

        if self.step_count >= (self.max_steps - 1) and not terminated and not truncated:
            truncated = True

        # Unpack action
        pid, dest, action_type = action
        # Get selected piece
        piece = self.game.pieces[(self.game.current_player, pid)]
        
        # Illegal action: selected piece != player's piece
        if piece.agent != self.game.current_player:
            truncated = True
            info = truncated, {"illegal": True}
            print("here2")
            print("piece agent:", piece.agent)
            print("pid", piece.pid)
            print(self.game.current_player)
            return self._get_obs(), reward, terminated, info

        # Identify piece selected by agent that owns the piece and the piece id
        self.game.selected = (piece.agent, pid)
        captured_new_tile = False
        spawned = False

        if action_type == 1:
            # SPAWN
            if self.game.resources[piece.agent] >= SPAWN_COST:
                spawned = self.game.spawn_piece(piece.agent, cost=SPAWN_COST)
            if not spawned:
                # If spawn fails, fall back to MOVE as per your original logic
                moved, captured_new_tile = self._apply_move(piece, dest)
            # else:
                # print(f"Agent: {piece.agent} \t Action: Spawn \t\t Resources: {self.game.resources[piece.agent]} ", end="")
        else:
            # MOVE
            moved, captured_new_tile = self._apply_move(piece, dest)

        if captured_new_tile:
            reward += CAPTURE_REWARD

        # print(f"\t\tReward: {reward} \t ", end="")
        # if spawned or moved:
        #     print(f"Successful Action")

        # ---------------------- Check victory ----------------------

        if self.game.check_victory(last_agent=piece.agent):
            terminated = True
            reward += WIN_REWARD

        obs = self._get_obs()
        # info = {"piece_key": piece_key}
        info = {"piece_key": (piece.agent, pid)}

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
        pygame.display.update()
        

    def _apply_move(self, piece, dest):
        """
        Helper to apply a move and report whether we captured a new tile.
        Returns (moved: bool, captured_new_tile: bool).
        """
        # Basic validity: dest must be inside board
        if dest < 0 or dest >= len(self.game.tiles):
            return False, False

        legal = self.game.legal_moves(piece)
        if dest not in legal:
            return False, False

        moved, captured_new_tile = self.game.move(piece, dest)

        # print(f"Agent: {piece.agent} \t Action: Move \t\t Resources: {self.game.resources[piece.agent]} ", end="")

        return moved, captured_new_tile

    def _get_obs(self):
        ownership = np.full((self.num_tiles,), -1, dtype=np.int8)
        piece_owner = np.full((self.num_tiles,), -1, dtype=np.int8)

        for tid, tile in self.game.tiles.items():
            if tile.owner is not None:
                ownership[tid] = tile.owner
            if tile.piece is not None:
                piece_owner[tid] = tile.piece[0]

        resources = np.array(
            [self.game.resources[p] for p in range(self.players)],
            dtype=np.int32
        )

        return {
            "ownership": ownership,
            "piece_owner": piece_owner,
            "resources": resources,
            "current_player": self.game.current_player,
            "step_count": np.int32(self.step_count),
            "tiles_to_win": np.int32(self.tiles_to_win),
        }