import random
from .Tile import Tile
from .Piece import Piece
from .Player import Player

class Game:
    # the main game class, contains all logic for turns, moving pieces, spawning pieces, and checking victory conditions
    # also stores a log of state transitions for RL training purposes   
    # can be scaled up for more players very easily, just need to implement proper turn ordering
    def __init__(self, hexs, players=2, pieces_per=1):
        self.hex, self.players = hexs, players
        self.tiles = {vi: Tile(vi,t["center"]) for vi,t in hexs.tiles.items()}
        self.resources = {p:0 for p in range(players)}
        self.pieces = {}
        self.player_objs = [Player(p) for p in range(players)]  # concrete Player objects

        # create first piece per player
        for p in range(players):
            for pid in range(pieces_per):
                free = [t for t in self.tiles if self.tiles[t].piece is None]
                start = random.choice(free)
                piece = Piece(p,pid,start)
                self.pieces[(p,pid)] = piece
                self.tiles[start].piece = (p,pid)
                self.tiles[start].owner = p
                self.player_objs[p].add_piece(pid)
                self.player_objs[p].add_tile(start)
                self.resources[p]+=self.tiles[start].resources
                self.player_objs[p].acqumilated_resources += self.tiles[start].resources

        self.current_player = 0
        self.selected = None
        self.rot = [0,0]
        self.winner = None
        self.episode_log = []  # store transitions for RL

    def legal_moves(self, piece):
        # return list of legal tile ids the piece can move to, will need to be fine tuned
        # currently can only move to tile already owned or unoccupied tiles
        current_id = piece.tile_id
        neighbors = self.hex.tiles[current_id]["neighbors"]
        legal = [current_id]
        for n in neighbors:
            tile = self.tiles[n]
            if tile.owner is None or tile.owner == piece.agent:
                legal.append(n)
        return legal

    def move(self, piece, dest):
        if dest not in self.legal_moves(piece): 
            return
        return self.player_objs[piece.agent].move(piece, dest, self)

    def spawn_piece(self, agent, cost):
        return self.player_objs[agent].spawn_piece(cost, self)

    def check_victory(self, last_agent=None):
        # basic victory check of if the agent owns half or more of the tiles
        total = len(self.tiles)
        half = total / 2.0
        counts = {p: 0 for p in range(self.players)}
        for t in self.tiles.values():
            if t.owner is not None:
                counts[t.owner] += 1
        for p, cnt in counts.items():
            if cnt >= half:
                self.winner = p
                return True
        return False
