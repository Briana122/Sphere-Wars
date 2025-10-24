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
        # create first piece per player
        for p in range(players):
            for pid in range(pieces_per):
                free = [t for t in self.tiles if self.tiles[t].piece is None]
                start = random.choice(free)
                piece = Piece(p,pid,start)
                self.pieces[(p,pid)] = piece
                self.tiles[start].piece = (p,pid)
                self.tiles[start].owner = p
                self.resources[p]+=self.tiles[start].resources

        self.current_player = 0
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
        old = self.tiles[piece.tile_id]
        old.piece = None
        piece.tile_id = dest
        target = self.tiles[dest]
        if target.piece and target.piece[0] != piece.agent:
            del self.pieces[target.piece]
        target.piece = (piece.agent, piece.pid)
        if target.owner != piece.agent:
            target.owner = piece.agent
            self.resources[piece.agent] += target.resources
            self.check_victory(piece.agent)
        # store state transition for RL
        self.episode_log.append({
            "agent": piece.agent,
            "action": dest,
            "resources": dict(self.resources),
            "winner": self.winner
        })

    def spawn_piece(self, agent, cost):
        # some basic logic for spawning new pieces, can only spawn on owned tiles that are unoccupied
        owned_free = [t for t in self.tiles.values() if t.owner == agent and t.piece is None]
        if not owned_free:
            return False
        tile = random.choice(owned_free)
        existing = [pid for (a, pid) in self.pieces if a == agent]
        pid = max(existing, default=-1) + 1
        new_piece = Piece(agent, pid, tile.id)
        self.pieces[(agent, pid)] = new_piece
        tile.piece = (agent, pid)
        self.resources[agent] -= cost
        self.check_victory(agent)
        return True

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
