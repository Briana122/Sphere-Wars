import random
from .Piece import Piece # We need this to spawn a piece

# Player Class:
# This class is responsible for managing tiles, pieces and resources owned.
# The Game class will call the Player class for any movement, adding of pieces, or managing inventory
class Player:
    # ---------------------------------------------------------------------
    # Initialization & Editing Functions:
    # ---------------------------------------------------------------------
    def __init__(self, player_id):
        self.player_id = int(player_id)
        self.tiles_owned = []
        self.pieces_owned = []
        self.acqumilated_resources = 0
        self.total_moves = 0

    def add_tile(self, tile_id):
        if tile_id not in self.tiles_owned:
            self.tiles_owned.append(int(tile_id))

    def remove_tile(self, tile_id):
        if tile_id in self.tiles_owned:
            self.tiles_owned.remove(tile_id)

    def add_piece(self, piece_id):
        if piece_id not in self.pieces_owned:
            self.pieces_owned.append(int(piece_id))

    def remove_piece(self, piece_id):
        if piece_id in self.pieces_owned:
            self.pieces_owned.remove(piece_id)

    def gain_resources(self, amount, game):
        if amount <= 0:
            return
        game.resources[self.player_id] += int(amount)
        self.acqumilated_resources += int(amount)

    def spend_resources(self, amount, game):
        game.resources[self.player_id] -= int(amount)

    # ---------------------------------------------------------------------
    # Important Player Functions:
    # ---------------------------------------------------------------------

    # Take the piece, move it to the destination tile, and update the environment as well as self
    def move(self, piece, dest, game):
        # Just incase we've inserted the wrong piece
        if piece.agent != self.player_id:
            raise ValueError("This piece does not belong to this player.")

        # Clear old tile's piece reference
        old_tile = game.tiles[piece.tile_id]
        old_tile.piece = None

        target_tile = game.tiles[dest]

        # No attacking: we assume we never move onto enemy-owned tiles,
        # so target_tile.piece should be None or friendly.
        # If you ever see an enemy piece here, treat it as illegal move upstream.
        if (target_tile.piece is not None and
                target_tile.owner is not None and
                target_tile.owner != self.player_id):
            # Safety check; should not happen if legal_moves is correct
            # Undo and fail.
            old_tile.piece = (self.player_id, piece.pid)
            return False, False

        # Move the piece and update the tile to show occupancy
        piece.tile_id = dest
        target_tile.piece = (self.player_id, piece.pid)

        # Ownership & capture rewards
        captured_new_tile = False
        if target_tile.owner != self.player_id:
            if target_tile.owner is not None and target_tile.owner != self.player_id:
                prev_owner = target_tile.owner
                game.player_objs[prev_owner].remove_tile(target_tile.id)
                
            target_tile.owner = self.player_id
            self.add_tile(target_tile.id)
            self.gain_resources(target_tile.resources, game)
            # game.check_victory(self.player_id)
            captured_new_tile = True

        # Update total moves & game episode log
        self.total_moves += 1

        game.episode_log.append({
            "agent": self.player_id,
            "action": ("move", dest),
            "resources": dict(game.resources),
            "winner": getattr(game, "winner", None),
        })

        return True, captured_new_tile

    # Spawns a new piece on a random owned and unoccupied tile
    def spawn_piece(self, cost, game):
        if game.resources[self.player_id] < cost:
            return False
        
        # Get list of owned tiles with no piece on it 
        owned_free_tiles = [t for t in game.tiles.values() if t.owner == self.player_id and t.piece is None]
        # If no tiles are unoccupied, cannot spawn
        if not owned_free_tiles:
            return False

        # Choose random tile to spawn new piece
        tile = random.choice(owned_free_tiles)

        # Give this piece the next pid
        existing = [pid for (a, pid) in game.pieces.keys() if a == self.player_id]
        if not existing:
            pid = 0
        else:
            pid = max(existing) + 1

        # Create and register piece
        new_piece = Piece(self.player_id, pid, tile.id)
        game.pieces[(self.player_id, pid)] = new_piece
        tile.piece = (self.player_id, pid)

        # Spend the resources to 'buy' piece and add to inventory
        self.spend_resources(cost, game)
        self.add_piece(pid)

        # Update game episode log
        game.episode_log.append({
            "agent": self.player_id,
            "action": ("spawn", tile.id),
            "resources": dict(game.resources),
            "winner": getattr(game, "winner", None),
        })

        # Check for victory
        # game.check_victory(self.player_id)
        return True

    # ---------------------------------------------------------------------
    # Getter Functions:
    # ---------------------------------------------------------------------
    def get_num_tiles_owned(self):
        return len(self.tiles_owned)

    def get_num_pieces_owned(self):
        return len(self.pieces_owned)

    def get_acqumilated_resources(self):
        return int(self.acqumilated_resources)

    def get_current_resources(self, game):
        return int(game.resources.get(self.player_id, 0))
