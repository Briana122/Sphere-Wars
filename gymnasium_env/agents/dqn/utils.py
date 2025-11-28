import numpy as np
from .dqn_model import tuple_to_index

def make_legal_mask(env):
    max_pieces = env.max_pieces_per_player
    num_tiles = env.num_tiles

    action_dim = max_pieces * env.num_tiles * 2
    mask = np.zeros(action_dim, dtype=bool)

    piece_keys = sorted(list(env.game.pieces.keys()))

    current_player = env.game.current_player
    resources = env.game.resources[current_player]

    for piece_key in piece_keys:
        piece = env.game.pieces[piece_key]
        if piece.agent != current_player:
            continue
        piece_id = piece_key[1]
        if piece_id >= max_pieces:
            continue

        ## MOVE ACTIONS ##
        legal_moves = env.game.legal_moves(piece)

        for dest in legal_moves:
            if dest == piece.tile_id: # skip any idle moves
                continue
            if not (0 <= dest < num_tiles): # invalid tile
                continue

            index = tuple_to_index((piece_id, dest, 0), max_pieces, num_tiles)
            
            # mask[index] = True
            if 0 <= index < action_dim:
                mask[index] = True

        ## SPAWN ACTIONS ##
        SPAWN_COST = 10
        if resources >= SPAWN_COST:
            index = tuple_to_index((piece_id, piece.tile_id, 1), max_pieces, num_tiles)

            # mask[index] = True
            if 0 <= index < action_dim:
                mask[index] = True

    return mask
