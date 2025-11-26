# from gymnasium_env.utils.constants import SPAWN_COST

# def get_legal_actions(env):
#     game = env.game
#     current = game.current_player
#     legal = []

#     can_spawn = game.resources[current] >= SPAWN_COST

#     # iterate only over existing pieces owned by current player
#     for (agent, pid), piece in game.pieces.items():
#         if agent != current:
#             continue
#         if game.moved_flags[agent].get(pid, False):
#             continue

#         piece_id = env.encode_piece_id(agent, pid)

#         moves = game.legal_moves(piece)
#         for dest in moves:
#             legal.append((piece_id, dest, 0))

#         if can_spawn:
#             legal.append((piece_id, piece.tile_id, 1))

#     return legal

from gymnasium_env.utils.constants import SPAWN_COST


def get_legal_actions(env):
    game = env.game
    current = game.current_player
    legal = []

    can_spawn = game.resources[current] >= SPAWN_COST

    all_keys = list(game.pieces.keys())
    # all_keys = sorted(game.pieces.keys())

    for idx, (agent, pid) in enumerate(all_keys):
        if agent != current:
            # print("here")
            continue
        if game.moved_flags[current].get(pid, False):
            continue

        piece = game.pieces[(agent, pid)]
        moves = game.legal_moves(piece)

        # print("agent", agent)
        # print("PID")
        for dest in moves:
            # print((pid, dest, 0))
            # legal.append((idx, dest, 0))
            legal.append((pid, dest, 0))

        if can_spawn:
            # legal.append((idx, 0, 1))
            # legal.append((idx, piece.tile_id, 1))
            legal.append((pid, piece.tile_id, 1))
            # legal.append((idx, None, 1))

    return legal
