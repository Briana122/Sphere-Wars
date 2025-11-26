from gymnasium_env.utils.constants import SPAWN_COST


def get_legal_actions(env):
    game = env.game
    current = game.current_player
    legal = []

    can_spawn = game.resources[current] >= SPAWN_COST

    all_keys = list(game.pieces.keys())

    for idx, (agent, pid) in enumerate(all_keys):
        if agent != current:
            continue
        if game.moved_flags[current].get(pid, False):
            continue

        piece = game.pieces[(agent, pid)]
        moves = game.legal_moves(piece)

        for dest in moves:
            legal.append((pid, dest, 0))

        if can_spawn:
            legal.append((pid, piece.tile_id, 1))

    return legal
