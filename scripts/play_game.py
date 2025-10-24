import random
import time
from gymnasium_env.envs.game_env import GameEnv

env = GameEnv(players=2, pieces_per=1, render_mode="human")
obs, _ = env.reset()

done = False

# Change based on board size
tile_count = 100

piece_keys = list(env.game.pieces.keys())
print("Valid piece IDs:", piece_keys)

while not done:
    # Keep updated list of pieces
    piece_keys = list(env.game.pieces.keys())
    for piece_index, piece_id in enumerate(piece_keys):
        piece = env.game.pieces[piece_id]

        legal_moves = env.game.legal_moves(piece) # follow the game rules
        if not legal_moves:
            continue

        # Chooses a random move
        dest = random.choice(legal_moves)
        # action = (piece_id, dest)

        # # Policy: spawn if enough resources, else move
        if env.game.resources[piece.agent] >= 10:
            action_type = 1  # spawn
        else:
            action_type = 0  # move

        # action_type = random.choice([0, 1])

        action = (piece_index, dest, action_type)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1) # my eyes!

        if terminated or truncated:
            done = True
            break

env.close()
