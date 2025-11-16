import random
import time
from gymnasium_env.envs.game_env import GameEnv

def main():
    env = GameEnv(players=2, pieces_per=1, render_mode="human")
    obs, _ = env.reset()

    done = False

    while not done:
        game = env.game
        current = game.current_player

        piece_keys = [
            key for key in game.pieces.keys()
            if key[0] == current and not game.moved_flags[current].get(key[1], False)
        ]

        if len(piece_keys) == 0:
            game.end_turn()
            continue

        agent, pid = random.choice(piece_keys)
        piece = game.pieces[(agent, pid)]

        legal_moves = game.legal_moves(piece)

        if not legal_moves:
            game.moved_flags[agent][pid] = True
            continue

        cost = 10
        can_spawn = (game.resources[current] >= cost)

        if can_spawn:
            action_type = 1
            dest = 0  
        else:
            action_type = 0
            dest = random.choice(legal_moves)


        all_piece_keys = list(game.pieces.keys())
        piece_index = all_piece_keys.index((agent, pid))

        action = (piece_index, dest, action_type)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)

        if terminated or truncated:
            print("Game Over! Winner:", game.winner)
            done = True
            break

    env.close()

if __name__ == "__main__":
    main()
