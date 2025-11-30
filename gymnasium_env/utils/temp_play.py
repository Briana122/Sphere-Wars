import time
import os
import random
import torch

from gymnasium_env.envs.game_env import GameEnv
from gymnasium_env.agents import make_agent
from gymnasium_env.utils.action_utils import get_legal_actions


# -----------------------------------------
# CONFIG
# -----------------------------------------
MODEL_PATH = "C:\\Users\\byron\\Desktop\\COMP\\COMP4010\\Sphere-Wars\\trained_mcts.pt"     # path to your trained model
RENDER_SLEEP = 0.01                 # seconds between frames
MCTS_SIMULATIONS = 64               # reduce for faster gameplay


def load_mcts_agent(env, model_path):
    """
    Load MCTSNNAgent and restore its network weights.
    """
    agent = make_agent("mcts_nn", env.action_space, env.observation_space)
    agent.set_env(env)

    # override default search settings
    agent.num_simulations = MCTS_SIMULATIONS

    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=agent.device)
        agent.net.load_state_dict(state)
        print(f"[OK] Loaded MCTS-NN model: {model_path}")
    else:
        print(f"[WARN] No saved model found at '{model_path}'. Using untrained network.")

    return agent


def main():
    print("\n=== Running Game: MCTS-NN (Player 0) vs Random (Player 1) ===\n")

    # Build environment with rendering
    env = GameEnv(players=2, pieces_per=1, render_mode="human")
    obs, _ = env.reset()

    # Player 0 = MCTS
    agent0 = load_mcts_agent(env, MODEL_PATH)

    done = False

    while not done:
        env.render()
        time.sleep(RENDER_SLEEP)

        game = env.game
        current = game.current_player
        legal_actions = get_legal_actions(env)

        if not legal_actions:
            print(f"--- END TURN for Player {current} (no legal actions) ---")
            game.end_turn()
            continue

        # -------------------------------------------------------
        # Player 0: MCTS-NN agent
        # -------------------------------------------------------
        if current == 0:
            print("\n[Player 0] MCTS selecting action...")
            action = agent0.select_action(obs, legal_actions)

        # -------------------------------------------------------
        # Player 1: Random agent
        # -------------------------------------------------------
        else:
            print("\n[Player 1] Random selecting action...")
            action = random.choice(legal_actions)

        # Apply action to environment
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        time.sleep(RENDER_SLEEP)

        if terminated or truncated:
            print("\n=== GAME OVER ===")
            print("Winner:", env.game.winner)
            done = True
            break

        # End turn if next player has no legal moves
        next_actions = get_legal_actions(env)
        if not next_actions:
            game.end_turn()

    env.close()


if __name__ == "__main__":
    main()
