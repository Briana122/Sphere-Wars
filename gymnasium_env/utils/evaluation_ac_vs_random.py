import os
import random

from gymnasium_env.envs.game_env import GameEnv
from gymnasium_env.utils.action_utils import get_legal_actions
from gymnasium_env.agents.actor_critic.ac_agent import ActorCriticAgent
from gymnasium_env.utils.constants import SUBDIV

# Path of the pre-trained model you want to test

# MODEL_PATH = os.path.join(
#     "checkpoints",
#     "lr3e-4_ent0.02",
#     "lr3e-4_ent0.02_final.pt",
# )

# MODEL_PATH = r"checkpoints\base_lr3e-4_ent0.01\base_lr3e-4_ent0.01_final.pt"

MODEL_PATH = r"checkpoints\lr3e-4_ent0.005\lr3e-4_ent0.005_final.pt"

# Number of evaluation games
NUM_GAMES = 20   # increase to 100+ for more stable stats


def evaluate_model(model_path: str, num_games: int = NUM_GAMES):
    # Sanity check
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Evaluating model:\n  {model_path}")
    print(f"Number of games: {num_games}")

    # Create a temporary env just to initialize the agent with correct spaces
    tmp_env = GameEnv(players=2, pieces_per=1, subdiv=SUBDIV, render_mode=None)
    ac_agent = ActorCriticAgent(
        action_space=tmp_env.action_space,
        observation_space=tmp_env.observation_space,
    )
    ac_agent.load_model(model_path)
    tmp_env.close()

    ac_wins = 0
    random_wins = 0
    draws = 0 

    for game_idx in range(num_games):
        env = GameEnv(players=2, pieces_per=1, subdiv=SUBDIV, render_mode=None)
        obs, _ = env.reset()
        done = False

        while not done:
            game = env.game
            current_player = game.current_player

            legal_actions = get_legal_actions(env)
            if not legal_actions:
                # No legal moves: end this player's turn
                game.end_turn()
                continue

            if current_player == 0:
                # Actor-Critic agent plays as Player 0 (greedy evaluation)
                action, _ = ac_agent.select_action(
                    obs=obs,
                    legal_actions=legal_actions,
                    current_player=0,
                    greedy=True,
                    # greedy=False,
                )
            else:
                # Random policy as Player 1
                action = random.choice(legal_actions)

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # After acting, if current player has no moves, end their turn
            remaining_actions = get_legal_actions(env)
            if not remaining_actions:
                game.end_turn()

        # Game finished
        winner = env.game.winner
        print(f"Game {game_idx + 1}/{num_games}: Winner = {winner}")

        if winner == 0:
            ac_wins += 1
        elif winner == 1:
            random_wins += 1
        else:
            draws += 1

        env.close()

    total = ac_wins + random_wins + draws
    print("\n=== Evaluation Results ===")
    print(f"Total games:  {total}")
    print(f"AC Wins:      {ac_wins}")
    print(f"Random Wins:  {random_wins}")
    print(f"Draws:        {draws}")
    if total > 0:
        print(f"AC Winrate:   {ac_wins / total:.3f}")


if __name__ == "__main__":
    evaluate_model(MODEL_PATH, NUM_GAMES)
