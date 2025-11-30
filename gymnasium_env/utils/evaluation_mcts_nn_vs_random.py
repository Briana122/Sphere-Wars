import os
import random
import numpy as np
import torch

from gymnasium_env.envs.game_env import GameEnv
from gymnasium_env.agents import make_agent
from gymnasium_env.utils.action_utils import get_legal_actions


# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "C:\\Users\\byron\\Desktop\\COMP\\COMP4010\\Sphere-Wars\\trained_mcts.pt"     # path to your trained model
NUM_GAMES = 25
MAX_STEPS = 2000
SUBDIV_VALUE = 2                     # board size


# ============================================================
# Helper: Run MCTS selection manually (argmax or sample)
# ============================================================
def select_mcts_action(agent, env, obs, deterministic=True):
    """
    Runs MCTS from the current state and selects an action.
    deterministic=True  -> picks argmax of visit counts
    deterministic=False -> samples from visit-count distribution
    """
    agent.set_env(env)   # ensure MCTS has correct root


    legal_actions = get_legal_actions(env)
    if not legal_actions:
        return None

    # Run MCTS search (no temperature)
    visit_counts, _ = agent._run_mcts(obs, env.game, legal_actions)

    # Convert visit-count dict to a probability vector
    counts_vec = np.zeros(agent.action_size, dtype=np.float32)
    total = sum(visit_counts.values())

    if total == 0:
        return random.choice(legal_actions)

    for a, n in visit_counts.items():
        idx = agent.action_to_index(a)
        counts_vec[idx] = n

    probs = counts_vec / total

    if deterministic:
        action_idx = int(np.argmax(probs))
    else:
        action_idx = np.random.choice(agent.action_size, p=probs)

    return agent.index_to_action(action_idx)


# ============================================================
# Core Evaluation Loop
# ============================================================
def evaluate_mcts_vs_random(
    model_path: str,
    num_games: int = NUM_GAMES,
    subdiv: int = SUBDIV_VALUE,
    deterministic: bool = True
):
    """
    Evaluate *new* MCTSNNAgent (Player 0) vs Random (Player 1).
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("\n============================================")
    print("Evaluating MCTSNNAgent vs Random Agent")
    print(f"Model: {model_path}")
    print(f"Games: {num_games}")
    print(f"Subdivisions: {subdiv}")
    print(f"MCTS deterministic mode: {deterministic}")
    print("============================================\n")

    # Create temporary env to load agent
    tmp_env = GameEnv(players=2, pieces_per=1, subdiv=subdiv, render_mode=None)
    agent = make_agent("mcts_nn", tmp_env.action_space, tmp_env.observation_space)
    agent.set_env(tmp_env)
    agent.num_simulations = 128  

    
    # Load trained neural network weights
    agent.load_model(model_path)
    tmp_env.close()

    mcts_wins = 0
    random_wins = 0
    draws = 0

    # ===================================================
    #   Play Games
    # ===================================================
    for game_idx in range(num_games):

        env = GameEnv(players=2, pieces_per=1, subdiv=subdiv, render_mode=None)
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:

            current_player = env.game.current_player
            legal_actions = get_legal_actions(env)

            if not legal_actions:
                env.game.end_turn()
                continue

            # ------------------------------
            #   Player 0 = MCTSNNAgent
            # ------------------------------
            if current_player == 0:
                action = select_mcts_action(
                    agent=agent,
                    env=env,
                    obs=obs,
                    deterministic=deterministic
                )

            # ------------------------------
            #   Player 1 = Random Agent
            # ------------------------------
            else:
                action = random.choice(legal_actions)

            if action is None:
                env.game.end_turn()
                continue

            # Apply action
            obs, reward, terminated, truncated, info = env.step(action)

            # End turn if the next player cannot act
            if not get_legal_actions(env):
                env.game.end_turn()

            done = terminated or truncated
            steps += 1

        winner = env.game.winner
        print(f"Game {game_idx + 1}/{num_games}: Winner = {winner}")

        if winner == 0:
            mcts_wins += 1
        elif winner == 1:
            random_wins += 1
        else:
            draws += 1

        env.close()

    # ===================================================
    #   Final Statistics
    # ===================================================
    total = mcts_wins + random_wins + draws

    print("\n================ Evaluation Results ================")
    print(f"Total games:        {total}")
    print(f"MCTS Wins:          {mcts_wins}")
    print(f"Random Wins:        {random_wins}")
    print(f"Draws:              {draws}")

    if total > 0:
        print(f"MCTS Winrate:       {mcts_wins / total:.3f}")
    print("===================================================\n")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    evaluate_mcts_vs_random(
        model_path=MODEL_PATH,
        num_games=NUM_GAMES,
        subdiv=SUBDIV_VALUE,
        deterministic=True
    )
