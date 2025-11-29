import random
import os
import torch

from gymnasium_env.envs.game_env import GameEnv
from gymnasium_env.agents.dqn.dqn_agent import DQNAgent
from gymnasium_env.agents.dqn.dqn_model import encode_observation, index_to_tuple
from gymnasium_env.agents.dqn.utils import make_legal_mask

MODEL_PATH = "gymnasium_env/agents/dqn/dqn_final_model.pt"

NUM_GAMES = 20
MAX_STEPS = 50 # optional

def evaluate_dqn_vs_random(model_path: str, num_games: int = NUM_GAMES):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating DQN model on {num_games} games using device: {device}")
    print(f"Model path: {model_path}")

    # Create a temporary env just to initialize the agent with correct spaces
    tmp_env = GameEnv(players=2, pieces_per=1, render_mode=None)
    agent = DQNAgent(tmp_env, device=device)
    # agent.load(model_path)

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(agent.policy_net.state_dict())


    tmp_env.close()

    dqn_wins = 0
    random_wins = 0
    draws = 0

    for game_idx in range(num_games):
        env = GameEnv(players=2, pieces_per=1, render_mode=None)
        obs, _ = env.reset()
        state = encode_observation(obs, env.players)
        done = False
        step_count = 0

        while not done and step_count < MAX_STEPS:
            current_player = env.game.current_player
            legal_mask = make_legal_mask(env)
            legal_actions = [i for i, valid in enumerate(legal_mask) if valid]

            if not legal_actions:
                env.game.end_turn()
                continue

            if current_player == 0:
                # DQN plays as Player 0
                with torch.no_grad():
                    action_index, action_tuple = agent.select_action(obs, legal_mask)
            else:
                # Random plays as Player 1
                action_tuple = random.choice(legal_actions)
                action_tuple = index_to_tuple(action_tuple, env.max_pieces_per_player, env.num_tiles)

            obs, reward, terminated, truncated, info = env.step(action_tuple)
            env.game.end_turn()
            state = encode_observation(obs, env.players)
            done = terminated or truncated
            step_count += 1

        # Game finished
        winner = env.game.winner
        print(f"Game {game_idx + 1}/{num_games} Winner: {winner}")

        if winner == 0:
            dqn_wins += 1
        elif winner == 1:
            random_wins += 1
        else:
            draws += 1

        env.close()

    total_games = dqn_wins + random_wins + draws
    print("\n=== Evaluation Results ===")
    print(f"Total games:  {total_games}")
    print(f"DQN Wins:     {dqn_wins}")
    print(f"Random Wins:  {random_wins}")
    print(f"Draws:        {draws}")
    if total_games > 0:
        print(f"DQN Winrate:  {dqn_wins / total_games:.3f}")


if __name__ == "__main__":
    evaluate_dqn_vs_random(MODEL_PATH, NUM_GAMES)