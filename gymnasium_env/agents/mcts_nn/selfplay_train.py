import argparse
import os
import numpy as np
import random
import torch

from gymnasium_env.envs.game_env import GameEnv
from gymnasium_env.agents import make_agent
from gymnasium_env.utils.action_utils import get_legal_actions


def run_self_play_game(env, agent, temperature=1.0):
    """
    Runs one full self-play game with a single agent controlling all players.
    Returns list of (obs, policy, player), and the game winner.
    """
    obs, _ = env.reset()
    game = env.game

    trajectory = []  # (obs, policy_vector, player)
    done = False

    while not done:
        legal_actions = get_legal_actions(env)

        if not legal_actions:
            # turn over
            game.end_turn()
            continue

        # Run MCTS, get visit-count distribution
        root_game_copy = env.game
        visit_counts, chosen_action = agent._run_mcts(obs, root_game_copy, legal_actions)

        # Convert to a normalized policy vector over ALL actions
        policy_vec = np.zeros(agent.action_size, dtype=np.float32)
        total = sum(visit_counts.values())
        for a, n in visit_counts.items():
            idx = agent.action_to_index(a)
            policy_vec[idx] = n / total

        # Apply temperature (optional)
        if temperature > 0:
            p = policy_vec ** (1 / temperature)
            p /= p.sum()
            action_idx = np.random.choice(agent.action_size, p=p)
            chosen_action = agent.index_to_action(action_idx)

        # Store transition
        trajectory.append((obs, policy_vec, game.current_player))

        # Play move
        obs, reward, terminated, truncated, info = env.step(chosen_action)

        if terminated or truncated:
            done = True
            break

        # After acting, if no moves remain, end turn
        if not get_legal_actions(env):
            game.end_turn()

    winner = env.game.winner
    return trajectory, winner


def main():
    parser = argparse.ArgumentParser(description="Self-play training loop for MCTS-NN agent.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--save-path", type=str, default="mcts_nn_model.pt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--pieces-per", type=int, default=1)

    args = parser.parse_args()

    # Environment without rendering for training
    env = GameEnv(players=args.players, pieces_per=args.pieces_per, render_mode=None)
    obs, _ = env.reset()

    # Single agent for all players
    agent = make_agent("mcts_nn", env.action_space, env.observation_space)
    agent.set_env(env)

    print("=== Starting Self-Play Training ===")

    replay_obs = []
    replay_policy = []
    replay_value = []

    for episode in range(args.episodes):
        trajectory, winner = run_self_play_game(env, agent, temperature=args.temperature)

        # Winner-based target values
        for obs, policy, player in trajectory:
            if winner is None:
                v = 0.0
            else:
                v = 1.0 if winner == player else -1.0

            replay_obs.append(obs)
            replay_policy.append(policy)
            replay_value.append(v)

        # Train after each episode
        batch = {
            "obs": replay_obs,
            "policy_target": replay_policy,
            "value_target": replay_value,
        }

        stats = agent.train(batch)
        print(f"[Episode {episode+1}/{args.episodes}] "
              f"Loss={stats['loss']:.4f}  "
              f"Value={stats['value_loss']:.4f}  "
              f"Policy={stats['policy_loss']:.4f}")

    # Save trained model
    torch.save(agent.net.state_dict(), args.save_path)
    print(f"Saved trained MCTS-NN model to {args.save_path}")


if __name__ == "__main__":
    main()
