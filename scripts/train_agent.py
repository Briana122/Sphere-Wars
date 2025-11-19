import argparse
import os
import time
import numpy as np

from gymnasium_env.envs.game_env import GameEnv
from gymnasium_env.agents import make_agent
from gymnasium_env.utils.action_utils import get_legal_actions

def train(agent_type="random", episodes=500, max_steps_per_episode=10000,
          save_path="models/random.pth", players=2, pieces_per=1):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = GameEnv(players=players, pieces_per=pieces_per, render_mode=None)
    obs, _ = env.reset()

    agent = make_agent(
        agent_type,
        env.action_space,
        env.observation_space,
        player_id=0
    )

    print("Training agent:", agent_type)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps_per_episode:

            legal_actions = get_legal_actions(env)
            game = env.game

            if not legal_actions:
                game.end_turn()
                continue

            action = agent.select_action(obs, legal_actions)
            if action is None:
                action = legal_actions[0]

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_legal = get_legal_actions(env)

            transition = {
                "state": obs,
                "action": action,
                "reward": reward,
                "next_state": next_obs,
                "done": done,
                "legal_next_actions": next_legal
            }

            agent.train(transition)

            obs = next_obs
            steps += 1

            if not get_legal_actions(env):
                game.end_turn()

        print(f"Episode {ep} | Reward total: {total_reward}")

        if ep % 100 == 0 or ep == episodes:
            agent.save_model(save_path)
            print("Model saved:", save_path)

    env.close()
    print("Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-type", default="qfa")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--save-path", default="models/qfa_agent.pth")
    args = parser.parse_args()

    train(agent_type=args.agent_type,
          episodes=args.episodes,
          save_path=args.save_path)


if __name__ == "__main__":
    main()
