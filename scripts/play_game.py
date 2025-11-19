import argparse
import random
import time
import os
from gymnasium_env.envs.game_env import GameEnv
from gymnasium_env.agents import make_agent
from gymnasium_env.utils.action_utils import get_legal_actions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-type", default="random")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sleep", type=float, default=0.02)
    args = parser.parse_args()

    env = GameEnv(render_mode="human")
    obs, _ = env.reset()

    agent = make_agent(args.agent_type, env.action_space, env.observation_space)

    if args.model_path and os.path.exists(args.model_path):
        agent.load_model(args.model_path)
        print("Loaded model:", args.model_path)

    done = False

    while not done:
        legal = get_legal_actions(env)

        if not legal:
            env.game.end_turn()
            continue

        action = agent.select_action(obs, legal)
        if action is None:
            action = random.choice(legal)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(args.sleep)

        if terminated or truncated:
            print("Winner:", env.game.winner)
            done = True

        if not get_legal_actions(env):
            env.game.end_turn()

    env.close()

if __name__ == "__main__":
    main()
