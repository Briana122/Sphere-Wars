import argparse
import random
import time
import os
from gymnasium_env.envs.game_env import GameEnv
from gymnasium_env.agents.actor_critic.ac_agent import ActorCriticAgent
from gymnasium_env.agents import make_agent
from gymnasium_env.utils.action_utils import get_legal_actions

def main():
    parser = argparse.ArgumentParser(description="Play a game in the Hexasphere environment.")
    parser.add_argument("--agent0-type", default="random", help="Agent type for player 0.")
    parser.add_argument("--agent1-type", default="random", help="Agent type for player 1.")
    parser.add_argument("--agent0-model", default=None, help="Path to saved pre-trained model for player 0 (optional).")
    parser.add_argument("--agent1-model", default=None, help="Path to saved pre-traained model for player 1 (optional).")
    parser.add_argument("--players", type=int, default=2, help="Number of players (currently only support 2).")
    parser.add_argument("--pieces-per", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.02, help="Seconds to sleep between frames for rendering.")
    args = parser.parse_args()

    # Create environment with rendering enabled
    env = GameEnv(players=args.players, pieces_per=args.pieces_per, render_mode="human")
    obs, _ = env.reset()

    # Create one agent per player
    agent0 = make_agent(args.agent0_type, env.action_space, env.observation_space)
    agent1 = make_agent(args.agent1_type, env.action_space, env.observation_space)

    # Load models if provided
    if args.agent0_model and os.path.exists(args.agent0_model):
        try:
            agent0.load_model(args.agent0_model)
            print("Player 0 loaded model:", args.agent0_model)
        except Exception as e:
            print("Failed to load model for player 0:", e)

    if args.agent1_model and os.path.exists(args.agent1_model):
        try:
            agent1.load_model(args.agent1_model)
            print("Player 1 loaded model:", args.agent1_model)
        except Exception as e:
            print("Failed to load model for player 1:", e)

    # If agents have epsilon (like Q-learning FA), use greedy play:
    if hasattr(agent0, "epsilon"):
        agent0.epsilon = 0.0
    if hasattr(agent1, "epsilon"):
        agent1.epsilon = 0.0

    done = False

    print("Starting game. Close the window to exit.")

    while not done:
        game = env.game
        current = game.current_player

        legal_actions = get_legal_actions(env)

        if not legal_actions:
            # No legal moves for this player, end their turn
            print(f"--- END TURN for player {game.current_player} ---")
            game.end_turn()
            continue

        # Choose which agent acts based on current player
        if current == 0:
            acting_agent = agent0
        elif current == 1:
            acting_agent = agent1
        # Fallback for now or if we ever go beyond 2 players (or raise an error)
        else:
            acting_agent = agent0 

        # Let the agent pick an action from the legal set
        if isinstance(acting_agent, ActorCriticAgent):
            # ActorCriticAgent expects (obs, legal_actions, current_player, greedy)
            # and returns (action, info)
            action, _info = acting_agent.select_action(
                obs=obs,
                legal_actions=legal_actions,
                current_player=current,
                greedy=True,  # use greedy policy for evaluation
            )
        else:
            # Other agents use the simpler interface and return just the action
            action = acting_agent.select_action(obs, legal_actions)
            
        if action is None:
            action = random.choice(legal_actions)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(args.sleep)

        if terminated or truncated:
            print("Game Over! Winner:", env.game.winner)
            done = True
            break

        # After acting, if this player has no more moves, end their turn
        remaining_actions = get_legal_actions(env)
        if not remaining_actions:
            print(f"--- END TURN for player {current} ---")
            game.end_turn()

    env.close()

if __name__ == "__main__":
    main()
