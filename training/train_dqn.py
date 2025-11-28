import time
import numpy as np
import torch

from gymnasium_env.agents.dqn.dqn_agent import DQNAgent
from gymnasium_env.envs.game_env import GameEnv
from gymnasium_env.agents.dqn.dqn_model import encode_observation
from gymnasium_env.agents.dqn.utils import make_legal_mask

## SETUP ##
device = "cuda" if torch.cuda.is_available() else "cpu"
num_episodes = 5
max_steps = 50
render_every = 50 # don't render every episode for visibility purposes
render_delay = 0.3

# Initialize weights and empty buffer
env = GameEnv(players=2, pieces_per=1, render_mode="human")
agent = DQNAgent(env, device=device)


## LOOP FOR TRAINING ##
for ep in range(num_episodes):
    obs, _ = env.reset()
    state = encode_observation(obs, env.players)
    total_reward = 0

    render = (ep % render_every == 0)

    for step in range(max_steps):
        env.game.selected = None  # reset selected piece on each step
        legal_mask = make_legal_mask(env)

        # # Debug info
        # print("legal_mask length:", len(legal_mask))
        # print("Legal true count:", legal_mask.sum())
        # if legal_mask.sum() == 0:
        #     print("ERROR: No legal actions. Current obs:", obs)
        #     print("Current player:", env.game.current_player)
        #     print("Pieces:", env.game.pieces)
        #     raise RuntimeError("No legal actions for this state")

        # Epsilon-greedy action selection and application
        action_index, action_tuple = agent.select_action(obs, legal_mask)
        # print(f"Chosen action index: {action_index}, legal: {legal_mask[action_index]}")
        next_obs, reward, done, truncated, info = env.step(action_tuple)
        env.game.end_turn()

        # print(f"Reward from environment: {reward}")
        piece_id, dest, action_type = action_tuple

        # Highlight selected piece in render
        piece_key = info.get("piece_key", None)
        if piece_key is not None and piece_key in env.game.pieces:
            env.game.selected = piece_key # valid piece
        else:
            env.game.selected = None

        # # Debug info
        # print("piece_id:", piece_id)
        # print("piece_keys:", list(env.game.pieces.keys()))

        before_owner = env.game.tiles[dest].owner
        after_owner = env.game.tiles[dest].owner

        # small negative reward for idle moves
        if action_type == 0 and before_owner == after_owner:
            reward -= 0.01
        next_state = encode_observation(next_obs, env.players)

        agent.add_transition(state, action_index, reward, next_state, done)

        # Sample mini-batch and compute targets
        loss = agent.train_step()

        state = next_state
        obs = next_obs
        total_reward += reward

        print(f"state: {state}, action: {action_index}, reward: {reward}, next_state: {next_state}")

        ## RENDER SELECT EPISODES ##
        if render:
            env.render()
            time.sleep(render_delay)

        if done:
            break


    ## LOGGING ##
    if (ep + 1) % 10 == 0:
        print(f"Episode {ep+1}, total_reward={total_reward:.2f}, "
              f"buffer_size={len(agent.replay_buffer)}, epsilon={agent.epsilon():.3f}")

    ## SAVE CHECKPOINT ##
    if (ep + 1) % 500 == 0:
        agent.save(f"dqn_checkpoint_ep{ep+1}.pt")