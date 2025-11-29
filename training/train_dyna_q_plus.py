from typing import Optional

import os
import random
from collections import deque

import numpy as np

from gymnasium_env.utils.constants import MAX_STEPS, NUM_EPISODES, SUBDIV
from gymnasium_env.envs.game_env import GameEnv
from gymnasium_env.agents.dyna_q_plus.dyna_gym_agent import DynaQPlusGymAgent
from gymnasium_env.agents.random_agent import RandomAgent
from gymnasium_env.utils.action_utils import get_legal_actions

# -------------------- Utility Functions --------------------
def ensure_dir(path: str):
    if path is not None and path != "":
        os.makedirs(path, exist_ok=True)

def choose_opponent_type(ep: int,
                         num_episodes: int,
                         snapshot_available: bool) -> str:
    '''
    The DynaQ+ Training has 3 phases for opponent selection:

    Phase 1 (0 - 40% episodes):
        100% vs_random
    Phase 2 (40 - 80% episodes):
        50% vs_random, 50% vs_frozen
    Phase 3 (80 - 100% episodes):
        20% vs_random, 80% vs_frozen
    '''
    progress = ep / float(num_episodes)

    if progress < 0.4:
        probs = {"vs_random": 1.0, "vs_frozen": 0.0}
    elif progress < 0.8:
        if snapshot_available:
            probs = {"vs_random": 0.5, "vs_frozen": 0.5}
        else:
            probs = {"vs_random": 1.0, "vs_frozen": 0.0}
    else:
        if snapshot_available:
            probs = {"vs_random": 0.2, "vs_frozen": 0.8}
        else:
            probs = {"vs_random": 1.0, "vs_frozen": 0.0}

    r = random.random()
    cumulative = 0.0
    for ot, p in probs.items():
        cumulative += p
        if r <= cumulative:
            return ot
    return "vs_random"


def make_frozen_opponent(snapshot_path: str,
                         action_space,
                         observation_space):
    '''Create a frozen DynaQPlusGymAgent from a snapshot file.'''
    frozen = DynaQPlusGymAgent(
        action_space=action_space,
        observation_space=observation_space,
    )
    frozen.load_model(snapshot_path)
    frozen.epsilon = 0.0
    return frozen

### PROGRESS: small helper to draw a text progress bar
def format_progress_bar(progress: float, width: int = 30) -> str:
    """
    progress: 0.0 -> 1.0
    """
    filled = int(round(width * progress))
    bar = "█" * filled + "-" * (width - filled)
    percent = int(progress * 100)
    return f"[{bar}] {percent:3d}%"

# -------------------- Training Function --------------------
def train_dyna_q_plus(
    *,
    subdiv: int = SUBDIV,
    num_episodes: int = 2000,
    max_steps_per_episode: int = MAX_STEPS,
    save_dir: str = "models_dyna",
    save_prefix: str = "dyna_q_plus",
    snapshot_interval: Optional[int] = 500,
    moving_avg_window: int = 100,
    epsilon: float = 0.9,
    epsilon_min: float = 0.1,
):
    """
    Train one DynaQPlusGymAgent.
    """

    ensure_dir(save_dir)

    env = GameEnv(players=2, pieces_per=1, render_mode=None, subdiv=subdiv)
    obs, _ = env.reset()

    epsilon_decay = (epsilon_min / epsilon) ** (1.0 / num_episodes)

    dyna_agent = DynaQPlusGymAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    random_opponent = RandomAgent(env.action_space, env.observation_space)

    frozen_paths: list[str] = []
    frozen_agents: dict[str, DynaQPlusGymAgent] = {}

    if snapshot_interval is None:
        snapshot_interval = max(500, num_episodes // 4)

    moving_avg_returns = deque(maxlen=moving_avg_window)
    recent_wins = deque(maxlen=100)  # track last 100 results (1 = win, 0 = not win)


    # --- Main Training Loop ---
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        dyna_agent.start_episode()
        done = False
        ep_return = 0.0

        opponent_type = choose_opponent_type(
            ep,
            num_episodes,
            snapshot_available=len(frozen_paths) > 0,
        )

        step_in_ep = 0

        if opponent_type == "vs_frozen" and frozen_paths:
            snap_path = random.choice(frozen_paths)
            if snap_path not in frozen_agents:
                frozen_agents[snap_path] = make_frozen_opponent(
                    snap_path, env.action_space, env.observation_space
                )
            frozen_opponent = frozen_agents[snap_path]
        else:
            frozen_opponent = None

        while not done and step_in_ep < max_steps_per_episode:
            step_in_ep += 1
            game = env.game
            current_player = game.current_player

            legal_actions = get_legal_actions(env)
            if not legal_actions:
                game.end_turn()
                continue

            if current_player == 0:
                action = dyna_agent.select_action(obs, legal_actions)
                if action is None:
                    action = random.choice(legal_actions)
                actor_is_dyna = True
            else:
                if opponent_type == "vs_random" or frozen_opponent is None:
                    action = random_opponent.select_action(obs, legal_actions)
                else:
                    action = frozen_opponent.select_action(obs, legal_actions)
                if action is None:
                    action = random.choice(legal_actions)
                actor_is_dyna = False

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if actor_is_dyna:
                ep_return += reward

                next_legal = get_legal_actions(env)
                dyna_agent.train_step(
                    prev_obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                    next_legal_actions=next_legal,
                )

            obs = next_obs

        dyna_agent.end_episode()
        
        # 1 if DynaQ+ (player 0) won, else 0
        if env.game.winner == 0:
            recent_wins.append(1)
        else:
            recent_wins.append(0)

        moving_avg_returns.append(ep_return)
        avg_return = float(np.mean(moving_avg_returns))

        # Recent winrate over last up-to-100 episodes
        recent_winrate = float(np.mean(recent_wins)) if recent_wins else 0.0
        progress = ep / float(num_episodes)

        # Get Epsilon
        current_epsilon = dyna_agent.get_epsilon()

        ### PROGRESS: per-episode progress line
        bar = format_progress_bar(progress)
        print(
            f"{bar}  "
            f"Ep {ep:5d}/{num_episodes}  "
            f"Opp={opponent_type:9s}  "
            f"Ret={ep_return:7.3f}  "
            f"AvgRet={avg_return:7.3f}  "
            f"WinRate100={recent_winrate:5.3f}  "
            f"Curr ε={current_epsilon:5.3f}",
            flush=True,
        )

        # --- Save snapshots & checkpoints ---
        if ep % snapshot_interval == 0:
            snap_path = os.path.join(
                save_dir, f"{save_prefix}_snapshot_ep{ep}.npz"
            )
            dyna_agent.save_model(snap_path)
            frozen_paths.append(snap_path)
            print(f"Saved snapshot: {snap_path}", flush=True)

        if ep % max(10, num_episodes // 2) == 0:
            ckpt_path = os.path.join(save_dir, f"{save_prefix}_latest.npz")
            dyna_agent.save_model(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}", flush=True)

    final_path = os.path.join(save_dir, f"{save_prefix}_final.npz")
    dyna_agent.save_model(final_path)
    print(f"Training complete. Final model saved to: {final_path}", flush=True)

    env.close()
    return final_path


if __name__ == "__main__":
    # Usage: .venv\Scripts\python.exe -m training.train_dyna_q_plus
    train_dyna_q_plus()
