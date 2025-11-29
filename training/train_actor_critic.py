import os
import random
from collections import deque

import numpy as np

from gymnasium_env.utils.constants import MAX_STEPS, NUM_EPISODES, SUBDIV
from gymnasium_env.envs.game_env import GameEnv
from gymnasium_env.agents.actor_critic.ac_agent import ActorCriticAgent
from gymnasium_env.agents.random_agent import RandomAgent
from gymnasium_env.utils.action_utils import get_legal_actions

# For plotting without needing a display (saves to PNG)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------

def ensure_dir(path: str):
    if path is not None and path != "":
        os.makedirs(path, exist_ok=True)


def choose_opponent_type_option_a(ep: int,
                                  num_episodes: int,
                                  snapshot_available: bool) -> str:
    """
    Phase 1 (0 - 30% episodes):
        70% vs_random
        30% self_play
        0% vs_frozen
    Phase 2 (30 - 70% episodes):
        40% vs_random
        30% self_play
        30% vs_frozen   (if no snapshots yet, reallocated to random)
    Phase 3 (70 - 100% episodes):
        10% vs_random
        30% self_play
        60% vs_frozen   (if no snapshots yet, reallocated to self_play)
    """
    progress = ep / float(num_episodes)

    if progress < 0.3:
        # Phase 1
        probs = {
            "vs_random": 0.7,
            "self_play": 0.3,
            "vs_frozen": 0.0,
        }
    elif progress < 0.7:
        # Phase 2
        if snapshot_available:
            probs = {
                "vs_random": 0.4,
                "self_play": 0.3,
                "vs_frozen": 0.3,
            }
        else:
            # no snapshots yet, give that prob to random
            probs = {
                "vs_random": 0.7,
                "self_play": 0.3,
                "vs_frozen": 0.0,
            }
    else:
        # Phase 3
        if snapshot_available:
            probs = {
                "vs_random": 0.1,
                "self_play": 0.3,
                "vs_frozen": 0.6,
            }
        else:
            # no snapshots yet, give that prob to self-play
            probs = {
                "vs_random": 0.1,
                "self_play": 0.9,
                "vs_frozen": 0.0,
            }

    # Sample opponent_type from these probabilities
    r = random.random()
    cumulative = 0.0
    for ot, p in probs.items():
        cumulative += p
        if r <= cumulative:
            return ot
    # Fallback (should not happen due to float rounding, but just in case)
    return "self_play"


def select_frozen_opponent_action(env,
                                  ac_agent: ActorCriticAgent,
                                  frozen_pool,
                                  obs,
                                  legal_actions,
                                  current_player: int):
    """
    Select an action from a frozen snapshot opponent.
    frozen_pool: dict with keys:
        "paths": list[str]
        "agents": dict[path -> ActorCriticAgent]
    """
    if not frozen_pool["paths"]:
        # Safety fallback to random action
        return random.choice(legal_actions)

    # Sample a snapshot path
    snapshot_path = random.choice(frozen_pool["paths"])

    # Lazily load or reuse cached frozen agent
    if snapshot_path not in frozen_pool["agents"]:
        frozen_agent = ActorCriticAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            lr=0.0,                      # not training this one
            gamma=ac_agent.gamma,
            value_coef=ac_agent.value_coef,
            entropy_coef=0.0,
        )
        frozen_agent.load_model(snapshot_path)
        frozen_pool["agents"][snapshot_path] = frozen_agent
    else:
        frozen_agent = frozen_pool["agents"][snapshot_path]

    action, _ = frozen_agent.select_action(
        obs=obs,
        legal_actions=legal_actions,
        current_player=current_player,
        greedy=True,
    )
    if action is None:
        action = random.choice(legal_actions)
    return action


def train_stage(
    subdiv,
    num_episodes,
    max_steps_per_episode,
    lr,
    gamma,
    value_coef,
    entropy_coef,
    save_dir,
    save_prefix,
    load_model_path=None,
    moving_avg_window=50,
    snapshot_interval=None,
):
    """
    Train a single ActorCriticAgent with opponent diversity:
      - Self-play (shared AC controls both players, transitions for both)
      - Versus random opponent (AC as Player 0 only, opponent is RandomAgent)
      - Versus frozen snapshot opponents (AC as Player 0 only)
    Returns:
        history: dict with episode-wise metrics
        final_path: path to final saved model
    """
    print(
        f"\n=== Starting training stage: subdiv={subdiv}, "
        f"episodes={num_episodes}, lr={lr}, ent={entropy_coef} ==="
    )

    ensure_dir(save_dir)

    # Create environment for this stage
    env = GameEnv(players=2, pieces_per=1, subdiv=subdiv, render_mode=None)
    obs, _ = env.reset()

    # Create or load learning agent
    ac_agent = ActorCriticAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        lr=lr,
        gamma=gamma,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
    )

    if load_model_path is not None and os.path.exists(load_model_path):
        ac_agent.load_model(load_model_path)
        print(f"Loaded model from: {load_model_path}")

    # Random opponent as its own agent
    random_agent = RandomAgent(env.action_space, env.observation_space)

    # Snapshot pool: paths + cached frozen agents
    frozen_pool = {
        "paths": [],
        "agents": {},  # path -> ActorCriticAgent
    }

    # Default snapshot interval if not given
    if snapshot_interval is None:
        # e.g. 5 snapshots over the run
        snapshot_interval = max(1000, num_episodes // 5)

    # For logging & plotting
    moving_avg_returns = deque(maxlen=moving_avg_window)
    win_count = 0
    save_interval = max(10, num_episodes // 10)

    history = {
        "episodes": [],
        "returns": [],
        "avg_returns": [],
        "winrates": [],
        "losses": [],
        "value_losses": [],
        "entropies": [],
        "opponent_types": [],
    }

    # ---------------------------------------------------------
    # Entropy annealing setup
    # ---------------------------------------------------------
    initial_entropy = entropy_coef
    final_entropy = 1e-4
    decay_start = int(0.3 * num_episodes)
    decay_end   = int(0.9 * num_episodes)

    for ep in range(1, num_episodes + 1):
        # Update entropy coefficient for this episode
        if ep < decay_start:
            current_ent = initial_entropy
        elif ep > decay_end:
            current_ent = final_entropy
        else:
            # linear decay between initial_entropy and final_entropy
            t = (ep - decay_start) / float(decay_end - decay_start)
            current_ent = initial_entropy + t * (final_entropy - initial_entropy)
        ac_agent.entropy_coef = current_ent

        obs, _ = env.reset()
        done = False

        states = []
        action_indices = []
        rewards = []
        dones = []
        masks = []

        ep_return = 0.0

        # Pick opponent type for this episode using Option A curriculum
        snapshot_available = len(frozen_pool["paths"]) > 0
        opponent_type = choose_opponent_type_option_a(
            ep, num_episodes, snapshot_available
        )

        # For readability, we define a flag:
        # - self_play: AC controls both players (like original script)
        # - otherwise: AC controls only Player 0; Player 1 is opponent.
        is_self_play = (opponent_type == "self_play")

        for step in range(max_steps_per_episode):
            game = env.game
            current_player = game.current_player

            legal_actions = get_legal_actions(env)

            if not legal_actions:
                game.end_turn()
                continue

            # --------------------------------------------------
            # Select action depending on opponent_type
            # --------------------------------------------------
            if is_self_play:
                # AC controls both players; record transitions for both.
                action, info = ac_agent.select_action(
                    obs=obs,
                    legal_actions=legal_actions,
                    current_player=current_player,
                    greedy=False,
                )
                if action is None:
                    action = random.choice(legal_actions)
                    record_transition = False
                else:
                    record_transition = True

            else:
                # AC only controls Player 0; Player 1 is opponent.
                if current_player == 0:
                    # Our learning agent
                    action, info = ac_agent.select_action(
                        obs=obs,
                        legal_actions=legal_actions,
                        current_player=current_player,
                        greedy=False,
                    )
                    if action is None:
                        action = random.choice(legal_actions)
                        record_transition = False
                    else:
                        record_transition = True
                else:
                    # Opponent controls Player 1
                    if opponent_type == "vs_random":
                        # Use RandomAgent class here
                        action = random_agent.select_action(obs, legal_actions)
                    elif opponent_type == "vs_frozen":
                        action = select_frozen_opponent_action(
                            env=env,
                            ac_agent=ac_agent,
                            frozen_pool=frozen_pool,
                            obs=obs,
                            legal_actions=legal_actions,
                            current_player=current_player,
                        )
                    else:
                        # Safety fallback: random uniform
                        action = random.choice(legal_actions)
                    record_transition = False
                    info = None  # unused

            # --------------------------------------------------
            # Step environment
            # --------------------------------------------------
            next_obs, reward, terminated, truncated, info_dict = env.step(action)
            done = terminated or truncated

            ep_return += reward

            # Only record transitions when this action belongs to the learning agent
            if record_transition and info is not None:
                states.append(info["state"])
                action_indices.append(info["action_index"])
                rewards.append(reward)
                dones.append(done)
                masks.append(info["mask"])

            remaining_actions = get_legal_actions(env)
            if not remaining_actions:
                game.end_turn()

            obs = next_obs

            if done:
                break

        # Track win rate (player 0 is the perspective)
        if env.game.winner == 0:
            win_count += 1

        moving_avg_returns.append(ep_return)
        recent_avg = float(np.mean(moving_avg_returns))
        recent_winrate = win_count / ep

        # Update agent
        if states:
            states_arr = np.stack(states, axis=0).astype(np.float32)
            action_indices_arr = np.array(action_indices, dtype=np.int64)
            rewards_arr = np.array(rewards, dtype=np.float32)
            dones_arr = np.array(dones, dtype=bool)
            masks_arr = np.stack(masks, axis=0)

            stats = ac_agent.update(
                states=states_arr,
                action_indices=action_indices_arr,
                rewards=rewards_arr,
                dones=dones_arr,
                masks=masks_arr,
            )
        else:
            stats = None

        # --- Logging to console ---
        if stats is not None:
            loss = float(stats["loss"])
            vloss = float(stats["value_loss"])
            ent = float(stats["entropy"])
            print(
                f"[subdiv={subdiv}] Ep {ep:5d} | "
                f"Opp {opponent_type:9s} | "
                f"Ret {ep_return:7.2f} | "
                f"AvgRet {recent_avg:7.2f} | "
                f"WinRate {recent_winrate:.3f} | "
                f"Loss {loss:.4f} | "
                f"VLoss {vloss:.4f} | "
                f"Ent {ent:.4f} | EntCoef {ac_agent.entropy_coef:.5f} | "
                f"Winner: {env.game.winner}"
            )
        else:
            loss = vloss = ent = np.nan
            print(
                f"[subdiv={subdiv}] Ep {ep:5d} | Opp {opponent_type:9s} | "
                f"Ret {ep_return:7.2f} | "
                f"AvgRet {recent_avg:7.2f} | "
                f"WinRate {recent_winrate:.3f} | "
                f"(no update) | EntCoef {ac_agent.entropy_coef:.5f}"
            )

        # --- Save to history for plotting/tuning ---
        history["episodes"].append(ep)
        history["returns"].append(ep_return)
        history["avg_returns"].append(recent_avg)
        history["winrates"].append(recent_winrate)
        history["losses"].append(loss)
        history["value_losses"].append(vloss)
        history["entropies"].append(ent)
        history["opponent_types"].append(opponent_type)

        # Periodic checkpoint
        if ep % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"{save_prefix}_ep{ep}.pt")
            ac_agent.save_model(ckpt_path)
            print(f"   Saved checkpoint: {ckpt_path}")

        # Periodic snapshot for frozen-opponent pool
        if ep % snapshot_interval == 0:
            snapshot_path = os.path.join(save_dir, f"{save_prefix}_snapshot_ep{ep}.pt")
            ac_agent.save_model(snapshot_path)
            frozen_pool["paths"].append(snapshot_path)
            print(f"   Saved snapshot for opponent pool: {snapshot_path}")

    # Final save
    final_path = os.path.join(save_dir, f"{save_prefix}_final.pt")
    ac_agent.save_model(final_path)
    print(f"Stage finished. Saved final model: {final_path}")

    env.close()
    return history, final_path
