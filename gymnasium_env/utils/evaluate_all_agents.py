import os
import random
import numpy as np
import torch
from gymnasium_env.agents import make_agent

from gymnasium_env.agents.dyna_q_plus.dyna_gym_agent import DynaQPlusGymAgent
from gymnasium_env.envs.game_env import GameEnv
from gymnasium_env.utils.action_utils import get_legal_actions
from gymnasium_env.utils.constants import SUBDIV, MAX_STEPS

from gymnasium_env.agents.mcts_nn.mcts_nn import MCTSNNAgent
from gymnasium_env.agents.dqn.dqn_agent import DQNAgent
from gymnasium_env.agents.dqn.utils import make_legal_mask
from gymnasium_env.agents.actor_critic.ac_agent import ActorCriticAgent
from gymnasium_env.agents.random_agent import RandomAgent


def act_dqplus(agent, env, obs):
    """
    Dyna-Q+ agent (DynaQPlusGymAgent wrapper)
    uses:
        agent.select_action(obs, legal_actions)
    which returns a tuple or None.
    """

    legal_actions = get_legal_actions(env)
    if not legal_actions:
        return None

    return agent.select_action(obs, legal_actions)

def act_mcts(agent, env, obs):
    agent.set_env(env)

    legal_actions = get_legal_actions(env)
    if not legal_actions:
        return None

    visit_counts, chosen = agent._run_mcts(
        obs,
        env.game,
        legal_actions
    )

    # Convert visit counts to vector
    vec = np.zeros(agent.action_size, dtype=np.float32)
    total = sum(visit_counts.values())

    if total == 0:
        return random.choice(legal_actions)

    for a, n in visit_counts.items():
        vec[agent.action_to_index(a)] = n

    idx = int(np.argmax(vec))
    return agent.index_to_action(idx)



def act_dqn(agent, env, obs):
    """Matches DQN evaluator behaviour exactly."""
    legal_mask = make_legal_mask(env)
    if legal_mask.sum() == 0:
        return None

    _, action_tuple = agent.select_action(
        obs=obs,
        legal_mask=legal_mask
    )
    return action_tuple


def act_ac(agent, env, obs):
    """MATCHES the GOOD AC evaluator (softmax sampling, NOT greedy)."""

    legal_actions = get_legal_actions(env)
    if not legal_actions:
        return None

    current_player = env.game.current_player

    action, _ = agent.select_action(
        obs=obs,
        legal_actions=legal_actions,
        current_player=current_player,
        greedy=False,
    )
    return action


def act_random(agent, env, obs):
    legal_actions = get_legal_actions(env)
    if not legal_actions:
        return None
    return random.choice(legal_actions)


def build_dqplus_agent(model_path, subdiv):
    env = GameEnv(players=2, pieces_per=1, subdiv=subdiv)

    a = DynaQPlusGymAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.0, 
        epsilon_min=0.0,
        epsilon_decay=1.0,
        plan_n=20,
        bonus_c=0.01,
        bonus_mode="sqrt",
    )

    a.load_model(model_path)
    env.close()
    return ("DynaQPlus", a, act_dqplus)


def build_mcts_agent(model_path, subdiv):
    env = GameEnv(players=2, pieces_per=1, subdiv=subdiv)

    # New correct constructor signature
    a = make_agent("mcts_nn", env.action_space, env.observation_space)
    a.set_env(env)
    a.num_simulations = 128
    a.load_model(model_path)

    a.num_simulations = 128
    a.set_env(env)
    a.net.eval()      
    a.load_model(model_path)

    env.close()
    return ("MCTS", a, act_mcts)



def build_dqn_agent(model_path, subdiv):
    env = GameEnv(players=2, pieces_per=1, subdiv=subdiv)
    a = DQNAgent(env, device="cuda")
    a.load(model_path)
    env.close()
    return ("DQN", a, act_dqn)


def build_ac_agent(model_path, subdiv):
    """Uses SAME constructor as the good evaluator (NO lr=0.0 bug)."""
    env = GameEnv(players=2, pieces_per=1, subdiv=subdiv)
    a = ActorCriticAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
    )
    a.load_model(model_path)
    env.close()
    return ("ActorCritic", a, act_ac)


def build_random_agent(subdiv):
    env = GameEnv(players=2, pieces_per=1, subdiv=subdiv)
    a = RandomAgent(env.action_space, env.observation_space)
    env.close()
    return ("Random", a, act_random)


# ============================================================
# Game runner (unchanged)
# ============================================================

def play_game(agentA, actA, agentB, actB, subdiv):
    env = GameEnv(players=2, pieces_per=1, subdiv=subdiv, render_mode=None)
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        p = env.game.current_player
        legal_actions = get_legal_actions(env)
        if not legal_actions:
            env.game.end_turn()
            continue

        # Player 0 vs Player 1
        action = actA(agentA, env, obs) if p == 0 else actB(agentB, env, obs)

        if action is None:
            env.game.end_turn()
            continue

        obs, reward, terminated, truncated, _ = env.step(action)

        if not get_legal_actions(env):
            env.game.end_turn()

        done = terminated or truncated
        steps += 1

    winner = env.game.winner
    env.close()
    return winner


def evaluate_all(model_paths, subdiv=SUBDIV, games_per_pair=30):
    agents = []

    if "mcts" in model_paths:
        agents.append(build_mcts_agent(model_paths["mcts"], subdiv))
    if "dqn" in model_paths:
        agents.append(build_dqn_agent(model_paths["dqn"], subdiv))
    if "ac" in model_paths:
        agents.append(build_ac_agent(model_paths["ac"], subdiv))
    if "dqplus" in model_paths:
        agents.append(build_dqplus_agent(model_paths["dqplus"], subdiv))



    agents.append(build_random_agent(subdiv))

    names = [a[0] for a in agents]
    N = len(agents)
    win_matrix = np.zeros((N, N))

    print("\n=======================================")
    print("Evaluating Agents:", names)
    print("Games per matchup:", games_per_pair)
    print("=======================================\n")

    for i, (nameA, agentA, actA) in enumerate(agents):
        for j, (nameB, agentB, actB) in enumerate(agents):
            if i == j:
                continue

            print(f"--- {nameA} (P0) vs {nameB} (P1) ---")

            winsA = 0
            for g in range(games_per_pair):
                winner = play_game(agentA, actA, agentB, actB, subdiv)
                print(f"Game {g+1}/{games_per_pair}: winner={winner}")
                if winner == 0:
                    winsA += 1

            winrate = winsA / games_per_pair
            win_matrix[i, j] = winrate

            print(f"{nameA} winrate vs {nameB}: {winrate:.3f}\n")

    print("\n=========== FINAL WINRATE MATRIX ===========")
    print("Rows = Player 0 agent")
    print("Cols = Player 1 agent")
    print("Agents:", names)
    print(win_matrix)
    print("============================================\n")

    return names, win_matrix


if __name__ == "__main__":
    model_paths = {
        "mcts":   "mcts_checkpoints/mcts_nn_modelok.pt",
        "dqn":    "gymnasium_env/agents/dqn/dqn_final_model.pt",
        "ac":     "gymnasium_env/agents/actor_critic/ac_final_model.pt",
        "dqplus": "gymnasium_env/agents/dyna_q_plus/dqplus_model.npz",
    }

    evaluate_all(model_paths, subdiv=2, games_per_pair=20)
