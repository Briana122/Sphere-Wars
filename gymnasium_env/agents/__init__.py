from gymnasium_env.agents.actor_critic.ac_agent import ActorCriticAgent
from gymnasium_env.agents.mcts_nn.mcts_nn import MCTSNNAgent  
from gymnasium_env.agents.dqn.dqn_agent import DQNAgent
from .base_agent import BaseAgent
from .random_agent import RandomAgent
from gymnasium_env.agents.dyna_q_plus import DynaQPlusGymAgent
# from .dqn.dqn_agent import DQNAgent

def make_agent(name, action_space, observation_space, **kwargs):
    """
    Creates an agent instance by string name.
    """

    name = name.lower()

    if name in ("random", "rand"):
        return RandomAgent(action_space, observation_space)

    if name in ("ac"):
        return ActorCriticAgent(
            action_space=action_space,
            observation_space=observation_space,
            **kwargs
        )
    
    if name in ("dyna", "dyna_q", "dyna_q_plus"):
        return DynaQPlusGymAgent(
            action_space=action_space,
            observation_space=observation_space,
            **kwargs
        )

    if name == "mcts_nn":
        from .mcts_nn.mcts_nn import MCTSNNAgent
        return MCTSNNAgent(action_space, observation_space)


    # Do this for each model
    
    if name == "dqn":
        return DQNAgent(action_space, observation_space, **kwargs)

    raise ValueError("Unknown agent type: " + name)
