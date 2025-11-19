from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .q_learning_FA.q_learning_fa_agent import QLearningFAAgent
# from .dqn.dqn_agent import DQNAgent

def make_agent(name, action_space, observation_space, **kwargs):
    """
    Creates an agent instance by string name.
    """

    name = name.lower()

    if name in ("random", "rand"):
        return RandomAgent(action_space, observation_space)

    if name in ("qfa"):
        return QLearningFAAgent(
            action_space=action_space,
            observation_space=observation_space,
            **kwargs
        )

    # Do this for each model
    
    # if name == "dqn":
    #     return DQNAgent(action_space, observation_space, **kwargs)

    raise ValueError("Unknown agent type: " + name)
