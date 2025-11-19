import random
import numpy as np

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Simple agent that picks random legal actions.
    """

    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)

    def select_action(self, state, legal_actions):
        if not legal_actions:
            return None
        return random.choice(legal_actions)
