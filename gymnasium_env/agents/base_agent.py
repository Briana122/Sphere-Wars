# basic agent that the models can inherit from

class BaseAgent:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def select_action(self, state, legal_actions):
        raise NotImplementedError

    def train(self, transition):
        pass # temp

    def save_model(self, filepath):
        pass # temp

    def load_model(self, filepath):
        pass # temp