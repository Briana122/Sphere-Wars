## DQN AGENT ##
# implements:
# - epsilon-greedy action selection
# - masking invalid actions
# - training step
# - soft update of target network

import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch

from .dqn_model import DQNModel, encode_observation, get_state_action_size
from .replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(
        # setup the constants
        self,
        env,
        device="cpu",
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        buffer_capacity=100_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=5000,
        target_update_freq=1000,
    ):
        # Initialize the DQN Agent!
        self.env = env
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.state_dim, self.action_dim = get_state_action_size(env)

        ## NETWORKS ##
        self.policy_net = DQNModel(self.state_dim, self.action_dim).to(device)
        self.target_net = DQNModel(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, self.state_dim, device=device)

        self.steps_done = 0


    def epsilon(self):
        ''' exponential decay '''
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-1.0 * self.steps_done / self.epsilon_decay)

    def select_action(self, obs, legal_mask=None):
        epsilon = self.epsilon()
        self.steps_done += 1
        return self.policy_net.act(obs, self.env, epsilon=epsilon, legal_mask=legal_mask, device=self.device)

    def add_transition(self, state, action, reward, next_state, done):
        ''' To store the transition in replay buffer '''
        self.replay_buffer.add(state, action, reward, next_state, done)


    def train_step(self):
        '''
        Gradient descent on a batch of transitions
        i.e. minimize difference between Q(s,a;w) and target y = r + gamma * max_a' Q_target(s',a';w')
        '''

        if len(self.replay_buffer) < self.batch_size:
            return None  # not enough samples

        # Sample batch of transitions
        batch = self.replay_buffer.sample(self.batch_size)
        state = batch["state"]
        next_state = batch["next_state"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]

        # Q(s,a) from policy network
        q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # max_a' Q_target(s',a')
        with torch.no_grad():
            q_next = self.target_net(next_state)
            q_next_max, _ = q_next.max(dim=1)
            q_target = reward + (1 - done) * self.gamma * q_next_max

        loss = nn.MSELoss()(q_values, q_target)

        # Weight update from lecture pseudocode
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        # Try to avoid updating too frequently!!
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
