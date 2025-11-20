# store the transitions (s, a, r, s', done)

import torch
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity, state_dim, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.state_dim = state_dim

        # Pre-allocate memory
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

        self.pos = 0 # next insertion
        self.size = 0 # num stored experiences


    def add(self, state, action, reward, next_state, done):
        ''' When buffer is full, overwrite old experiences '''
        index = self.pos
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ''' Sample mini-batch of transitions '''

        # Randomly pick the indices from prev experiences
        indices = random.sample(range(self.size), batch_size)

        batch = dict(
            state=torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device),
            next_state=torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            action=torch.as_tensor(self.actions[indices], dtype=torch.long, device=self.device),

            reward=torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            done=torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device)

        )
        return batch
    
    def __len__(self):
        return self.size

    def is_full(self):
        return self.size == self.capacity