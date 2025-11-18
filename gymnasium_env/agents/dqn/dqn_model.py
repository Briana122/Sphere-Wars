# will have the actual model setup here

import torch.nn as nn

class DQNModel(nn.model):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
    def forward(self, x):
        return self.net(x)