# gymnasium_env/agents/actor_critic/ac_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNet(nn.Module):
    """
    Shared-body actor-critic neural network.

    Inputs:
        - state vector of size state_dim

    Outputs:
        - policy_logits: [batch, num_actions]
        - value:         [batch]
    """

    def __init__(self, state_dim, num_actions, hidden_dim=256):
        super().__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions

        # Shared torso
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy logits
        self.policy_head = nn.Linear(hidden_dim, num_actions)

        # State-value head
        self.value_head = nn.Linear(hidden_dim, 1)

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.
        x can be a 1D tensor or batch of tensors.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.body(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value
