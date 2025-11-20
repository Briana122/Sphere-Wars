import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

## Encoding Functions ##

def get_state_action_size(env):
    '''
    Compute dimensions of state and action for DQN
    The Q-network needs this fixed input/output size
    '''

    # obs_space = env.observation_space
    num_tiles = env.num_tiles
    players = env.players
    max_pieces = env.max_pieces

    ownership_channels = players + 1
    state_size = num_tiles * ownership_channels + players
    action_size = max_pieces * num_tiles * 2
    return state_size, action_size

def ownership_to_onehot(ownership_arr, players):
    ''' Make a numerical input of one-hot encoding '''
    num_tiles = ownership_arr.shape[0]
    channels = players + 1

    onehot = np.zeros((num_tiles, channels), dtype=np.float32)
    for i, v in enumerate(ownership_arr):
        index = players if v is None or v == -1 else v
        onehot[i, index] = 1.0
    return onehot

def encode_observation(obs, players):
    ''' dqn needs a fixed-size numerical input, so encode it '''

    ownership = obs["ownership"]
    resources = obs["resources"].astype(np.float32)

    onehot_ownership = ownership_to_onehot(ownership, players)
    flat_ownership = onehot_ownership.reshape(-1)
    state = np.concatenate([flat_ownership, resources.astype(np.float32)], axis=0)

    return state


## Action Mapping Functions ##

def tuple_to_index(piece_id, dest_tile, action_type, max_pieces, num_tiles):
    return ((piece_id * num_tiles) + dest_tile) * 2 + action_type

def index_to_tuple(index, max_pieces, num_tiles):
    action_type = index % 2
    temp = index // 2
    dest_tile = temp % num_tiles
    piece_id = temp // num_tiles
    return piece_id, dest_tile, action_type


## DQN Model ##

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=(256, 256), activation=nn.ReLU):
        ''' Fully connected Q-network with hidden layers '''
        super().__init__()
        layers = []
        last = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last, hidden_size))
            layers.append(activation())
            last = hidden_size

        layers.append(nn.Linear(last, action_size))
        self.net = nn.Sequential(*layers)

        # Initialize weights (add training stability)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        ''' Calculate the Q-values for all actions '''
        return self.net(x)
    
    def act(self, obs, env, epsilon=0.0, legal_mask=None, device=None):
        ''' Uses an epsilon-greedy policy to select an action '''

        if device is None:
            device = next(self.parameters()).device
        players = env.players

        if isinstance(obs, dict): # encode if needed
            state = encode_observation(obs, players)
        else:
            state = obs

        # shape is (1, state_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        ## EXPLORATION ##
        if np.random.rand() < epsilon:
            action_size = env.max_pieces * env.num_tiles * 2

            # Sample random action from legal actions
            if legal_mask is not None:
                legal_indices = np.where(legal_mask)[0]
                if len(legal_indices) == 0:
                    index = 0
                else:
                    index = np.random.choice(legal_indices)
            return index, index_to_tuple(index, env.max_pieces, env.num_tiles)
        
        ## EXPLOITATION ##        
        with torch.no_grad():
            q = self.forward(state_tensor).squeeze(0).cpu().numpy() # Q-values for all actions

            if legal_mask is not None:
                # Mask out illegal actions (set their Q-values to -inf)
                illegal_indices = np.where(~legal_mask)[0]
                q[illegal_indices] = -np.inf

            if np.all(np.isneginf(q)):
                # All actions are illegal, return a default action
                q = self.forward(state_tensor).squeeze(0).cpu().numpy()

            index = np.nanargmax(q) # highest q
            return index, index_to_tuple(index, env.max_pieces, env.num_tiles)


## Save and load the helpers ##

def save_model(model: DQNModel, filepath: str):
    torch.save(model.state_dict(), filepath)

def load_model(model: DQNModel, filepath: str, map_location=None):
    model.load_state_dict(torch.load(filepath, map_location=map_location))
    return model
