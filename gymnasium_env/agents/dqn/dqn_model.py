# will have the actual model setup here

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.model):
    # TODO: define the neural network architecture here


    def __init__(self, state_size, action_size):
        super.__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)
    
    def act(self, obs, env, epsilon=0.0, legal_mask=None, device=None):
        if device is None:
            device = next(self.parameters()).device

        # TODO: implement this
    

## Encoding Functions ##

def get_state_action_dims(env):
    # Compute state_dim and action_dim based on env's observation and action spaces
    obs_space = env.observation_space
    num_tiles = env.num_tiles
    players = env.players
    max_pieces = env.max_pieces

    ownership_channels = players + 1
    state_dim = num_tiles * ownership_channels + players
    action_dim = max_pieces * num_tiles * 2
    return state_dim, action_dim

def ownership_to_onehot(ownership_arr, players):
    num_tiles = ownership_arr.shape[0]
    channels = players + 1
    onehot = np.zeros((num_tiles, channels), dtype=np.float32)
    for i, v in enumerate(ownership_arr):
        if v is None or v == -1:
            index = players
        else:
            index = int(v)
        onehot[i, index] = 1.0
    return onehot

def encode_observation(obs, players):
    ownership = obs["ownership"]
    resources = obs["resources"].astype(np.float32)

    onehot_ownership = ownership_to_onehot(ownership, players)
    flat_ownership = onehot_ownership.reshape(-1)
    state = np.concatenate([flat_ownership, resources.astype(np.float32)], axis=0)

    return state


## Action Mapping Functions ##

def tuple_to_index(piece_id, dest_tile, action_type, max_pieces, num_tiles):
    return ((int(piece_id) * num_tiles) + int(dest_tile)) * 2 + int(action_type)

def index_to_tuple(index, max_pieces, num_tiles):
    action_type = index % 2
    temp = index // 2
    dest_tile = temp % num_tiles
    piece_id = temp // num_tiles
    return piece_id, dest_tile, action_type


## Save and load the helpers ##

def save_model(model: DQNModel, filepath: str):
    torch.save(model.state_dict(), filepath)

def load_model(model: DQNModel, filepath: str, map_location=None):
    model.load_state_dict(torch.load(filepath, map_location=map_location))
    return model