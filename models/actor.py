# models/actor.py

import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)     # logits, no Tanh here!
        )

    def forward(self, x):
        return self.model(x)
