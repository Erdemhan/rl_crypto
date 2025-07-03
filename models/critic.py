# models/critic.py

import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Skaler değer
        )

    def forward(self, x):
        return self.model(x)
