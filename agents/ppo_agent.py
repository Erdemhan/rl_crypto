# agents/ppo_agent.py

import torch
import numpy as np
from torch.distributions import Categorical
from models.actor import Actor
from models.critic import Critic

class PPOAgent:
    def __init__(self, obs_dim, action_dim, config, device="cpu"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.config = config

        self.gamma = config.get("ppo.gamma")
        self.lam = config.get("ppo.lam")
        self.clip_epsilon = config.get("ppo.clip_epsilon")
        self.entropy_coeff = config.get("ppo.entropy_coeff")
        self.value_coeff = config.get("ppo.value_coeff")
        self.lr = config.get("ppo.learning_rate")

        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def select_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = state.to(self.device).unsqueeze(0)

        logits = self.actor(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = probs.argmax() if deterministic else dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate_action(self, state, action):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        values = self.critic(state).squeeze(-1)
        return log_probs, entropy, values

    def compute_gae(self, rewards, dones, values, next_values):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, batch):
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(batch["log_probs"]).to(self.device)
        returns = torch.FloatTensor(batch["returns"]).to(self.device)
        advantages = torch.FloatTensor(batch["advantages"]).to(self.device)

        for _ in range(self.config.get("ppo.epochs")):
            log_probs, entropy, values = self.evaluate_action(states, actions)

            ratios = (log_probs - old_log_probs).exp()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (returns - values).pow(2).mean()
            entropy_loss = -entropy.mean()

            total_loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.get("ppo.max_grad_norm"))
            self.actor_optim.step()
            self.critic_optim.step()
