"""PPO ajanini ve ayarlarini moduler halde tanimlar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from models.actor import Actor
from models.critic import Critic


@dataclass
class PPOSettings:
    """Config icerisinden cekilen PPO hiperparametrelerini tutar."""

    gamma: float
    lam: float
    clip_epsilon: float
    entropy_coeff: float
    entropy_min_coeff: Optional[float]
    entropy_decay_epochs: Optional[int]
    value_coeff: float
    learning_rate: float
    epochs: int
    minibatch_size: int
    rollout_steps: int
    max_grad_norm: float

    @classmethod
    def from_config(cls, config) -> "PPOSettings":
        """Config nesnesinden degerleri okur."""
        return cls(
            gamma=config.get("ppo.gamma"),
            lam=config.get("ppo.lam"),
            clip_epsilon=config.get("ppo.clip_epsilon"),
            entropy_coeff=config.get("ppo.entropy_coeff"),
            entropy_min_coeff=config.get("ppo.entropy_min_coeff"),
            entropy_decay_epochs=config.get("ppo.entropy_decay_epochs"),
            value_coeff=config.get("ppo.value_coeff"),
            learning_rate=config.get("ppo.learning_rate"),
            epochs=config.get("ppo.epochs"),
            minibatch_size=config.get("ppo.minibatch_size"),
            rollout_steps=config.get("ppo.rollout_steps"),
            max_grad_norm=config.get("ppo.max_grad_norm"),
        )


class PPOAgent:
    """Politika ve deger aglarini yoneten PPO ajan sinifi."""

    def __init__(self, obs_dim: int, action_dim: int, config, device: torch.device):
        self.device = device
        self.settings = PPOSettings.from_config(config)

        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.settings.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.settings.learning_rate)
        self.entropy_coeff = float(self.settings.entropy_coeff)

    # Aksiyon secimi ---------------------------------------------------- #
    def select_action(
        self,
        state: np.ndarray | torch.Tensor,
        deterministic: bool = False,
        action_mask: Optional[np.ndarray | torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Politikadan aksiyon ornekle; maskelenen aksiyonlari secmez."""
        state_tensor = self._ensure_tensor(state)
        logits = self.actor(state_tensor)
        logits = self._apply_action_mask(logits, action_mask)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs=probs)
        action = probs.argmax(dim=-1) if deterministic else dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Politika log-olasiligi ve deger tahminini uretir."""
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).squeeze(-1)
        return log_probs, entropy, values

    # GAE ve guncelleme ------------------------------------------------- #
    def compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
    ) -> np.ndarray:
        """Genellestirilmis avantaj kestirimi hesaplar."""
        adv = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.settings.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.settings.gamma * self.settings.lam * (1 - dones[t]) * gae
            adv.insert(0, gae)
        return np.asarray(adv, dtype=np.float32)

    def update(self, batch: Dict[str, Any]):
        """Toplanan rollout ile politika ve deger agini optimize eder."""
        states = torch.as_tensor(np.stack(batch["states"]).astype(np.float32), device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device, dtype=torch.long)
        old_log_probs = torch.as_tensor(batch["log_probs"], device=self.device, dtype=torch.float32)
        returns = torch.as_tensor(batch["returns"], device=self.device, dtype=torch.float32)
        advantages = torch.as_tensor(batch["advantages"], device=self.device, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.settings.epochs):
            log_probs, entropy, values = self.evaluate(states, actions)
            ratios = (log_probs - old_log_probs).exp()

            surr1 = ratios * advantages
            surr2 = torch.clamp(
                ratios,
                1 - self.settings.clip_epsilon,
                1 + self.settings.clip_epsilon,
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values).pow(2).mean()
            entropy_bonus = -entropy.mean()

            total_loss = policy_loss + self.settings.value_coeff * value_loss + self.entropy_coeff * entropy_bonus

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.settings.max_grad_norm)
            self.actor_optim.step()
            self.critic_optim.step()

    def set_entropy_coeff(self, value: float) -> None:
        """Güncel entropi katsayısını dışarıdan ayarlamaya izin verir."""
        self.entropy_coeff = float(value)

    # Yardimcilar ------------------------------------------------------- #
    def _ensure_tensor(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Girdi durumunu tensore cevirir."""
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float()
        else:
            state_tensor = state.float()
        return state_tensor.to(self.device).unsqueeze(0)

    def _apply_action_mask(
        self,
        logits: torch.Tensor,
        action_mask: Optional[np.ndarray | torch.Tensor],
    ) -> torch.Tensor:
        if action_mask is None:
            return logits
        if isinstance(action_mask, np.ndarray):
            mask = torch.from_numpy(action_mask.astype(np.bool_))
        else:
            mask = action_mask.bool()
        mask = mask.to(logits.device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        masked_logits = logits.clone()
        masked_logits[~mask] = -1e9
        return masked_logits
