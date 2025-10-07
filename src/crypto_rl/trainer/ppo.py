"""PPO egitimi icin moduler trainer bilesenleri."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from crypto_rl.env.trading import CryptoTradingEnv
from crypto_rl.agents.ppo import PPOAgent


@dataclass
class RolloutCollector:
    """PPO roll-out verilerini toplar."""

    env: CryptoTradingEnv
    agent: PPOAgent
    rollout_steps: int

    def collect(self) -> Dict[str, List]:
        """Ortamda ilerleyerek ham deneyimleri toplar."""
        rollout = defaultdict(list)
        state = self.env.reset()
        action_mask = self.env.valid_action_mask()
        step = 0

        while step < self.rollout_steps:
            action, log_prob, _ = self.agent.select_action(state, action_mask=action_mask)
            next_state, reward, done, info = self.env.step(action)

            rollout["states"].append(state)
            rollout["actions"].append(action)
            rollout["log_probs"].append(log_prob.item())
            rollout["rewards"].append(reward)
            rollout["dones"].append(done)

            if done:
                state = self.env.reset()
                action_mask = self.env.valid_action_mask()
            else:
                state = next_state
                action_mask = info.get("action_mask", self.env.valid_action_mask())
            step += 1

        return rollout


@dataclass
class RolloutPostProcessor:
    """Toplanan rollout uzerinden avantaj ve getiri hesaplar."""

    agent: PPOAgent
    device: torch.device

    def enrich(self, rollout: Dict[str, List]) -> Dict[str, List]:
        """GAE ve toplam getirileri ekler."""
        stacked_states = np.stack(rollout["states"], axis=0).astype(np.float32)
        with torch.no_grad():
            values = (
                self.agent.critic(torch.from_numpy(stacked_states).to(self.device))
                .squeeze(-1)
                .cpu()
                .numpy()
            )
        next_values = np.append(values[1:], values[-1])
        advantages = self.agent.compute_gae(
            rewards=np.asarray(rollout["rewards"], dtype=np.float32),
            dones=np.asarray(rollout["dones"], dtype=np.float32),
            values=values,
            next_values=next_values,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        rollout["advantages"] = advantages.tolist()
        rollout["returns"] = returns.tolist()
        return rollout


@dataclass
class ValidationRunner:
    """Validation veri seti uzerinde modeli test eder."""

    agent: PPOAgent
    config: Any
    val_data: Any

    def run(self, epoch: int):
        """Validation ortamini calistirir ve metrikleri dondurur."""
        val_env = CryptoTradingEnv(self.val_data, self.config)
        obs = val_env.reset()
        done = False
        deterministic = self.config.get("test.use_deterministic_policy", True)

        while not done:
            action, _, _ = self.agent.select_action(obs, deterministic=deterministic)
            obs, _, done, _ = val_env.step(action)

        equity = np.asarray(val_env.equity_curve, dtype=np.float32)
        net_profit = (equity[-1] - equity[0]) / (equity[0] + 1e-8)
        returns = np.diff(equity) / (equity[:-1] + 1e-8)
        sharpe = returns.mean() / (returns.std() + 1e-8) if len(returns) > 1 else 0.0
        max_dd = val_env.max_drawdown(equity)

        log_path = Path(self.config.get("validation.log_path", "outputs/validation/validation_results.csv"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{net_profit:.4f},{sharpe:.4f},{max_dd:.4f},{equity[-1]:.2f}\n"
            )

        return {
            "net_profit": float(net_profit),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "final_budget": float(equity[-1]),
            "log_path": str(log_path),
        }


class PPOTrainer:
    """Egitim dongusunu yurutur."""

    def __init__(
        self,
        env: CryptoTradingEnv,
        agent: PPOAgent,
        config,
        *,
        val_data=None,
        logger=None,
    ):
        self.env = env
        self.agent = agent
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))

        self.collector = RolloutCollector(
            env=self.env,
            agent=self.agent,
            rollout_steps=config.get("ppo.rollout_steps"),
        )
        self.processor = RolloutPostProcessor(agent=self.agent, device=self.device)
        self.validator: Optional[ValidationRunner] = (
            ValidationRunner(agent=self.agent, config=config, val_data=val_data)
            if val_data is not None
            else None
        )
        self.validate_every = config.get("training.validate_every")
        self.logger = logger
        self._entropy_initial = float(self.agent.settings.entropy_coeff)
        entropy_min = config.get("ppo.entropy_min_coeff", self._entropy_initial)
        self._entropy_min = float(entropy_min) if entropy_min is not None else self._entropy_initial
        decay_cfg = config.get("ppo.entropy_decay_epochs")
        self._entropy_decay_epochs = self._resolve_decay_epochs(decay_cfg, config)
        if self._entropy_decay_epochs is not None and self._entropy_decay_epochs <= 0:
            self._entropy_decay_epochs = None
        if abs(self._entropy_min - self._entropy_initial) < 1e-8:
            self._entropy_decay_epochs = None
        metric_cfg = config.get("training.best_metric", "net_profit")
        self._best_metric_name, self._maximize_metric = self._parse_metric_choice(metric_cfg)
        self._best_metric_value: Optional[float] = None
        self._best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self._best_metric_epoch: Optional[int] = None

    def train(self, num_epochs: int):
        """Belirli sayida epoch boyunca PPO egitimi yapar."""
        for epoch in range(1, num_epochs + 1):
            self._update_entropy_coeff(epoch)
            # Ortamdan yeni rollout topluyoruz.
            rollout = self.collector.collect()
            # Avantaj ve getiri hesaplari ile rollout'u zenginlestiriyoruz.
            rollout = self.processor.enrich(rollout)
            self.agent.update(rollout)
            named_dist = self._log_action_distribution(epoch, rollout)

            if self.validate_every:
                if self.validator and epoch % self.validate_every == 0:
                    metrics = self.validator.run(epoch)
                    if self.logger:
                        self.logger.info(
                            "Dogrulama | epoch=%s net_profit=%.4f sharpe=%.4f max_dd=%.4f final=%.2f",
                            epoch,
                            metrics["net_profit"],
                            metrics["sharpe"],
                            metrics["max_drawdown"],
                            metrics["final_budget"],
                        )
                    self._maybe_update_best(metrics, epoch)

            self._log_equity(epoch, named_dist)

    # Yardimcilar ------------------------------------------------------- #
    def _update_entropy_coeff(self, epoch: int) -> None:
        """Entropi katsayisini belirtilen çizelgeye göre ayarlar."""
        if not self._entropy_decay_epochs or self._entropy_min is None:
            return
        progress = min(max((epoch - 1) / self._entropy_decay_epochs, 0.0), 1.0)
        new_coeff = self._entropy_initial + (self._entropy_min - self._entropy_initial) * progress
        self.agent.set_entropy_coeff(new_coeff)
        if self.logger and (
            epoch == 1
            or epoch == self._entropy_decay_epochs + 1
            or epoch % max(self._entropy_decay_epochs // 5, 1) == 0
        ):
            self.logger.info(
                "Entropi katsayisi guncellendi | epoch=%s coeff=%.4f", epoch, new_coeff
            )

    def _resolve_decay_epochs(self, decay_cfg, config) -> Optional[int]:
        """Entropi çürüme süresini sayıya dönüştürür."""
        if decay_cfg in (None, False):
            return None
        try:
            value = float(decay_cfg)
        except (TypeError, ValueError):
            return None
        if value > 1:
            return int(value)
        if value <= 0:
            return None
        total_epochs = config.get("training.total_epochs")
        if not total_epochs:
            return None
        return max(int(total_epochs * value), 1)

    def _maybe_update_best(self, metrics: Dict[str, Any], epoch: int) -> None:
        """Validation metriklerine gore en iyi agirliklari saklar."""
        if not metrics or self._best_metric_name not in metrics:
            return
        try:
            metric_value = float(metrics[self._best_metric_name])
        except (TypeError, ValueError):
            return
        improved = (
            self._best_metric_value is None
            or (metric_value > self._best_metric_value if self._maximize_metric else metric_value < self._best_metric_value)
        )
        if not improved:
            return
        self._best_metric_value = metric_value
        self._best_state_dict = self._snapshot_actor_state()
        self._best_metric_epoch = epoch
        if self.logger:
            direction = "↑" if self._maximize_metric else "↓"
            self.logger.info(
                "Yeni en iyi model | metric=%s %s %.4f | epoch=%s",
                self._best_metric_name,
                direction,
                metric_value,
                epoch,
            )

    def _snapshot_actor_state(self) -> Dict[str, torch.Tensor]:
        """Akt�r agirliklarini CPU'ya klonlayarak dondurur."""
        return {key: param.detach().cpu().clone() for key, param in self.agent.actor.state_dict().items()}

    def _parse_metric_choice(self, setting: Any) -> tuple[str, bool]:
        """Metrik ayarini (isim, maximize?) seklinde dondurur."""
        if not setting:
            return "net_profit", True
        metric = str(setting).strip()
        maximize = True
        if metric.startswith("-"):
            maximize = False
            metric = metric[1:]
        metric = metric or "net_profit"
        return metric, maximize

    def best_checkpoint(self) -> Dict[str, Any]:
        """Kaydedilecek en iyi model hakkinda bilgi dondurur."""
        state_dict = self._best_state_dict or self._snapshot_actor_state()
        return {
            "state_dict": state_dict,
            "metric_name": self._best_metric_name,
            "metric_value": self._best_metric_value,
            "epoch": self._best_metric_epoch,
        }

    def _count_invalid_sells(self, rollout: Dict[str, List]) -> Dict[str, float]:
        """Gecersiz SELL aksiyonlarini sayar."""
        invalid = 0
        for state, action in zip(rollout["states"], rollout["actions"]):
            position = state[-1]
            if action == 1 and position == 0:
                invalid += 1
        total = len(rollout["actions"])
        return {"count": invalid, "ratio": invalid / total if total else 0.0}

    def _log_action_distribution(self, epoch: int, rollout: Dict[str, List]) -> Dict[str, int]:
        """Aksiyon dagilimini hesaplar."""
        dist = Counter(rollout["actions"])
        labels = self._build_action_labels()
        return {labels[idx]: dist.get(idx, 0) for idx in range(len(labels))}

    def _log_equity(
        self,
        epoch: int,
        action_dist: Dict[str, int],
    ):
        """Aksiyon ve performans metriklerini tek satirda loglar."""
        if not self.logger:
            return

        equity = self.env.equity_curve
        if len(equity) < 2:
            self.logger.warning(
                "Epoch %s | actions=%s | Equity egrisi icin yeterli veri yok.",
                epoch,
                action_dist,
            )
            return

        eq = np.asarray(equity, dtype=np.float32)
        returns = np.diff(eq) / (eq[:-1] + 1e-8)
        net = (eq[-1] - eq[0]) / (eq[0] + 1e-8)
        sharpe = returns.mean() / (returns.std() + 1e-8)
        dd = self.env.max_drawdown(eq) if hasattr(self.env, "max_drawdown") else 0.0

        self.logger.info(
            "Epoch %s | actions=%s net_profit=%.4f sharpe=%.4f max_dd=%.4f final=%.2f",
            epoch,
            action_dist,
            net,
            sharpe,
            dd,
            eq[-1],
        )

    def _build_action_labels(self) -> Dict[int, str]:
        """Aksiyon indekslerini okunabilir etiketlere cevirir."""
        labels = {0: "HOLD", 1: "SELL"}
        for idx, coin in enumerate(self.env.coin_list):
            labels[idx + 2] = f"BUY {coin}"
        return labels
