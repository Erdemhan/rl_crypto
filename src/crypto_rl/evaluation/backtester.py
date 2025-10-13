"""Backtest surecini moduler hale getiren siniflar."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os

import pandas as pd


@dataclass
class BacktestResult:
    """Backtest ciktilarini temsil eden basit veri sinifi."""

    equity_curve: List[float] = field(default_factory=list)
    trades: List[dict] = field(default_factory=list)
    trades_path: Optional[str] = None


class Backtester:
    """Ortam ve ajani kullanarak backtest calistirir."""

    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config

    def run(self, deterministic: bool = True) -> BacktestResult:
        """Backtest dongusunu yurutur."""
        state = self.env.reset()
        action_mask = self.env.valid_action_mask()
        done = False

        while not done:
            action, _, _ = self.agent.select_action(
                state, deterministic=deterministic, action_mask=action_mask
            )
            step = self.env.step(action)
            if step is None:
                break
            state, _, done, info = step
            action_mask = info.get("action_mask", self.env.valid_action_mask())

        result = BacktestResult(
            equity_curve=list(self.env.equity_curve),
            trades=list(getattr(self.env, "trade_log", [])),
        )
        self._persist(result)
        return result

    def _persist(self, result: BacktestResult):
        """CSV kayitlarini olusturur."""
        trades_path = self.config.get("test.backtest_log_path")

        if trades_path and result.trades:
            os.makedirs(os.path.dirname(trades_path), exist_ok=True)
            trades_df = pd.DataFrame(result.trades)
            if "step" not in trades_df.columns:
                trades_df.insert(0, "step", range(len(trades_df)))
            trades_df.to_csv(trades_path, index=False)
            result.trades_path = trades_path
