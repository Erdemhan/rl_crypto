"""Tahmin günlüklerini dışa aktarmaya yarayan yardımcı."""

from __future__ import annotations

import csv
import os
from typing import List

from crypto_rl.env.trading import CryptoTradingEnv


def dump_predictions(agent, config, data, results_dir: str, deterministic: bool = True) -> str:
    """Ajanın tahminlerini `trades_log.csv` olarak kaydet."""
    env = CryptoTradingEnv(data, config)
    obs = env.reset()
    done = False

    os.makedirs(results_dir, exist_ok=True)

    while not done:
        action, _, _ = agent.select_action(obs, deterministic=deterministic)
        obs, _, done, _ = env.step(action)

    rows: List[dict] = env.prediction_log

    out_csv = os.path.join(results_dir, "trades_log.csv")
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return out_csv


__all__ = ["dump_predictions"]
