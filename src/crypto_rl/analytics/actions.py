"""Aksiyon gunlugu olusturma fonksiyonlari."""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import pandas as pd

from crypto_rl.env.trading import CryptoTradingEnv


def dump_actions_with_flags(agent, config, data, results_dir: str, deterministic: bool = True):
    """Backtest aksiyonlarini bayrak bilgileri ile kaydeder."""
    env = CryptoTradingEnv(data, config)
    obs = env.reset()
    done = False

    os.makedirs(results_dir, exist_ok=True)

    labels: Dict[int, str] = {0: "HOLD", 1: "SELL"}
    for idx, coin in enumerate(env.coin_list):
        labels[idx + 2] = f"BUY {coin}"

    rows: List[Dict[str, float]] = []
    step = 0
    previous_position = env.position_mgr.position

    while not done:
        action, _, _ = agent.select_action(obs, deterministic=deterministic)
        invalid_sell = action == 1 and previous_position == 0
        redundant_buy = action >= 2 and previous_position == (action - 2 + 1)
        obs, _, done, _ = env.step(action)
        current_position = env.position_mgr.position

        rows.append(
            {
                "step": step,
                "action": labels.get(action, str(action)),
                "position": int(current_position),
                "cash_only": int(current_position == 0),
                "invalid_sell": int(invalid_sell),
                "redundant_buy": int(redundant_buy),
            }
        )
        previous_position = current_position
        step += 1

    equity_path = os.path.join(results_dir, "equity_curve.csv")
    if os.path.exists(equity_path):
        equity_df = pd.read_csv(equity_path)
        limit = min(len(rows), len(equity_df))
        for idx in range(limit):
            rows[idx]["portfolio_value"] = float(equity_df.loc[idx, "portfolio_value"])

    out_csv = os.path.join(results_dir, "actions.csv")
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
