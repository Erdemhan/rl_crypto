# evaluation/logger.py

import os
from typing import List, Dict

import pandas as pd


class TradeLogger:
    def __init__(self, config):
        self.config = config
        self.logs: List[Dict] = []
        self.portfolio_values: List[Dict] = []

    def log_trade(self, time, action, coin, price, amount, cash_after, position_after):
        self.logs.append({
            "time": time,
            "action": action,
            "coin": coin,
            "price": price,
            "amount": amount,
            "cash_after": cash_after,
            "position_after": position_after,
        })

    def log_equity(self, time, value):
        self.portfolio_values.append({
            "time": time,
            "portfolio_value": value,
        })

    def save(self):
        trade_path = self.config.get("test.backtest_log_path")
        if not trade_path:
            raise ValueError("Config must define test.backtest_log_path")

        os.makedirs(os.path.dirname(trade_path), exist_ok=True)

        records: List[Dict] = []
        if self.logs:
            records = [dict(entry) for entry in self.logs]

        if self.portfolio_values:
            if not records:
                records = [dict(entry) for entry in self.portfolio_values]
            else:
                for idx, equity_entry in enumerate(self.portfolio_values):
                    if idx < len(records):
                        records[idx]["portfolio_value"] = equity_entry.get("portfolio_value")
                    else:
                        records.append(dict(equity_entry))

        df = pd.DataFrame(records) if records else pd.DataFrame()
        df.to_csv(trade_path, index=False)
        print(f"? Trades saved to {trade_path}")
