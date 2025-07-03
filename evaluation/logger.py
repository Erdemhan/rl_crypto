# evaluation/logger.py

import pandas as pd
import os
from datetime import datetime

class TradeLogger:
    def __init__(self, config):
        self.config = config
        self.logs = []
        self.portfolio_values = []

    def log_trade(self, time, action, coin, price, amount, cash_after, position_after):
        self.logs.append({
            "time": time,
            "action": action,
            "coin": coin,
            "price": price,
            "amount": amount,
            "cash_after": cash_after,
            "position_after": position_after
        })

    def log_equity(self, time, value):
        self.portfolio_values.append({
            "time": time,
            "portfolio_value": value
        })

    def save(self):
        trade_path = self.config.get("test.backtest_log_path")
        equity_path = self.config.get("test.equity_curve_path")
        os.makedirs(os.path.dirname(trade_path), exist_ok=True)

        pd.DataFrame(self.logs).to_csv(trade_path, index=False)
        pd.DataFrame(self.portfolio_values).to_csv(equity_path, index=False)
        print(f"✅ Trade log saved to {trade_path}")
        print(f"✅ Equity curve saved to {equity_path}")
