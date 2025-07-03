# evaluation/backtester.py

import pandas as pd
import torch

class Backtester:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        self.device = config.get("device", "cpu")

        self.trades = []
        self.equity_curve = []

    def run(self):
        state = self.env.reset()
        done = False

        while not done:
            action, _, _ = self.agent.select_action(state, deterministic=True)
            
            # Güvenli ilerleme
            step_result = self.env.step(action)
            if step_result is None:
                break

            next_state, reward, done, info = step_result
            self.equity_curve.append(info["portfolio_value"])
            state = next_state

        return self.equity_curve


    def save_results(self):
        trade_path = self.config.get("test.backtest_log_path")
        curve_path = self.config.get("test.equity_curve_path")

        # Trade log yazılmak istenirse: self.trades → DataFrame
        # trades_df.to_csv(trade_path, index=False)

        curve_df = pd.DataFrame({
            "time": range(len(self.equity_curve)),
            "portfolio_value": self.equity_curve
        })
        curve_df.to_csv(curve_path, index=False)
        print(f"💾 Equity curve saved to {curve_path}")
