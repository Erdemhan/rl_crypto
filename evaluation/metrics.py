# evaluation/metrics.py

import pandas as pd
import numpy as np


def load_trades_log(path):
    df = pd.read_csv(path)
    if "step" in df.columns:
        df = df.sort_values("step")
    df["returns"] = df["portfolio_value"].pct_change().fillna(0)
    return df.reset_index(drop=True)


def compute_cumulative_return(df):
    return_pct = df["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0] - 1
    return return_pct


def compute_sharpe_ratio(df, risk_free_rate=0.0, scale=252):
    returns = df["returns"]
    excess_returns = returns - risk_free_rate / scale
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    sharpe = mean_return / std_return if std_return != 0 else 0
    return sharpe * np.sqrt(scale)


def compute_max_drawdown(df):
    cumulative = df["portfolio_value"].cummax()
    drawdowns = df["portfolio_value"] / cumulative - 1
    return drawdowns.min()


def print_metrics(trades_path):
    df = load_trades_log(trades_path)
    cum_return = compute_cumulative_return(df)
    sharpe = compute_sharpe_ratio(df)
    mdd = compute_max_drawdown(df)

    print("\n== Performance Metrics ==")
    print(f" - Cumulative Return: {cum_return:.2%}")
    print(f" - Sharpe Ratio:      {sharpe:.2f}")
    print(f" - Max Drawdown:      {mdd:.2%}")
