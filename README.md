# 🤖 Crypto PPO Trading Bot

This project implements a crypto trading bot using Proximal Policy Optimization (PPO) in a custom Gym environment. It operates on the top 10 cryptocurrencies by trading volume and uses 7 technical indicators to form the observation space. The agent learns to make discrete decisions (Buy\[i], Sell, Hold) to maximize risk-adjusted returns using cumulative profit, Sharpe ratio, and maximum drawdown as reward components.

## 🫠 Features

* PPO-based reinforcement learning agent (Actor-Critic)
* Custom Gym-compatible environment
* Discrete action space: Hold, Sell, Buy\[1]–Buy\[10]
* 10 coins × 7 technical indicators: RSI, MACD, ADX, ATR, Bollinger Width, OBV, Stochastic
* Reward combines profit, Sharpe Ratio, and Max Drawdown
* Live / paper trading support
* Fully modular and OOP-structured Python code

## 📦 Installation

```bash
git clone https://github.com/erdmhn/crypto.git
cd crypto
pip install -r requirements.txt
```

## 🛠️ Project Structure

* `agents/` – PPO agent class
* `data/` – data loading, indicator calculation, and dataset splitting
* `envs/` – custom Gym environment for trading logic
* `models/` – Actor and Critic neural network models
* `trainer/` – PPO training loop
* `evaluation/` – backtesting, metric calculation, logging
* `scripts/` – `train.py`, `test.py`, and `live.py`
* `config/` – training and environment parameters in `config.yaml`
* `outputs/` – saved models, logs, equity curves

## 🧪 Training

```bash
python scripts/train.py
```

* Trains a PPO agent on historical hourly OHLCV data.
* Uses technical indicators per coin.
* Saves best model to: `outputs/models/best_model.pth`

## 🗰️ Backtesting

```bash
python scripts/test.py
```

* Runs evaluation on a separate test set.
* Outputs:

  * `outputs/results/trades_log.csv`
  * `outputs/results/equity_curve.csv`

## 📊 Performance Evaluation

```python
from evaluation.metrics import print_metrics
print_metrics("outputs/results/equity_curve.csv")
```

Outputs:

* Cumulative Return
* Sharpe Ratio
* Max Drawdown

## 📉 Live / Paper Trading

```bash
python scripts/live.py
```

Simulates live execution (can be adapted for Binance API):

* Feeds current market state to trained agent
* Logs trades and portfolio value in real time

## ⚙️ Configuration

All behavior is controlled via `config/config.yaml`, including:

* coin list and data intervals
* PPO hyperparameters
* environment and reward settings
* logging & model saving paths

## 📜 License

MIT License

## 👥 Author

Developed by [@erdmhn](https://github.com/erdmhn)
