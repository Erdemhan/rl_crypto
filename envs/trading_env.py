import gym
import numpy as np
from gym import spaces

class CryptoTradingEnv(gym.Env):
    def __init__(self, data, config):
        super().__init__()
        self.data = data.copy()
        self.config = config

        self.equity_curve = []


        self.coin_list = config.get("data.coin_list")
        self.num_coins = len(self.coin_list)
        self.num_indicators = 7
        self.initial_balance = config.get("environment.initial_balance")
        self.commission = config.get("environment.commission")
        self.use_one_hot = config.get("environment.use_one_hot_position")

        reward_cfg = self.config.get("reward.weights")
        self.w_p = reward_cfg.get("profit", 1.0)
        self.w_s = reward_cfg.get("sharpe", 0.5)
        self.w_d = reward_cfg.get("drawdown", 1.0)


        self.timestamps = sorted(self.data["timestamp"].unique())
        self.total_steps = len(self.timestamps)

        self.action_space = spaces.Discrete(12)  # 0=Hold, 1=Sell, 2-11=Buy[coin0-coin9]

        obs_len = self.num_coins * self.num_indicators
        obs_len += (self.num_coins + 1) if self.use_one_hot else 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: cash, 1–10: coin index + 1
        self.coin_holdings = np.zeros(self.num_coins)
        self.trade_log = []
        self.equity_curve = []

        return self._get_observation()


    def step(self, action):
        done = (self.current_step >= self.total_steps - 1
                or self._get_portfolio_value() < self.initial_balance * 0.03  # ↖️ portföy tükendiğinde bitir
            )


        prev_value = self._get_portfolio_value()
        self._execute_action(action)

        if not done:
            self.current_step += 1

        current_value = self._get_portfolio_value()
        reward = (current_value - prev_value) / (prev_value + 1e-8)
        self.equity_curve.append(current_value)

        # 🎯 Risk-duyarlı shaping: episode sonunda sadece
        if done and len(self.equity_curve) > 2:
            reward += self._compute_final_reward()

        return self._get_observation(), reward, done, {"portfolio_value": current_value}



    def _get_observation(self):
        timestamp = self.timestamps[self.current_step]
        step_data = self.data[self.data["timestamp"] == timestamp]

        indicators = []
        for coin in self.coin_list:
            row = step_data[step_data["symbol"] == coin].iloc[0]
            indicators.append(np.array(row["indicators"]))

        indicators_flat = np.concatenate(indicators)

        if self.use_one_hot:
            pos_vec = np.zeros(self.num_coins + 1)
            pos_vec[self.position] = 1
        else:
            pos_vec = np.array([self.position], dtype=np.float32)

        return np.concatenate([indicators_flat, pos_vec]).astype(np.float32)

    def _execute_action(self, action):
        timestamp = self.timestamps[self.current_step]
        step_data = self.data[self.data["timestamp"] == timestamp]
        prices = [step_data[step_data["symbol"] == coin].iloc[0]["close"] for coin in self.coin_list]

        if action == 0:
            return  # Hold

        elif action == 1:
            if self.position == 0:
                return  # no coin to sell
            coin_idx = self.position - 1
            sell_price = prices[coin_idx]
            self.balance += self.coin_holdings[coin_idx] * sell_price * (1 - self.commission)
            self.coin_holdings[coin_idx] = 0
            self.position = 0

        elif 2 <= action <= 11:
            coin_idx = action - 2
            if self.position != 0:
                self._execute_action(1)  # sell current holding first

            buy_price = prices[coin_idx]
            quantity = self.balance / buy_price * (1 - self.commission)
            self.coin_holdings[coin_idx] = quantity
            self.balance = 0
            self.position = coin_idx + 1

    def _get_portfolio_value(self):
        timestamp = self.timestamps[self.current_step]
        step_data = self.data[self.data["timestamp"] == timestamp]
        prices = [step_data[step_data["symbol"] == coin].iloc[0]["close"] for coin in self.coin_list]

        value = self.balance
        for i in range(self.num_coins):
            value += self.coin_holdings[i] * prices[i]

        return value
    
    def _compute_final_reward(self):
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        net_profit = (equity[-1] - equity[0]) / equity[0]
        sharpe = returns.mean() / (returns.std() + 1e-8)
        max_dd = self._max_drawdown(equity)

        shaped_reward = self.w_p * net_profit + self.w_s * sharpe - self.w_d * max_dd
        return shaped_reward

    
    def _max_drawdown(self, equity_curve):
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        return np.max(drawdown)

