# trainer/train_loop.py

import numpy as np
import torch
from collections import defaultdict
from tqdm import trange
from tqdm import tqdm
import logging
import os
from datetime import datetime

class PPOTrainer:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        self.device = config.get("device", "cpu")

        self.rollout_steps = config.get("ppo.rollout_steps")
        self.validate_every = config.get("training.validate_every")

    def collect_rollout(self):
        rollout = defaultdict(list)
        state = self.env.reset()
        step_count = 0

        while step_count < self.rollout_steps:
            action, log_prob, entropy = self.agent.select_action(state)
            next_state, reward, done, info = self.env.step(action)

            rollout["states"].append(state)
            rollout["actions"].append(action)
            rollout["log_probs"].append(log_prob.item())
            rollout["rewards"].append(reward)
            rollout["dones"].append(done)

            state = next_state
            step_count += 1

            if done:
                state = self.env.reset()

        return rollout

    def compute_returns_and_advantages(self, rollout):
        states = rollout["states"]
        rewards = rollout["rewards"]
        dones = rollout["dones"]

        with torch.no_grad():
            values = self.agent.critic(torch.FloatTensor(states).to(self.device)).squeeze(-1).cpu().numpy()
            next_values = np.append(values[1:], values[-1])

        advantages = self.agent.compute_gae(rewards, dones, values, next_values)
        returns = np.array(advantages) + values

        rollout["advantages"] = advantages
        rollout["returns"] = returns
        return rollout


    

    

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"\n📦 Epoch {epoch + 1}/{num_epochs}")
            with tqdm(total=1, desc=f"Epoch {epoch + 1}", bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}") as pbar:
                rollout = self.collect_rollout()
                rollout = self.compute_returns_and_advantages(rollout)
                self.agent.update(rollout)
                pbar.update()

            # 📋 Loglama: Skorlar
            rewards = np.array(rollout["rewards"])
            mean_reward = rewards.mean()
            total_reward = rewards.sum()

            from collections import Counter
            action_counts = Counter(rollout["actions"])
            logging.info(f"Epoch {epoch+1} action distribution: {dict(action_counts)}")


            # Eğer ortam log tutuyorsa (CryptoTradingEnv)
            if hasattr(self.env, "equity_curve") and len(self.env.equity_curve) > 2:
                equity = np.array(self.env.equity_curve)
                net_profit = (equity[-1] - equity[0]) / equity[0]
                returns = np.diff(equity) / equity[:-1]
                sharpe = returns.mean() / (returns.std() + 1e-8)
                drawdown = self.env._max_drawdown(equity)

                logging.info(
                    f"Epoch {epoch+1}: mean_reward={mean_reward:.4f}, total_reward={total_reward:.2f}, "
                    f"net_profit={net_profit:.4f}, sharpe={sharpe:.4f}, max_dd={drawdown:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch+1}: mean_reward={mean_reward:.4f}, total_reward={total_reward:.2f}"
                )


