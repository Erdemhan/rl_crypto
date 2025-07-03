# scripts/live.py

import time
import torch
import numpy as np
from utils.config_loader import Config
from agents.ppo_agent import PPOAgent
from evaluation.logger import TradeLogger

# Gerçek zamanlı veri simülasyonu – ileri geliştirmede Binance API ile değiştirilebilir
def get_live_state(env, config):
    state = env._get_observation()
    return state

def main():
    config = Config()
    device = torch.device(config.get("device", "cpu"))
    logger = TradeLogger(config)

    from envs.trading_env import CryptoTradingEnv
    from data.data_loader import load_price_data
    from data.split_utils import split_data

    # Dummy veri ile başla (gerçek versiyonda canlı veri buraya gelecek)
    df = load_price_data(config)
    _, _, live_df = split_data(df, config)
    env = CryptoTradingEnv(live_df, config)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(obs_dim, action_dim, config, device)
    agent.actor.load_state_dict(torch.load(config.get("training.model_save_path"), map_location=device))

    state = env.reset()
    print("🚀 Starting paper trading loop...")
    for _ in range(100):  # veya sonsuz döngü + time.sleep
        state = get_live_state(env, config)
        action, _, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)

        # loglama
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.log_equity(now, info["portfolio_value"])
        # logger.log_trade(...)  # trade detayları ortamdan alınarak doldurulabilir

        if done:
            break
        time.sleep(1)  # canlıda bu süre 1 saatlik periyotla eşleşir

    logger.save()

if __name__ == "__main__":
    main()
