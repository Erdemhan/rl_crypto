# scripts/live.py

import time
import sys
from pathlib import Path

import numpy as np
import torch

from utils.config_loader import Config
from evaluation.logger import TradeLogger

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from crypto_rl.agents.ppo import PPOAgent  # noqa: E402
from crypto_rl.env.trading import CryptoTradingEnv  # noqa: E402

from data.data_loader import load_price_data  # noqa: E402
from data.split_utils import split_data  # noqa: E402

# Gerçek zamanlı veri simülasyonu – ileri geliştirmede Binance API ile değiştirilebilir
def get_live_state(env, config):
    state = env._get_observation()
    return state

def main():
    config = Config()
    device = torch.device(config.get("device", "cpu"))
    logger = TradeLogger(config)

    # Dummy veri ile başla (gerçek versiyonda canlı veri buraya gelecek)
    df = load_price_data(config)
    _, _, live_df = split_data(df, config)
    env = CryptoTradingEnv(live_df, config)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(obs_dim, action_dim, config, device)
    model_path = config.get("training.model_save_path")
    try:
        actor_state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        actor_state = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(actor_state)

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
