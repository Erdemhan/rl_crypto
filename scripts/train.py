# scripts/train.py

import os
import torch
from utils.config_loader import Config
from data.data_loader import load_price_data
from data.split_utils import split_data
from envs.trading_env import CryptoTradingEnv
from agents.ppo_agent import PPOAgent
from trainer.train_loop import PPOTrainer

import logging
import os
from datetime import datetime


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging started. Log file: {log_path}")


def main():
    setup_logging()
    config = Config()
    device = torch.device(config.get("device", "cpu"))

    print("📊 Loading data...")
    df = load_price_data(config)


    print("🔀 Splitting data...")
    train_df, val_df, _ = split_data(df, config)
    if train_df.empty:
        raise ValueError("train_df boş. Tarih aralıklarını veya filtreleme mantığını kontrol et.")

    print(f"🧪 train_df satır sayısı: {len(train_df)}")
    print("📋 train_df sütunları:", train_df.columns.tolist())


    print("📈 Preparing environment...")
    train_env = CryptoTradingEnv(train_df, config)

    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    print("🤖 Initializing PPO agent...")
    agent = PPOAgent(obs_dim, action_dim, config, device)

    print("🏋️ Starting training loop...")
    trainer = PPOTrainer(train_env, agent, config)
    trainer.train(config.get("training.total_epochs"))

    if config.get("training.save_best_model"):
        model_path = config.get("training.model_save_path")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(agent.actor.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    main()
