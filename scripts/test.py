# scripts/test.py

import torch
from utils.config_loader import Config
from data.data_loader import load_price_data
from data.split_utils import split_data
from envs.trading_env import CryptoTradingEnv
from agents.ppo_agent import PPOAgent
from evaluation.backtester import Backtester

def main():
    config = Config()
    device = torch.device(config.get("device", "cpu"))

    print("📊 Loading test data...")
    df = load_price_data(config)
    _, _, test_df = split_data(df, config)

    print("📈 Preparing test environment...")
    env = CryptoTradingEnv(test_df, config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("🤖 Loading trained PPO agent...")
    agent = PPOAgent(obs_dim, action_dim, config, device)
    agent.actor.load_state_dict(torch.load(config.get("training.model_save_path"), map_location=device))

    print("🧪 Running backtest...")
    backtester = Backtester(env, agent, config)
    backtester.run()
    backtester.save_results()

if __name__ == "__main__":
    main()
