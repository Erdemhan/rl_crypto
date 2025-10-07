"""PPO eğitim boru hattının temel uygulaması."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

from crypto_rl.agents.ppo import PPOAgent
from crypto_rl.config.loader import load_config
from crypto_rl.env.trading import CryptoTradingEnv
from crypto_rl.pipelines.common import DictConfigAdapter, RunPaths, prepare_run_paths, set_seed, setup_logging
from crypto_rl.trainer.ppo import PPOTrainer
from data.data_loader import load_price_data
from data.split_utils import split_data


def _resolve_model_path(paths: RunPaths, profile: Optional[str]) -> Path:
    """Model ağırlıklarını profil bazlı ve zaman damgalı kaydeder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{profile}_{timestamp}.pth" if profile else f"model_{timestamp}.pth"
    return paths.models / filename


def _materialize_config(loader, profile: Optional[str], paths: RunPaths) -> DictConfigAdapter:
    """Çalışma zamanı yollarını config üzerine işler."""
    merged = loader.resolved(profile)
    adapter = DictConfigAdapter(merged)

    adapter.set("training.model_save_path", str(paths.models / "best_model.pth"))
    adapter.set("training.log_path", str(paths.logs / "train_logs.csv"))
    adapter.set("test.equity_curve_path", str(paths.results / "equity_curve.csv"))
    adapter.set("test.backtest_log_path", str(paths.results / "trades_log.csv"))
    adapter.set("validation.log_path", str(paths.validation / "validation_results.csv"))

    return adapter


def run_training(
    config_path: Optional[str] = None,
    profile: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Config dosyasını okuyup PPO eğitim döngüsünü çalıştırır."""
    loader = load_config(config_path)
    meta = loader.meta()

    effective_profile = profile
    effective_run_id = run_id or meta.get("current_run_id")

    paths = prepare_run_paths(meta, effective_profile, effective_run_id)
    paths.ensure()

    config = _materialize_config(loader, effective_profile, paths)
    log_path = setup_logging(paths.logs, effective_profile, prefix="training")

    seed = config.get("seed", 42)
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    logging.info("Seed=%s | Device=%s", seed, device)

    logging.info("Veri yükleniyor...")
    df = load_price_data(config)

    logging.info("Veri train/val/test olarak ayrılıyor...")
    train_df, val_df, _ = split_data(df, config)
    if train_df is None or len(train_df) == 0:
        raise ValueError("Eğitim verisi boş. Tarih aralıklarını veya veri hazırlığını kontrol edin.")

    logging.info("Trading ortamı oluşturuluyor...")
    env = CryptoTradingEnv(train_df, config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    logging.info("PPO ajanı hazırlanıyor...")
    agent = PPOAgent(obs_dim, action_dim, config, device)

    logging.info("Eğitim döngüsü başlatılıyor...")
    trainer = PPOTrainer(env, agent, config, val_data=val_df, logger=logging.getLogger(__name__))
    trainer.train(config.get("training.total_epochs"))

    best_info = trainer.best_checkpoint()
    best_metric_value = best_info.get("metric_value")
    best_metric_name = best_info.get("metric_name")
    best_metric_epoch = best_info.get("epoch")

    model_artifact: Optional[str] = None
    if config.get("training.save_best_model", True):
        paths.models.mkdir(parents=True, exist_ok=True)
        best_path = paths.models / "best_model.pth"
        torch.save(best_info["state_dict"], best_path)
        timestamp_path = _resolve_model_path(paths, effective_profile)
        if timestamp_path != best_path:
            torch.save(best_info["state_dict"], timestamp_path)
        model_artifact = str(best_path)
        logging.info(
            "En iyi model kaydedildi | path=%s metric=%s value=%s epoch=%s",
            best_path,
            best_metric_name,
            f"{best_metric_value:.4f}" if best_metric_value is not None else "N/A",
            best_metric_epoch if best_metric_epoch is not None else "N/A",
        )

    return {
        "model_path": model_artifact,
        "log_path": str(log_path),
        "run_directory": str(paths.base),
        "best_metric": {
            "name": best_metric_name,
            "value": best_metric_value,
            "epoch": best_metric_epoch,
        },
    }


__all__ = ["run_training"]
