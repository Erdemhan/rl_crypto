"""PPO eÄŸitim boru hattÄ±nÄ±n temel uygulamasÄ±."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from crypto_rl.agents.ppo import PPOAgent
from crypto_rl.config.loader import load_config
from crypto_rl.env.trading import CryptoTradingEnv
from crypto_rl.pipelines.common import DictConfigAdapter, RunPaths, prepare_run_paths, set_seed, setup_logging
from crypto_rl.trainer.ppo import PPOTrainer
from data.data_loader import load_price_data
from data.split_utils import split_data


def _resolve_model_path(paths: RunPaths, profile: Optional[str]) -> Path:
    """Model aÄŸÄ±rlÄ±klarÄ±nÄ± profil bazlÄ± ve zaman damgalÄ± kaydeder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{profile}_{timestamp}.pth" if profile else f"model_{timestamp}.pth"
    return paths.models / filename


def _materialize_config(
    loader,
    profile: Optional[str],
    paths: RunPaths,
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfigAdapter:
    """Ã‡alÄ±ÅŸma zamanÄ± yollarÄ±nÄ± config Ã¼zerine iÅŸler."""
    merged = loader.resolved(profile)
    adapter = DictConfigAdapter(merged)

    adapter.set("training.model_save_path", str(paths.models / "best_model.pth"))
    adapter.set("training.log_path", str(paths.logs / "train_logs.csv"))
    adapter.set("test.backtest_log_path", str(paths.results / "trades_log.csv"))
    adapter.set("validation.log_path", str(paths.validation / "validation_results.csv"))

    if overrides:
        for key, value in overrides.items():
            adapter.set(key, value)

    return adapter


def run_training(
    config_path: Optional[str] = None,
    profile: Optional[str] = None,
    run_id: Optional[str] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
    log_to_console: bool = True,
    output_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Optional[str]]:
    """Config dosyasÄ±nÄ± okuyup PPO eÄŸitim dÃ¶ngÃ¼sÃ¼nÃ¼ Ã§alÄ±ÅŸtÄ±rÄ±r."""
    loader = load_config(config_path)
    meta = dict(loader.meta())
    if output_root is not None:
        meta["profiles_dirname"] = str(output_root)

    effective_profile = profile
    effective_run_id = run_id or meta.get("current_run_id")

    paths = prepare_run_paths(meta, effective_profile, effective_run_id)
    paths.ensure()

    config = _materialize_config(loader, effective_profile, paths, overrides)
    log_path = setup_logging(paths.logs, effective_profile, prefix="training", enable_console=log_to_console)

    seed = config.get("seed", 42)
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    logging.info("Seed=%s | Device=%s", seed, device)

    logging.info("Veri yÃ¼kleniyor...")
    df = load_price_data(config)

    logging.info("Veri train/val/test olarak ayrÄ±lÄ±yor...")
    train_df, val_df, _ = split_data(df, config)
    if train_df is None or len(train_df) == 0:
        raise ValueError("EÄŸitim verisi boÅŸ. Tarih aralÄ±klarÄ±nÄ± veya veri hazÄ±rlÄ±ÄŸÄ±nÄ± kontrol edin.")

    logging.info("Trading ortamÄ± oluÅŸturuluyor...")
    env = CryptoTradingEnv(train_df, config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    logging.info("PPO ajanÄ± hazÄ±rlanÄ±yor...")
    agent = PPOAgent(obs_dim, action_dim, config, device)

    logging.info("EÄŸitim dÃ¶ngÃ¼sÃ¼ baÅŸlatÄ±lÄ±yor...")
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
