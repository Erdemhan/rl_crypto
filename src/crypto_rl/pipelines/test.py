"""Kayıtlı modeller için backtest boru hattı."""

from __future__ import annotations

import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from crypto_rl.agents.ppo import PPOAgent
from crypto_rl.config.loader import load_config
from crypto_rl.evaluation.backtester import Backtester
from crypto_rl.env.trading import CryptoTradingEnv
from crypto_rl.pipelines.common import DictConfigAdapter, prepare_run_paths, set_seed, setup_logging
from data.data_loader import load_price_data
from data.split_utils import split_data


def _latest_model(models_dir: Path, profile: Optional[str]) -> Path:
    """Profil için en yeni ağırlık dosyasını getirir."""
    if not models_dir.exists():
        raise FileNotFoundError(f"Model klasörü bulunamadı: {models_dir}")
    pattern = f"{profile}_*.pth" if profile else "*.pth"
    candidates = sorted(models_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"Model dosyası bulunamadı: {models_dir}")
    return candidates[-1]


def _materialize_config(loader, profile: Optional[str], paths) -> DictConfigAdapter:
    """Backtest sırasında kullanılacak yolları config üzerine işler."""
    adapter = DictConfigAdapter(loader.resolved(profile))
    adapter.set("test.equity_curve_path", str(paths.results / "equity_curve.csv"))
    adapter.set("test.backtest_log_path", str(paths.results / "trades_log.csv"))
    return adapter


def _run_profile_job(job_args: Tuple[str, Optional[str], str, Optional[bool]]) -> Tuple[str, Optional[Dict[str, str]]]:
    """Tek profil için backtest akışını izolasyon içinde yürütür."""
    config_path, profile, run_id, deterministic_flag = job_args
    loader = load_config(config_path)
    meta = loader.meta()
    paths = prepare_run_paths(meta, profile, run_id)

    if not paths.models.exists():
        logging.warning("[%s] Model klasörü bulunamadı, profil atlanıyor: %s", profile, paths.models)
        return profile or "default", None

    paths.results.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(paths.logs, profile, prefix="backtest")
    logger = logging.getLogger(profile or "backtest")

    config = _materialize_config(loader, profile, paths)
    seed = config.get("seed", 42)
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    logger.info("Seed=%s | Device=%s | LogFile=%s", seed, device, log_path)

    df = load_price_data(config)
    _, _, test_df = split_data(df, config)
    if test_df is None or len(test_df) == 0:
        logger.warning("[%s] Test verisi boş, profil atlandı.", profile)
        return profile or "default", None

    env = CryptoTradingEnv(test_df, config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(obs_dim, action_dim, config, device)
    model_path = _latest_model(paths.models, profile)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # weights_only 2.2+; eski PyTorch s�r�mlerinde parametre desteklenmiyor
        state_dict = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()
    logger.info("Model yüklendi: %s", model_path)

    deterministic = deterministic_flag if deterministic_flag is not None else bool(
        config.get("test.use_deterministic_policy", False)
    )
    logger.info("Deterministik politika: %s", deterministic)

    backtester = Backtester(env, agent, config)
    result = backtester.run(deterministic=deterministic)

    logger.info("Backtest tamamlandı. Equity adım sayısı=%s", len(result.equity_curve))
    return profile or "default", {
        "model_path": str(model_path),
        "equity_curve_path": result.equity_curve_path or str(paths.results / "equity_curve.csv"),
        "actions_path": str(Path(config.get("test.backtest_log_path")).with_name("actions.csv"))
        if config.get("test.backtest_log_path")
        else str(paths.results / "actions.csv"),
        "log_path": str(log_path),
    }


def run_backtests(
    config_path: Optional[str],
    run_id: str,
    *,
    deterministic: Optional[bool] = None,
    profiles: Optional[List[str]] = None,
    processes: int = 1,
) -> Dict[str, Dict[str, str]]:
    """Belirtilen run kimliği altındaki tüm profilleri backtest eder."""
    if not run_id:
        raise ValueError("Backtest için run-id zorunludur.")

    loader = load_config(config_path)
    target_profiles = profiles or list(loader.profiles())
    if not target_profiles:
        raise ValueError("Konfigürasyonda çalıştırılabilir profil bulunamadı.")

    jobs = [(config_path or "config.yaml", profile, run_id, deterministic) for profile in target_profiles]

    if processes is None or processes <= 0:
        max_workers = cpu_count() or 1
        processes = min(len(jobs), max_workers)

    if processes <= 1 or len(jobs) == 1:
        results = [_run_profile_job(job) for job in jobs]
    else:
        worker_count = min(processes, len(jobs))
        with Pool(processes=worker_count) as pool:
            results = pool.map(_run_profile_job, jobs)

    summary: Dict[str, Dict[str, str]] = {}
    for profile, data in results:
        if data:
            summary[profile] = data

    return summary


__all__ = ["run_backtests"]
