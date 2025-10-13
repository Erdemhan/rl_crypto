"""Kayıtlı modeller için backtest boru hattı."""

from __future__ import annotations

import copy
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from crypto_rl.agents.ppo import PPOAgent
from crypto_rl.config.loader import load_config
from crypto_rl.evaluation.backtester import Backtester
from crypto_rl.env.trading import CryptoTradingEnv
from crypto_rl.pipelines.common import DictConfigAdapter, prepare_run_paths, set_seed, setup_logging
from data.data_loader import load_price_data
from data.split_utils import select_range


def _materialize_config(
    loader,
    profile: Optional[str],
    paths,
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfigAdapter:
    """Backtest sırasında kullanılacak yolları config üzerine işler."""
    adapter = DictConfigAdapter(loader.resolved(profile))
    adapter.set("test.backtest_log_path", str(paths.results / "trades_log.csv"))
    if overrides:
        for key, value in overrides.items():
            adapter.set(key, value)
    return adapter


def _collect_test_sets(
    df,
    config: DictConfigAdapter,
    profile_label: str,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    """Config içindeki test aralıklarına göre veri alt kümelerini hazırla."""
    test_sets: Dict[str, Dict[str, Any]] = {}
    ranges_cfg = config.get("data.test_ranges", {})
    if isinstance(ranges_cfg, dict) and ranges_cfg:
        iterable = ranges_cfg.items()
    else:
        iterable = [("default", config.get("data.test_range"))]

    for name, date_range in iterable:
        if not date_range:
            logger.warning("[%s/%s] Test aralığı tanımsız, senaryo atlandı.", profile_label, name)
            continue
        try:
            subset = select_range(df, date_range)
        except ValueError as exc:
            logger.warning("[%s/%s] Test aralığı geçersiz: %s", profile_label, name, exc)
            continue
        if subset.empty:
            logger.warning("[%s/%s] Test verisi boş, senaryo atlandı.", profile_label, name)
            continue
        test_sets[name] = {"range": list(date_range), "data": subset}
    return test_sets


def _run_profile_job(
    job_args: Tuple[str, Optional[str], str, Optional[bool], Optional[Dict[str, Any]], bool, Optional[Union[str, Path]]]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Tek profil için backtest akışını izolasyon içinde yürütür."""
    config_path, profile, run_id, deterministic_flag, overrides, log_to_console, output_root = job_args
    loader = load_config(config_path)
    meta = dict(loader.meta())
    if output_root is not None:
        meta["profiles_dirname"] = str(output_root)
    paths = prepare_run_paths(meta, profile, run_id)

    if not paths.models.exists():
        logging.warning("[%s] Model klasörü bulunamadı, profil atlanıyor: %s", profile, paths.models)
        return profile or "default", None

    paths.results.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(paths.logs, profile, prefix="backtest", enable_console=log_to_console)
    logger = logging.getLogger(profile or "backtest")

    config = _materialize_config(loader, profile, paths, overrides)
    seed = config.get("seed", 42)
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    logger.info("Seed=%s | Device=%s | LogFile=%s", seed, device, log_path)

    df = load_price_data(config)
    test_sets = _collect_test_sets(df, config, profile or "default", logger)
    if not test_sets:
        logger.warning("[%s] Test aralığı bulunamadı, profil atlandı.", profile)
        return profile or "default", None

    # Gözlem/aksiyon boyutlarını belirlemek için ilk senaryoyu kullan.
    first_scenario = next(iter(test_sets.values()))
    probe_env = CryptoTradingEnv(first_scenario["data"], config)
    obs_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.n

    agent = PPOAgent(obs_dim, action_dim, config, device)
    model_path = _latest_model(paths.models, profile)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # weights_only 2.2+; eski PyTorch sürümlerinde parametre desteklenmiyor
        state_dict = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()
    logger.info("Model yüklendi: %s", model_path)

    deterministic = deterministic_flag if deterministic_flag is not None else bool(
        config.get("test.use_deterministic_policy", False)
    )
    logger.info("Deterministik politika: %s", deterministic)

    scenario_results: Dict[str, Dict[str, Any]] = {}
    for scenario_name, payload in test_sets.items():
        scenario_dir = paths.results / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        scenario_config = DictConfigAdapter(copy.deepcopy(config.data()))
        scenario_config.set("data.test_range", payload["range"])
        scenario_config.set("test.backtest_log_path", str(scenario_dir / "trades_log.csv"))

        env = CryptoTradingEnv(payload["data"], scenario_config)
        backtester = Backtester(env, agent, scenario_config)
        result = backtester.run(deterministic=deterministic)

        logger.info(
            "[%s/%s] Backtest tamamlandı. Kayıt adım sayısı=%s",
            profile or "default",
            scenario_name,
            len(result.equity_curve),
        )
        scenario_results[scenario_name] = {
            "trades_log_path": result.trades_path or str(scenario_dir / "trades_log.csv"),
            "range": payload["range"],
        }

    if not scenario_results:
        return profile or "default", None

    return profile or "default", {
        "model_path": str(model_path),
        "log_path": str(log_path),
        "tests": scenario_results,
    }


def _latest_model(models_dir: Path, profile: Optional[str]) -> Path:
    """Profil için en yeni ağırlık dosyasını getirir."""
    if not models_dir.exists():
        raise FileNotFoundError(f"Model klasörü bulunamadı: {models_dir}")
    pattern = f"{profile}_*.pth" if profile else "*.pth"
    candidates = sorted(models_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"Model dosyası bulunamadı: {models_dir}")
    return candidates[-1]


def run_backtests(
    config_path: Optional[str],
    run_id: str,
    *,
    deterministic: Optional[bool] = None,
    profiles: Optional[List[str]] = None,
    processes: int = 1,
    overrides: Optional[Dict[str, Any]] = None,
    log_to_console: bool = True,
    output_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Belirtilen run kimliği altındaki tüm profilleri backtest eder."""
    if not run_id:
        raise ValueError("Backtest için run-id zorunludur.")

    loader = load_config(config_path)
    target_profiles = profiles or list(loader.profiles())
    if not target_profiles:
        raise ValueError("Konfigürasyonda çalıştırılabilir profil bulunamadı.")

    jobs = [
        (config_path or "config.yaml", profile, run_id, deterministic, overrides, log_to_console, output_root)
        for profile in target_profiles
    ]

    if processes is None or processes <= 0:
        max_workers = cpu_count() or 1
        processes = min(len(jobs), max_workers)

    if processes <= 1 or len(jobs) == 1:
        results = [_run_profile_job(job) for job in jobs]
    else:
        worker_count = min(processes, len(jobs))
        with Pool(processes=worker_count) as pool:
            results = pool.map(_run_profile_job, jobs)

    summary: Dict[str, Dict[str, Any]] = {}
    for profile, data in results:
        if data:
            summary[profile] = data

    return summary


__all__ = ["run_backtests"]
