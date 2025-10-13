"""Runs genetic algorithm based hyperparameter search for a single PPO profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from crypto_rl.config.loader import load_config  # noqa: E402
from crypto_rl.optimization.ga import GAOptimizer  # noqa: E402


def _load_ga_config(config_path: str | None, profile: str) -> Dict[str, Any]:
    loader = load_config(config_path)
    resolved = loader.resolved(profile)
    ga_cfg = resolved.get("optimization", {}).get("ga")
    if not ga_cfg:
        raise ValueError(
            "GA ayarları bulunamadı. Lütfen configs/config.yaml içinde 'optimization.ga' bloğunu tanımlayın."
        )
    return dict(ga_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genetik algoritma ile PPO hiperparametre optimizasyonu")
    parser.add_argument("--config", default=None, help="Config dosyası (varsayılan otomatik bulunur)")
    parser.add_argument("--profile", default="balanced", help="Optimize edilecek profil adı")
    parser.add_argument("--generations", type=int, help="Jenerasyon sayısı (override)")
    parser.add_argument("--population", type=int, help="Popülasyon boyutu (override)")
    parser.add_argument("--max-workers", type=int, help="Paralel değerlendirilecek birey sayısı")
    parser.add_argument("--vol-weight", type=float, help="Stabilite cezası katsayısı (volatility weight)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ga_cfg = _load_ga_config(args.config, args.profile)

    if args.generations:
        ga_cfg["generations"] = args.generations
    if args.population:
        ga_cfg["population_size"] = args.population
    if args.max_workers:
        ga_cfg["max_workers"] = args.max_workers
    if args.vol_weight is not None:
        ga_cfg["volatility_weight"] = args.vol_weight

    optimizer = GAOptimizer(config_path=args.config, profile=args.profile, ga_config=ga_cfg)
    best_result = optimizer.run()

    print("\n=== GA OPTIMIZATION COMPLETE ===")
    print(f"Best fitness   : {best_result.fitness:.6f}")
    print(f"Growth         : {best_result.growth:.6f}")
    print(f"Volatility     : {best_result.volatility:.6f}")
    print(f"Run ID         : {best_result.run_id}")
    print("Best parameters:")
    for key, value in sorted(best_result.individual.items()):
        print(f"  {key}: {value:.6f}")

    output_dir = optimizer.output_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "best_candidate.json"
    payload = {
        "fitness": best_result.fitness,
        "growth": best_result.growth,
        "volatility": best_result.volatility,
        "run_id": best_result.run_id,
        "parameters": best_result.individual,
        "metadata": best_result.meta,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nÖzet kaydedildi: {summary_path}")
    print("En iyi parametrelerle tam eğitim yapmak için run_training fonksiyonunu bu değerlerle override edebilirsiniz.")


if __name__ == "__main__":
    main()
