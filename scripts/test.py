"""Egitilen modeller icin backtest CLI komutu."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from crypto_rl.pipelines.testing import run_backtests


def parse_args():
    """CLI argumanlarini tanimlar."""
    parser = argparse.ArgumentParser(description="Kayitli PPO modellerini backtest eder.")
    parser.add_argument("--config", type=str, default=None, help="Config dosyasi (varsayilan otomatik aranir).")
    parser.add_argument("--run-id", type=str, required=True, help="Backtest edilecek egitim kosusunun kimligi.")
    parser.add_argument(
        "--deterministic",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Politikanin deterministik davranip davranmayacagi (config uzerine yazar).",
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=0,
        help="Paralel calistirilacak islem sayisi (0 => otomatik).",
    )
    return parser.parse_args()


def _to_bool(value: str | None) -> Optional[bool]:
    if value is None:
        return None
    return value.lower() in {"true", "1", "yes", "y"}


def main():
    """Backtest boru hattini tetikler."""
    args = parse_args()
    run_backtests(
        config_path=args.config,
        run_id=args.run_id,
        deterministic=_to_bool(args.deterministic),
        profiles=None,
        processes=args.procs,
    )


if __name__ == "__main__":
    main()
