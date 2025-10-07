"""PPO egitim komutunu moduler boru hattindan calistirir."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from crypto_rl.pipelines.training import run_training


def parse_args():
    """CLI argumanlarini toplar."""
    parser = argparse.ArgumentParser(description="PPO egitim boru hattini calistir.")
    parser.add_argument("--config", type=str, default=None, help="Konfigurasyon dosyasi yolu.")
    parser.add_argument("--profile", type=str, default=None, help="Egitim profili (aggressive/balanced/defensive vb.).")
    parser.add_argument("--run-id", type=str, default=None, help="Run kimligi (istege bagli).")
    return parser.parse_args()


def main():
    """Egitim boru hattini tetikler."""
    args = parse_args()
    artifacts = run_training(config_path=args.config, profile=args.profile, run_id=args.run_id)
    model_path = artifacts.get("model_path")
    log_path = artifacts.get("log_path")

    if model_path:
        print(f"[OK] Model kaydedildi: {model_path}")
    if log_path:
        print(f"[LOG] Egitim logu: {log_path}")


if __name__ == "__main__":
    main()
