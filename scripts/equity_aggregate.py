"""Backtest sonuçlarını birden fazla test senaryosu üzerinden kıyaslayan araç."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from crypto_rl.config.loader import load_config  # noqa: E402
from crypto_rl.pipelines.common import DictConfigAdapter  # noqa: E402
from data.data_loader import load_price_data  # noqa: E402
from data.split_utils import select_range  # noqa: E402


def summarize_predictions(df: pd.DataFrame) -> dict:
    rewards = df.get("reward")
    cumulative = df.get("portfolio_value")
    correct = df.get("correct")

    rewards_arr = rewards.astype(float).to_numpy() if rewards is not None else np.array([], dtype=float)
    cumulative_arr = cumulative.astype(float).to_numpy() if cumulative is not None else np.array([0.0], dtype=float)

    total_reward = float(cumulative_arr[-1]) if cumulative_arr.size else 0.0
    avg_reward = float(np.mean(rewards_arr)) if rewards_arr.size else 0.0
    reward_std = float(np.std(rewards_arr)) if rewards_arr.size else 0.0
    accuracy = float(correct.astype(float).mean()) if correct is not None else float("nan")

    return {
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "reward_std": reward_std,
        "accuracy": accuracy * 100.0 if np.isfinite(accuracy) else float("nan"),
        "steps": int(len(df)),
    }


def find_trades_logs(outputs_root: Path, run_id: str, target_profile: Optional[str] = None) -> Dict[str, List[Tuple[str, Path]]]:
    """Her senaryo için (profil, trades_log) eşlerini döndür."""
    mapping: Dict[str, List[Tuple[str, Path]]] = {}
    for profile_dir in outputs_root.glob(f"{run_id}_*"):
        profile = profile_dir.name.split(f"{run_id}_", 1)[-1]
        if target_profile and profile != target_profile:
            continue
        for container in ["results", "results_backtest"]:
            base = profile_dir / container
            if not base.exists():
                continue
            # Varsayılan olarak doğrudan results/trades_log.csv
            default_file = base / "trades_log.csv"
            if default_file.exists():
                mapping.setdefault("default", []).append((profile, default_file))
            for child in base.iterdir():
                if not child.is_dir():
                    continue
                candidate = child / "trades_log.csv"
                if candidate.exists():
                    mapping.setdefault(child.name, []).append((profile, candidate))
    return mapping


def _profile_candidates(profile: str) -> List[str]:
    """GA suffix'leri gibi varyasyonları profile eşleştirirken kullan."""
    chunks = profile.split("_")
    candidates: List[str] = []
    for idx in range(len(chunks)):
        candidate = "_".join(chunks[idx:])
        if candidate:
            candidates.append(candidate)
    return candidates


def _compute_price_series(
    config_loader,
    profile: str,
    scenario: str,
    target_coin: Optional[str],
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Config'ten senaryoya karşılık gelen fiyat serisini üret."""
    resolved = None
    if config_loader is None:
        return None, target_coin

    for candidate in _profile_candidates(profile):
        try:
            resolved = DictConfigAdapter(config_loader.resolved(candidate))
            break
        except KeyError:
            continue

    if resolved is None:
        print(f"Profil {profile!r} configte bulunamadı; fiyat serisi eklenmedi.")
        return None, target_coin

    coin_list = resolved.get("data.coin_list") or []
    coin = target_coin or (coin_list[0] if coin_list else None)
    if not coin:
        return None, None

    price_df = load_price_data(resolved)
    date_range = None
    if scenario != "default":
        date_range = (resolved.get("data.test_ranges") or {}).get(scenario)
    if not date_range:
        date_range = resolved.get("data.test_range")
    if not date_range:
        return None, coin

    test_df = select_range(price_df, date_range)
    coin_df = test_df[test_df["symbol"] == coin].copy()
    if coin_df.empty or "close" not in coin_df.columns:
        return None, coin

    coin_df = coin_df.sort_values("timestamp")
    series = coin_df["close"].reset_index(drop=True)
    return series, coin


def _align_price_series(series: pd.Series, target_length: int) -> Optional[np.ndarray]:
    if series is None or series.empty or target_length <= 0:
        return None
    data = series.to_numpy(dtype=float)
    if data.size == target_length - 1:
        data = np.concatenate(([data[0]], data))
    if data.size < target_length:
        pad = np.full(target_length - data.size, data[-1])
        data = np.concatenate((data, pad))
    elif data.size > target_length:
        data = data[:target_length]
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bir run altındaki backtest senaryolarını karşılaştırır.")
    parser.add_argument("--run-id", required=True, help="Eğitim/run kimliği (örn. 20251004_234628).")
    parser.add_argument("--outputs-root", default="outputs", help="Çıktı kökü (varsayılan: outputs).")
    parser.add_argument("--profile", default=None, help="Sadece belirtilen profili göster.")
    parser.add_argument("--save", action="store_true", help="Grafiği ve özet CSV'yi kaydet.")
    parser.add_argument("--config", default=None, help="Config dosyası (boş bırakılırsa otomatik aranır).")
    parser.add_argument("--coin", default=None, help="Fiyat serisi olarak eklenecek sembol.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs_root = Path(args.outputs_root)
    scenario_map = find_trades_logs(outputs_root, args.run_id, args.profile)
    if not scenario_map:
        raise SystemExit(f"Trades log bulunamadı: {outputs_root}/{args.run_id}_*/results*/trades_log.csv")

    config_loader = None
    if args.config is not False:
        try:
            config_loader = load_config(args.config)
        except Exception as exc:  # pragma: no cover
            print(f"Config yüklenemedi ({exc}); fiyat serisi eklenmeyecek.")
            config_loader = None

    profile_colors = {
        "aggressive": "#FF8C00",
        "balanced": "#1f77b4",
        "defensive": "#2E8B57",
    }

    scenario_profiles: Dict[str, List[Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for scenario in sorted(scenario_map.keys()):
        entries = sorted(scenario_map[scenario], key=lambda item: item[0])
        scenario_profiles_data: List[Dict[str, Any]] = []
        for profile, csv_path in entries:
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"[{scenario}/{profile}] trades_log boş, atlandı: {csv_path}")
                continue

            df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(0).astype(int)
            df["portfolio_value"] = pd.to_numeric(df["portfolio_value"], errors="coerce").fillna(method="ffill").fillna(method="bfill")
            df["portfolio_value"] = df["portfolio_value"].astype(float)
            if "reward" in df.columns:
                df["reward"] = pd.to_numeric(df["reward"], errors="coerce").fillna(0.0)
            if "correct" in df.columns:
                df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0.0)

            metrics = summarize_predictions(df)
            metrics.update(
                {
                    "scenario": scenario,
                    "profile": profile,
                    "trades_path": str(csv_path),
                }
            )
            summary_rows.append(metrics)

        correct_steps: List[int] = []
        incorrect_steps: List[int] = []
        if "correct" in df.columns:
            correct_mask = df["correct"].astype(float).fillna(0.0)
            correct_steps = df[correct_mask >= 0.5]["step"].astype(int).tolist()
            incorrect_steps = df[correct_mask < 0.5]["step"].astype(int).tolist()

        scenario_profiles_data.append(
            {
                "profile": profile,
                "equity": df["portfolio_value"].to_numpy(dtype=float),
                "steps": df["step"].to_numpy(dtype=int),
                "correct_steps": correct_steps,
                "incorrect_steps": incorrect_steps,
            }
        )

        if scenario_profiles_data:
            scenario_profiles[scenario] = scenario_profiles_data

    if not scenario_profiles:
        raise SystemExit("Gösterilecek senaryo veya profil bulunamadı.")

    scenario_names = list(scenario_profiles.keys())
    panel_entries: List[Tuple[str, Dict[str, Any]]] = []
    for scenario in sorted(scenario_profiles.keys()):
        for profile_info in scenario_profiles[scenario]:
            panel_entries.append((scenario, profile_info))

    fig, axes = plt.subplots(len(panel_entries), 1, figsize=(12, 5 * len(panel_entries)), sharex=False)
    if len(panel_entries) == 1:
        axes = [axes]  # type: ignore

    for ax, (scenario, profile_info) in zip(axes, panel_entries):
        profile = profile_info["profile"]
        equity = profile_info["equity"]
        steps = profile_info["steps"]
        color = profile_colors.get(profile.lower(), None)

        ax.plot(steps, equity, label=f"{profile} Portfolio", color=color or None)
        ax.set_title(f"{scenario} / {profile} - run {args.run_id}")
        ax.set_ylabel("Portfolio Value (USDT)")
        ax.grid(True)

        legend_handles, legend_labels = ax.get_legend_handles_labels()

        price_series = None
        price_label = None
        price_axis = None
        if config_loader is not None:
            series, label = _compute_price_series(config_loader, profile, scenario, args.coin)
            if series is not None:
                price_series = _align_price_series(series, len(equity))
                price_label = label or args.coin

        if price_series is not None:
            price_axis = ax.twinx()
            price_steps = np.arange(len(price_series))
            price_line = price_axis.plot(
                price_steps,
                price_series,
                color="black",
                linewidth=1.4,
                linestyle="--",
                label=f"{price_label or 'Price'}",
            )
            price_axis.set_ylabel(f"{price_label or 'Price'} (USDT)")
            legend_handles += price_line
            legend_labels += [price_label or "Price"]

        correct_steps = profile_info["correct_steps"]
        incorrect_steps = profile_info["incorrect_steps"]

        if correct_steps:
            clipped = [s for s in correct_steps if 0 <= s < len(equity)]
            if clipped:
                if price_axis is not None and price_series is not None:
                    buy_handle = price_axis.scatter(
                        clipped,
                        price_series[clipped],
                        marker="^",
                        s=36,
                        color="green",
                        alpha=0.75,
                        label="Correct",
                    )
                else:
                    buy_handle = ax.scatter(
                        clipped,
                        equity[clipped],
                        marker="^",
                        s=36,
                        color="green",
                        alpha=0.75,
                        label="Correct",
                    )
                legend_handles.append(buy_handle)
                legend_labels.append("Correct")
        if incorrect_steps:
            clipped = [s for s in incorrect_steps if 0 <= s < len(equity)]
            if clipped:
                if price_axis is not None and price_series is not None:
                    sell_handle = price_axis.scatter(
                        clipped,
                        price_series[clipped],
                        marker="v",
                        s=36,
                        color="red",
                        alpha=0.75,
                        label="Incorrect",
                    )
                else:
                    sell_handle = ax.scatter(
                        clipped,
                        equity[clipped],
                        marker="v",
                        s=36,
                        color="red",
                        alpha=0.75,
                        label="Incorrect",
                    )
                legend_handles.append(sell_handle)
                legend_labels.append("Incorrect")

        ax.legend(legend_handles, legend_labels)

    axes[-1].set_xlabel("Step")
    fig.tight_layout()

    print("\n=== SUMMARY ===")
    for scenario in scenario_names:
        rows = [row for row in summary_rows if row["scenario"] == scenario]
        if not rows:
            continue
        print(f"\n[{scenario}]")
        for row in rows:
            print(
                f"{row['profile']:>10} | "
                f"Reward: {row['total_reward']:.2f}  Avg: {row['avg_reward']:.4f}  "
                f"Std: {row['reward_std']:.4f}  Acc%: {row['accuracy']:.2f}  Steps: {row['steps']}"
            )

    if args.save:
        outdir = outputs_root / f"{args.run_id}_aggregate"
        outdir.mkdir(parents=True, exist_ok=True)
        fig_path = outdir / "portfolio_comparison.png"
        csv_path = outdir / "trades_summary.csv"

        fig.savefig(fig_path, dpi=160)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "scenario",
                    "profile",
                    "total_reward",
                    "avg_reward",
                    "reward_std",
                    "accuracy",
                    "steps",
                    "trades_path",
                ],
            )
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        print(f"\nSaved:\n  Figure : {fig_path}\n  Summary CSV : {csv_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
