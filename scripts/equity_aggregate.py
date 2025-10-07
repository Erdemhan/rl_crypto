# scripts/equity_aggregate.py
# Çoklu profil equity_curve analiz + tek grafikte karşılaştırma
# Kullanım:
#   python -m scripts.equity_aggregate --run-id 20251004_234628
# Opsiyonel:
#   python -m scripts.equity_aggregate --run-id 20251004_234628 --save
#   python -m scripts.equity_aggregate --run-id 20251004_234628 --outputs-root outputs

import argparse
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_sharpe(returns: np.ndarray, eps: float = 1e-8) -> float:
    r = np.asarray(returns, dtype=float)
    mu = np.nanmean(r)
    sd = np.nanstd(r)
    if not np.isfinite(sd) or sd < eps:
        return 0.0
    s = mu / (sd + eps)
    return float(s) if np.isfinite(s) else 0.0


def max_drawdown(equity: np.ndarray, eps: float = 1e-8) -> float:
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / (peak + eps)
    mdd = float(np.nanmax(dd)) if dd.size else 0.0
    return 100.0 * mdd


def analyze_equity(df: pd.DataFrame) -> dict:
    equity = df["portfolio_value"].astype(float).to_numpy()
    initial = float(equity[0])
    final = float(equity[-1])
    net_profit = final - initial
    return_pct = (final / initial - 1.0) * 100.0
    # step-to-step simple returns:
    ret = np.diff(equity) / equity[:-1]
    sharpe = safe_sharpe(ret)
    mdd_pct = max_drawdown(equity)
    return {
        "initial_value": initial,
        "final_value": final,
        "net_profit": net_profit,
        "return_pct": return_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": mdd_pct,
    }


def find_equity_paths(outputs_root: Path, run_id: str):
    """
    Arar:
      outputs/<run_id>_*/results_backtest/equity_curve.csv
      outputs/<run_id>_*/results/equity_curve.csv (fallback)
    """
    paths = []
    for p in outputs_root.glob(f"{run_id}_*"):
        profile = p.name.split(f"{run_id}_", 1)[-1]
        cand1 = p / "results_backtest" / "equity_curve.csv"
        cand2 = p / "results" / "equity_curve.csv"
        if cand1.exists():
            paths.append((profile, cand1))
        elif cand2.exists():
            paths.append((profile, cand2))
    return paths


def main():
    ap = argparse.ArgumentParser(description="Multi-profile equity analysis & comparison plot")
    ap.add_argument("--run-id", required=True, help="Eğitim timestamp’ı (örn. 20251004_234628)")
    ap.add_argument("--outputs-root", default="outputs", help="Çıktı kökü (default: outputs)")
    ap.add_argument("--save", action="store_true", help="Grafik ve özet CSV kaydet")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    paths = find_equity_paths(outputs_root, args.run_id)
    if not paths:
        raise SystemExit(f"Equity curve bulunamadı: {outputs_root}/{args.run_id}_*/results*/equity_curve.csv")

    summary = []
    plt.figure(figsize=(12, 6))

    for profile, csv_path in sorted(paths):
        df = pd.read_csv(csv_path)
        actions_path = csv_path.parent / "actions.csv"
        actions_df = None
        if actions_path.exists():
            actions_df = pd.read_csv(actions_path)
            for col in ("invalid_sell", "redundant_buy"):
                if col in actions_df.columns:
                    actions_df[col] = pd.to_numeric(actions_df[col], errors="coerce").fillna(0).astype(int)

        if "portfolio_value" not in df.columns:
            print(f"[{profile}] Uyarı: 'portfolio_value' kolonu yok: {csv_path}")
            continue

        metrics = analyze_equity(df)
        summary.append({
            "profile": profile,
            **metrics,
            "equity_path": str(csv_path),
        })

        # Tek grafikte tüm profillerin eğrileri (yalnız matplotlib; renk belirtme yok)
        plt.plot(df.index.to_numpy(), df["portfolio_value"].to_numpy(), label=profile)
        # aynı klasördeki actions.csv varsa oku ve marker bas
        actions_path = csv_path.parent / "actions.csv"
        if actions_path.exists():
            actions_df = pd.read_csv(actions_path)
            for col in ("invalid_sell", "redundant_buy"):
                if col in actions_df.columns:
                    actions_df[col] = pd.to_numeric(actions_df[col], errors="coerce").fillna(0).astype(int)

            if "step" in actions_df.columns and "portfolio_value" in actions_df.columns:
                buys = actions_df[actions_df["action"].str.startswith("BUY", na=False)]
                sells = actions_df[actions_df["action"] == "SELL"]

                if "invalid_sell" in actions_df.columns:
                    sells = sells[sells["invalid_sell"] == 0]

                if not buys.empty:
                    plt.scatter(
                        buys["step"].to_numpy(),
                        buys["portfolio_value"].to_numpy(),
                        marker="^",
                        s=28,
                        label=f"{profile} BUY (markers)",
                    )
                if not sells.empty:
                    plt.scatter(
                        sells["step"].to_numpy(),
                        sells["portfolio_value"].to_numpy(),
                        marker="v",
                        s=28,
                        label=f"{profile} SELL (markers)",
                    )

    plt.title(f"Equity Curves — run {args.run_id}")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value (USDT)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Konsol özeti
    print("\n=== SUMMARY ===")
    for row in summary:
        print(
            f"{row['profile']:>10} | "
            f"Init: {row['initial_value']:.2f}  Final: {row['final_value']:.2f}  "
            f"Net: {row['net_profit']:.2f}  Ret%: {row['return_pct']:.2f}  "
            f"Sharpe: {row['sharpe']:.4f}  MDD%: {row['max_drawdown_pct']:.2f}"
        )

    if args.save:
        outdir = outputs_root / f"{args.run_id}_aggregate"
        outdir.mkdir(parents=True, exist_ok=True)
        fig_path = outdir / "equity_comparison.png"
        csv_path = outdir / "equity_summary.csv"

        plt.savefig(fig_path, dpi=160)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "profile",
                    "initial_value",
                    "final_value",
                    "net_profit",
                    "return_pct",
                    "sharpe",
                    "max_drawdown_pct",
                    "equity_path",
                ],
            )
            writer.writeheader()
            for row in summary:
                writer.writerow(row)

        print(f"\nKaydedildi:\n  Grafik : {fig_path}\n  Özet CSV : {csv_path}")
    else:
        # Sadece göster
        plt.show()


if __name__ == "__main__":
    main()
