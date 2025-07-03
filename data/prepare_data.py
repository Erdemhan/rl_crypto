# prepare_data.py

import pandas as pd
import numpy as np
import os
from indicator_utils import compute_all_indicators

INPUT_PATH = "data/processed/coin_data.parquet"
OUTPUT_PATH = "data/processed/coin_data.parquet"

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Veri bulunamadı: {INPUT_PATH}")

print("📦 Loading raw data...")
df = pd.read_parquet(INPUT_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")


# Gerekli sütunların kontrolü
required_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Eksik sütun(lar): {missing_cols}")

print("📊 Computing technical indicators...")

processed = []
for sym, group in df.groupby("symbol"):
    group = group.copy()
    group["symbol"] = sym  # Sütun olarak kalsın
    group["timestamp"] = pd.to_datetime(group["timestamp"])

    enriched = compute_all_indicators(group)

    enriched["indicators"] = np.stack([
        enriched["RSI"].fillna(0),
        enriched["MACD"].fillna(0),
        enriched["OBV"].fillna(0),
        enriched["BOLLINGER_WIDTH"].fillna(0),
        enriched["ADX"].fillna(0),
        enriched["ATR"].fillna(0),
        enriched["STOCH_K"].fillna(0),
    ], axis=1).tolist()

    # enriched = enriched[["timestamp", "symbol", "close", "high", "low", "open", "volume", "indicators"]]
    enriched = enriched[["timestamp", "symbol", "close", "high", "low", "open", "volume"] + compute_all_indicators.indicator_names + ["indicators"]]

    processed.append(enriched)

df = pd.concat(processed).reset_index(drop=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by=["timestamp", "symbol"]).reset_index(drop=True)


df.to_parquet(OUTPUT_PATH, index=False)
print(f"✅ Veriler işlendi ve şuraya kaydedildi: {OUTPUT_PATH}")


def check_enriched_integrity(df):
    required_columns = [
        "timestamp", "symbol", "close", "open", "high", "low", "volume",
        "RSI", "MACD", "ADX", "ATR", "BOLLINGER_WIDTH", "OBV", "STOCH_K",
        "indicators"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    empty_columns = [col for col in required_columns if col in df.columns and df[col].isna().all()]

    result = {
        "missing_columns": missing_columns,
        "empty_columns": empty_columns,
        "row_count": len(df),
        "is_empty": df.empty
    }
