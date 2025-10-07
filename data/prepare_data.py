# prepare_data.py

import pandas as pd
import numpy as np
import os
import pandas_ta as ta

INPUT_PATH = "data/processed/coin_data.parquet"
OUTPUT_PATH = "data/processed/coin_data.parquet"

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Veri bulunamadı: {INPUT_PATH}")

print("📦 Loading raw data...")
df = pd.read_parquet(INPUT_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

required_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Eksik sütun(lar): {missing_cols}")

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df) < 21:
        print("⚠️  Uyarı: Çok az veri var, bazı teknik göstergeler eksik olabilir.")

    # RSI (0–100)
    df["RSI"] = ta.rsi(df["close"], length=14)

    # MACD
    macd = ta.macd(df["close"])
    df["MACD"] = macd["MACD_12_26_9"]

    # ADX
    adx = ta.adx(df["high"], df["low"], df["close"])
    df["ADX"] = adx["ADX_14"]

    # ATR
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"])

    # Bollinger Band Width
    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is not None and all(col in bb.columns for col in ["BBU_20_2.0", "BBL_20_2.0"]):
        df["BOLLINGER_WIDTH"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]).fillna(0)
    else:
        df["BOLLINGER_WIDTH"] = pd.NA

    # OBV
    df["OBV"] = ta.obv(df["close"], df["volume"])

    # Stochastic Oscillator
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df["STOCH_K"] = stoch["STOCHk_14_3_3"]

    # Normalize
    try:
        df["RSI_N"] = df["RSI"].fillna(0) / 100.0
        df["MACD_N"] = (df["MACD"] - df["MACD"].mean()) / (df["MACD"].std() + 1e-8)
        df["OBV_N"] = (df["OBV"] - df["OBV"].mean()) / (df["OBV"].std() + 1e-8)
        df["BOLLINGER_WIDTH_N"] = (df["BOLLINGER_WIDTH"] - df["BOLLINGER_WIDTH"].mean()) / (df["BOLLINGER_WIDTH"].std() + 1e-8)
        df["ADX_N"] = (df["ADX"] - df["ADX"].mean()) / (df["ADX"].std() + 1e-8)
        df["ATR_N"] = (df["ATR"] - df["ATR"].mean()) / (df["ATR"].std() + 1e-8)
        df["STOCH_K_N"] = df["STOCH_K"].fillna(0) / 100.0

        df["indicators"] = np.stack([
            df["RSI_N"],
            df["MACD_N"],
            df["OBV_N"],
            df["BOLLINGER_WIDTH_N"],
            df["ADX_N"],
            df["ATR_N"],
            df["STOCH_K_N"]
        ], axis=1).tolist()

    except Exception as e:
        print(f"💥 Normalize edilirken hata: {e}")
        df["indicators"] = [[] for _ in range(len(df))]

    return df


print("📊 Computing technical indicators...")

processed = []
for sym, group in df.groupby("symbol"):
    group = group.copy()
    group["symbol"] = sym
    group["timestamp"] = pd.to_datetime(group["timestamp"])
    enriched = compute_all_indicators(group)

    enriched = enriched[[
        "timestamp", "symbol", "close", "high", "low", "open", "volume",
        "RSI", "MACD", "ADX", "ATR", "BOLLINGER_WIDTH", "OBV", "STOCH_K",
        "indicators"
    ]]

    processed.append(enriched)

df = pd.concat(processed).reset_index(drop=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by=["timestamp", "symbol"]).reset_index(drop=True)

df.to_parquet(OUTPUT_PATH, index=False)
print(f"✅ Veriler işlendi ve şuraya kaydedildi: {OUTPUT_PATH}")
