# data/indicator_utils.py

import pandas_ta as ta
import pandas as pd
import numpy as np

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    close, high, low, volume içeren df'ye 7 teknik indikatörü hesaplayıp sütun olarak ekler.
    """
    df = df.copy()
    if len(df) < 21:
        print("⚠️  Uyarı: Çok az veri var, bazı teknik göstergeler eksik olabilir.")

    # RSI
    df["RSI"] = ta.rsi(df["close"], length=14)

    # MACD (histogram yerine MACD hattı alınır)
    macd = ta.macd(df["close"])
    df["MACD"] = macd["MACD_12_26_9"]

    # ADX
    adx = ta.adx(df["high"], df["low"], df["close"])
    df["ADX"] = adx["ADX_14"]

    # ATR
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"])

    bb = ta.bbands(df["close"], length=20, std=2.0)

    if bb is not None and all(col in bb.columns for col in ["BBU_20_2.0", "BBL_20_2.0"]):
        df["BOLLINGER_WIDTH"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]).fillna(0)
    else:
        print("❗ Uyarı: bbands hesaplanamadı. NaN atanacak.")
        df["BOLLINGER_WIDTH"] = pd.NA

    # OBV
    df["OBV"] = ta.obv(df["close"], df["volume"])

    # Stochastic Oscillator (%K kullanılır)
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df["STOCH_K"] = stoch["STOCHk_14_3_3"]

    # Toplu kontrol
    required = ["RSI", "MACD", "OBV", "BOLLINGER_WIDTH"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        print(f"❌ Eksik indikatör kolonları: {missing}")
    else:
        try:
            # Normalize: RSI (0–100) → 0–1, diğerleri z-score
            rsi = df["RSI"].fillna(0) / 100.0
            macd = (df["MACD"] - df["MACD"].mean()) / (df["MACD"].std() + 1e-8)
            obv = (df["OBV"] - df["OBV"].mean()) / (df["OBV"].std() + 1e-8)
            bw = (df["BOLLINGER_WIDTH"] - df["BOLLINGER_WIDTH"].mean()) / (df["BOLLINGER_WIDTH"].std() + 1e-8)

            df["indicators"] = np.stack([
                rsi, macd, obv, bw
            ], axis=1).tolist()

        except Exception as e:
            print(f"💥 indicators oluşturulurken hata: {e}")

    return df


# Hangi indikatörlerin çıktısı alınacak
compute_all_indicators.indicator_names = [
    "RSI", "MACD", "ADX", "ATR", "BOLLINGER_WIDTH", "OBV", "STOCH_K"
]
