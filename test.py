import sys
import os
sys.path.append(os.path.abspath("."))

import pandas as pd
import numpy as np
from data.split_utils import split_data
import yaml

# Config yükle
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Veri yolunu configten al
data_path = config.get("data", {}).get("path", "data/processed/coin_data.parquet")
df = pd.read_parquet(data_path)

# Veriyi split et
train_df, val_df, test_df = split_data(df, config)

# Kontrol fonksiyonu
def check_split_integrity(split_name, df):
    print(f"\n🔍 Checking {split_name} set:")
    print(f"📌 Satır sayısı: {len(df)}")

    required_cols = [
        "timestamp", "symbol", "close", "open", "high", "low", "volume",
        "RSI", "MACD", "ADX", "ATR", "BOLLINGER_WIDTH", "OBV", "STOCH_K", "indicators"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Eksik sütunlar: {missing_cols}")
    else:
        print(f"✅ Tüm gerekli sütunlar mevcut.")

        # Boşluk oranları
        nan_ratios = df[required_cols].isna().mean()
        print("📉 NaN oranları (%):")
        print((nan_ratios * 100).round(2).sort_values(ascending=False))

        # indicators örnek kontrol
        sample = df["indicators"].iloc[0]
        print(f"📈 Örnek indicator tipi: {type(sample)}, içerik: {sample if isinstance(sample, list) else 'not list/array'}")

# Her split için kontrol et
check_split_integrity("train", train_df)
check_split_integrity("val", val_df)
check_split_integrity("test", test_df)
