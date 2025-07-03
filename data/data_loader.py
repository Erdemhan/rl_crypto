# data/data_loader.py

import pandas as pd
from pathlib import Path
from data.indicator_utils import compute_all_indicators

def load_price_data(config):
    file_path = Path(config.get("data.offline_data_path"))

    # Dosya uzantısına göre oku
    df = pd.read_parquet(file_path) if file_path.suffix == ".parquet" and file_path else pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")



    coin_list = config.get("data.coin_list")

    all_coin_data = []
    for coin in coin_list:
        coin_df = df[df["symbol"] == coin].copy()

        # İlgili sütunları seç
        coin_df["symbol"] = coin
        coin_df["timestamp"] = pd.to_datetime(coin_df["timestamp"])
        coin_df = coin_df[[
            "timestamp", "symbol", "close", "high", "low", "open", "volume"] + compute_all_indicators.indicator_names + ["indicators"]]
        all_coin_data.append(coin_df)

    # Tüm coinleri uzun formatta birleştir
    merged = pd.concat(all_coin_data).reset_index(drop=True)
    merged = merged.sort_values(by=["timestamp", "symbol"]).reset_index(drop=True)

    # Eksik satırları at (indikatörler için ilk 20 adım boş olabilir)
    merged.dropna(inplace=True)


    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="raise")


    return merged
