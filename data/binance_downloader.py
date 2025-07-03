from binance.client import Client
import pandas as pd
from datetime import datetime
import time

def download_ohlcv(symbol, interval="1h", start_str="1 June 2021"):
    client = Client()
    print(f"⬇️ Downloading {symbol}...")
    klines = client.get_historical_klines(symbol, interval, start_str)
    
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "_", "_", "_", "_", "_", "_"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    df["symbol"] = symbol
    return df

def download_and_save_all():
    coin_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                 "ADAUSDT", "DOGEUSDT", "SHIBUSDT", "AVAXUSDT", "TRXUSDT"]
    all_data = []

    for symbol in coin_list:
        df = download_ohlcv(symbol, interval="1h", start_str="1 June 2021")
        all_data.append(df)
        time.sleep(1)  # Binance API rate limit

    full_df = pd.concat(all_data)
    full_df.to_parquet("data/processed/coin_data.parquet", index=False)
    print("✅ Saved to data/processed/coin_data.parquet")

if __name__ == "__main__":
    download_and_save_all()
