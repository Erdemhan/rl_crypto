# data/split_utils.py

import pandas as pd
import numpy as np

def split_data(df, config):
    """
    Çoklu coin'li ve çoklu feature'lı zaman serisini eğitim, doğrulama ve test aralıklarına ayırır.
    """


    # timestamp düzeltme
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp":
            df.reset_index(inplace=True)
        else:
            raise ValueError("Veride timestamp ne index'te ne de sütun olarak mevcut.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_start, train_end = config.get("data.train_range")
    val_start, val_end = config.get("data.val_range")
    test_start, test_end = config.get("data.test_range")
    # 👇 Zaman filtreleme düzgün çalışsın diye stringleri datetime nesnesine çeviriyoruz
    train_start = pd.to_datetime(train_start)
    train_end = pd.to_datetime(train_end)
    val_start = pd.to_datetime(val_start)
    val_end = pd.to_datetime(val_end)
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)


    # Eğer timestamp sütun değilse index'ten çek
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp":
            df = df.reset_index()


    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_df = df[(df["timestamp"] >= train_start) & (df["timestamp"] <= train_end)].copy()
    val_df   = df[(df["timestamp"] >= val_start) & (df["timestamp"] <= val_end)].copy()
    test_df  = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)].copy()

    return train_df, val_df, test_df
