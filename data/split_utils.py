# data/split_utils.py

import pandas as pd
def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Veride timestamp sütununu datetime formatına çevir."""
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp":
            df = df.reset_index()
        else:
            raise ValueError("Veride timestamp ne index'te ne de sütun olarak mevcut.")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def select_range(df: pd.DataFrame, date_range) -> pd.DataFrame:
    """Belirtilen [başlangıç, bitiş] tarih aralığına göre veri alt kümesi döndür."""
    if not date_range or len(date_range) != 2:
        raise ValueError("Tarih aralığı [başlangıç, bitiş] formatında olmalıdır.")
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = _ensure_timestamp(df)
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    return df.loc[mask].copy()


def split_data(df: pd.DataFrame, config):
    """
    Çoklu coin'li ve çoklu feature'lı zaman serisini eğitim, doğrulama ve test aralıklarına ayırır.
    """
    df = _ensure_timestamp(df)
    train_df = select_range(df, config.get("data.train_range"))
    val_df = select_range(df, config.get("data.val_range"))
    test_df = select_range(df, config.get("data.test_range"))
    return train_df, val_df, test_df
