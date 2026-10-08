import pandas as pd
from config import CSV_PATH, METRICS_PATH, META_PATH
from core.data_generator import generate_synthetic_data
from core.modeling import train_all
from core.anomaly_modeling import ensure_anomaly_initialized


def ensure_initialized(force: bool = False):
    if force or not CSV_PATH.exists():
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    if force or not METRICS_PATH.exists() or not META_PATH.exists():
        train_all(df)
    ensure_anomaly_initialized(df, force=force)
    return df


if __name__ == "__main__":
    data = ensure_initialized(force=False)
    print(f"Project ready with {len(data):,} rows")
