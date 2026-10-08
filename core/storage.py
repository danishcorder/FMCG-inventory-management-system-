import json
import pandas as pd
from config import CSV_PATH, METRICS_PATH, META_PATH


def load_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, parse_dates=["Date"])


def load_metrics() -> dict:
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_meta() -> dict:
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
