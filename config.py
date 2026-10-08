from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CSV_PATH = DATA_DIR / "synthetic_inventory.csv"
DB_PATH = DATA_DIR / "inventory.db"
METRICS_PATH = MODELS_DIR / "metrics.json"
META_PATH = MODELS_DIR / "meta.json"
ANOMALY_CSV_PATH = DATA_DIR / "synthetic_anomaly_inventory.csv"
ANOMALY_METRICS_PATH = MODELS_DIR / "anomaly_metrics.json"
ANOMALY_META_PATH = MODELS_DIR / "anomaly_meta.json"
RANDOM_SEED = 42

RISK_THRESHOLDS = {
    "LOW": 0.30,
    "MEDIUM": 0.60,
    "HIGH": 0.80,
}

USERS = {
    "platform_admin": {"password": "admin123", "modules": ["stockout", "demand", "anomaly"]},
    "stockout_admin": {"password": "admin123", "modules": ["stockout"]},
    "demand_admin": {"password": "admin123", "modules": ["demand"]},
    "anomaly_admin": {"password": "admin123", "modules": ["anomaly"]},
}
