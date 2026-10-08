from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from config import (
    ANOMALY_CSV_PATH,
    ANOMALY_METRICS_PATH,
    ANOMALY_META_PATH,
    MODELS_DIR,
    RANDOM_SEED,
)
from core.anomaly_data import enrich_and_inject_anomalies, explain_anomaly_row, business_impact


ANOMALY_FEATURES = [
    "Daily_Stock_Change",
    "Pct_Stock_Change",
    "Stock_Movement_Velocity",
    "Rolling_Sales_Mean_7",
    "Rolling_Sales_Std_28",
    "Demand_Variation",
    "Stock_To_Sales_Ratio",
    "Adjustment_Ratio",
    "Return_Percentage",
    "Damage_Percentage",
    "Inventory_Turnover_Proxy",
    "Inventory_Value",
    "Stock_Mismatch_Qty",
    "Supplier_Lead_Time_Deviation",
    "Short_Delivery_Ratio",
    "Warehouse_Level_Deviation",
    "Category_Level_Deviation",
    "Warehouse_Capacity_Utilization",
    "Stock_Age_Days",
    "Days_To_Expiry",
    "Days_of_Inventory",
    "Transfer_Mismatch_Qty",
    "Duplicate_Transaction_Flag",
    "Late_Adjustment_Flag",
]

MODEL_NAMES = [
    "Hybrid: Isolation Forest + Rules",
    "Isolation Forest",
    "Local Outlier Factor",
    "One-Class SVM",
    "Autoencoder",
    "Statistical Rules",
]

SCORE_COLUMNS = {
    "Hybrid: Isolation Forest + Rules": "Anomaly_Score_Hybrid",
    "Isolation Forest": "Anomaly_Score_Isolation_Forest",
    "Local Outlier Factor": "Anomaly_Score_LOF",
    "One-Class SVM": "Anomaly_Score_OCSVM",
    "Autoencoder": "Anomaly_Score_Autoencoder",
    "Statistical Rules": "Anomaly_Score_Rules",
}


def _safe_matrix(df: pd.DataFrame, imputer=None, scaler=None, fit=False):
    x = df[ANOMALY_FEATURES].replace([np.inf, -np.inf], np.nan)
    if fit:
        imputer = SimpleImputer(strategy="median")
        x_imp = imputer.fit_transform(x)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_imp)
        return x_scaled, imputer, scaler
    x_imp = imputer.transform(x)
    return scaler.transform(x_imp)


def _normalize_score(raw: np.ndarray, center: float, high: float) -> np.ndarray:
    denom = max(float(high - center), 1e-9)
    score = 12.0 + 88.0 * (np.asarray(raw, dtype=float) - center) / denom
    return np.clip(score, 0, 100)


def severity_label(score: float) -> str:
    if score <= 30:
        return "NORMAL"
    if score <= 60:
        return "MODERATE"
    if score <= 80:
        return "HIGH"
    return "CRITICAL"


def _metric_block(y_true, score, execution_time):
    pred = (np.asarray(score) > 30).astype(int)
    y = np.asarray(y_true).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Detected_Anomalies": int(pred.sum()),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "False_Positive_Rate": float(fp / max(fp + tn, 1)),
        "Execution_Time_Seconds": float(execution_time),
        "Confusion_Matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _prepare_shared_base(base: pd.DataFrame) -> pd.DataFrame:
    """Keep the anomaly prototype within an industry-demo transaction size."""
    d = base.copy()
    max_date = pd.to_datetime(d["Date"]).max()
    d = d[pd.to_datetime(d["Date"]) >= max_date - pd.Timedelta(days=179)].copy()

    # Shared platform currently has 100 SKUs. Use 90 SKUs to stay near 50k rows
    # while keeping contiguous time histories for feature engineering.
    sku_keep = sorted(d["SKU_ID"].unique())[:90]
    d = d[d["SKU_ID"].isin(sku_keep)].copy()
    return d.sort_values(["SKU_ID", "Warehouse_ID", "Date"]).reset_index(drop=True)


def train_anomaly_models(base: pd.DataFrame, force: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ANOMALY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    prototype = _prepare_shared_base(base)
    df = enrich_and_inject_anomalies(prototype, seed=RANDOM_SEED)

    dates = np.array(sorted(pd.to_datetime(df["Date"]).unique()))
    split_date = dates[max(1, int(len(dates) * 0.70)) - 1]
    train_mask = (pd.to_datetime(df["Date"]) <= split_date).to_numpy()
    test_mask = (pd.to_datetime(df["Date"]) > split_date).to_numpy()
    train = df.loc[train_mask].copy()

    # Fit preprocessing only on chronological training data.
    x_train_all, imputer, scaler = _safe_matrix(train, fit=True)
    x_all = _safe_matrix(df, imputer=imputer, scaler=scaler, fit=False)

    # Unsupervised models should learn mostly normal behavior.
    normal_train_mask = train["Synthetic_Anomaly_Label"].to_numpy() == 0
    normal_x = x_train_all[normal_train_mask]
    rng = np.random.default_rng(RANDOM_SEED)

    def sampled(x, n):
        if len(x) <= n:
            return x
        idx = rng.choice(len(x), size=n, replace=False)
        return x[idx]

    model_specs = {}
    timings = {}
    raw_scores = {}

    # Isolation Forest
    t0 = time.perf_counter()
    iforest = IsolationForest(
        n_estimators=180,
        contamination=0.06,
        max_samples=min(12000, max(256, len(normal_x))),
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iforest.fit(sampled(normal_x, 22000))
    raw_scores["Isolation Forest"] = -iforest.decision_function(x_all)
    timings["Isolation Forest"] = time.perf_counter() - t0
    model_specs["Isolation Forest"] = iforest
    joblib.dump(iforest, MODELS_DIR / "anomaly_isolation_forest.joblib")

    # Local Outlier Factor, novelty mode to score unseen/current records.
    t0 = time.perf_counter()
    lof = LocalOutlierFactor(
        n_neighbors=25,
        novelty=True,
        contamination=0.06,
        metric="minkowski",
        n_jobs=-1,
    )
    lof.fit(sampled(normal_x, 5500))
    raw_scores["Local Outlier Factor"] = -lof.decision_function(x_all)
    timings["Local Outlier Factor"] = time.perf_counter() - t0
    model_specs["Local Outlier Factor"] = lof
    joblib.dump(lof, MODELS_DIR / "anomaly_lof.joblib")

    # One-Class SVM: fit on a compact normal sample to keep the prototype responsive.
    t0 = time.perf_counter()
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.06, cache_size=500)
    ocsvm.fit(sampled(normal_x, 2600))
    raw_scores["One-Class SVM"] = -ocsvm.decision_function(x_all).ravel()
    timings["One-Class SVM"] = time.perf_counter() - t0
    model_specs["One-Class SVM"] = ocsvm
    joblib.dump(ocsvm, MODELS_DIR / "anomaly_ocsvm.joblib")

    # Lightweight autoencoder using sklearn MLP: X -> compressed hidden representation -> X.
    t0 = time.perf_counter()
    auto_x = sampled(normal_x, 7000)
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(14, 5, 14),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=0.002,
        max_iter=70,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=RANDOM_SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        autoencoder.fit(auto_x, auto_x)
    reconstructed = autoencoder.predict(x_all)
    raw_scores["Autoencoder"] = np.mean((x_all - reconstructed) ** 2, axis=1)
    timings["Autoencoder"] = time.perf_counter() - t0
    model_specs["Autoencoder"] = autoencoder
    joblib.dump(autoencoder, MODELS_DIR / "anomaly_autoencoder.joblib")

    # Save shared preprocessing.
    joblib.dump(
        {"imputer": imputer, "scaler": scaler, "features": ANOMALY_FEATURES},
        MODELS_DIR / "anomaly_preprocessor.joblib",
    )

    # Calibrate each ML score using TRAINING distribution only.
    calibrated = {}
    calibrations = {}
    for name in ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Autoencoder"]:
        ref = np.asarray(raw_scores[name][train_mask], dtype=float)
        center = float(np.quantile(ref, 0.50))
        high = float(np.quantile(ref, 0.995))
        if high <= center:
            high = float(np.quantile(ref, 0.99)) + 1e-6
        calibrated[name] = _normalize_score(raw_scores[name], center, high)
        calibrations[name] = {"center_q50": center, "high_q995": high}

    # Transparent business rule baseline.
    calibrated["Statistical Rules"] = df["Business_Rule_Score"].to_numpy(dtype=float)

    # Hybrid recommendation from the project brief: Isolation Forest + business rules.
    calibrated["Hybrid: Isolation Forest + Rules"] = np.maximum(
        0.56 * calibrated["Isolation Forest"] + 0.44 * calibrated["Statistical Rules"],
        0.90 * calibrated["Statistical Rules"],
    )
    calibrated["Hybrid: Isolation Forest + Rules"] = np.clip(
        calibrated["Hybrid: Isolation Forest + Rules"], 0, 100
    )
    timings["Statistical Rules"] = 0.0
    timings["Hybrid: Isolation Forest + Rules"] = timings["Isolation Forest"]

    for model_name, col in SCORE_COLUMNS.items():
        df[col] = calibrated[model_name]

    # Explainable anomaly fields used by every dashboard/model.
    explained = df.apply(explain_anomaly_row, axis=1)
    df["Anomaly_Type"] = explained.map(lambda x: x[0])
    df["Anomaly_Category"] = explained.map(lambda x: x[1])
    df["Anomaly_Reason"] = explained.map(lambda x: x[2])
    df["Recommended_Action"] = explained.map(lambda x: x[3])
    df["Business_Impact"] = df["Anomaly_Type"].map(business_impact)

    # Evaluate on newest chronological holdout against known injected anomalies.
    metrics = {}
    y_test = df.loc[test_mask, "Synthetic_Anomaly_Label"].to_numpy()
    for model_name in MODEL_NAMES:
        metrics[model_name] = _metric_block(
            y_test,
            calibrated[model_name][test_mask],
            timings.get(model_name, 0.0),
        )

    # Add per-model severity columns for responsive dashboard switching.
    for model_name, col in SCORE_COLUMNS.items():
        sev_col = "Severity_" + col.replace("Anomaly_Score_", "")
        df[sev_col] = df[col].map(severity_label)

    meta = {
        "models": MODEL_NAMES,
        "score_columns": SCORE_COLUMNS,
        "features": ANOMALY_FEATURES,
        "split_date": str(pd.Timestamp(split_date).date()),
        "rows": int(len(df)),
        "skus": int(df["SKU_ID"].nunique()),
        "warehouses": int(df["Warehouse_ID"].nunique()),
        "suppliers": int(df["Supplier_ID"].nunique()),
        "synthetic_anomaly_rate": float(df["Synthetic_Anomaly_Label"].mean()),
        "calibrations": calibrations,
        "scope_note": "Shared synthetic master data; anomaly prototype uses recent contiguous history and controlled anomaly injection.",
    }

    df.to_csv(ANOMALY_CSV_PATH, index=False)
    with open(ANOMALY_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(ANOMALY_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return df, metrics, meta


def ensure_anomaly_initialized(base: pd.DataFrame, force: bool = False):
    if (
        force
        or not ANOMALY_CSV_PATH.exists()
        or not ANOMALY_METRICS_PATH.exists()
        or not ANOMALY_META_PATH.exists()
    ):
        train_anomaly_models(base, force=True)


def load_anomaly_data() -> pd.DataFrame:
    return pd.read_csv(
        ANOMALY_CSV_PATH,
        parse_dates=["Date", "Expiry_Date"],
        low_memory=False,
    )


def load_anomaly_metrics() -> dict:
    with open(ANOMALY_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_anomaly_meta() -> dict:
    with open(ANOMALY_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def score_column(model_name: str) -> str:
    return SCORE_COLUMNS[model_name]


def severity_column(model_name: str) -> str:
    return "Severity_" + SCORE_COLUMNS[model_name].replace("Anomaly_Score_", "")
