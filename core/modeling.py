from __future__ import annotations
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score,
)

from config import MODELS_DIR, METRICS_PATH, META_PATH, RANDOM_SEED
from core.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FEATURES


def slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def _preprocessor(scale_numeric: bool = False):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")


def split_by_date(df: pd.DataFrame):
    dates = np.array(sorted(pd.to_datetime(df["Date"]).unique()))
    n = len(dates)
    train_end = dates[max(0, int(n * 0.70) - 1)]
    val_end = dates[max(0, int(n * 0.85) - 1)]
    d = pd.to_datetime(df["Date"])
    train = df[d <= train_end].copy()
    val = df[(d > train_end) & (d <= val_end)].copy()
    test = df[d > val_end].copy()
    return train, val, test, str(pd.Timestamp(train_end).date()), str(pd.Timestamp(val_end).date())


def stockout_metrics(y_true, prob, threshold: float = 0.5):
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else None,
        "PR_AUC": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else None,
        "Confusion_Matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def demand_metrics(y_true, pred):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.maximum(0, np.asarray(pred, dtype=float))
    return {
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(mean_squared_error(y_true, pred) ** 0.5),
        "WMAPE": float(np.abs(y_true - pred).sum() / max(np.abs(y_true).sum(), 1.0)),
        "Forecast_Bias": float((pred - y_true).sum() / max(np.abs(y_true).sum(), 1.0)),
        "R2": float(r2_score(y_true, pred)),
    }


def train_all(df: pd.DataFrame) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test, train_end, val_end = split_by_date(df)
    # Keep the chronological holdout intact; cap the fit sample for a responsive local prototype.
    fit_train = train.sample(min(len(train), 55000), random_state=RANDOM_SEED) if len(train) > 55000 else train
    x_train, x_test = fit_train[FEATURES], test[FEATURES]

    stock_models = {
        "Logistic Regression": Pipeline([
            ("prep", _preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=600, class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline([
            ("prep", _preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=45, max_depth=14, min_samples_leaf=4,
                class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED
            )),
        ]),
        "Hist Gradient Boosting": Pipeline([
            ("prep", _preprocessor()),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.07, max_iter=100, max_leaf_nodes=31,
                l2_regularization=0.5, random_state=RANDOM_SEED
            )),
        ]),
    }
    demand_models = {
        "Random Forest": Pipeline([
            ("prep", _preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=45, max_depth=16, min_samples_leaf=3,
                n_jobs=-1, random_state=RANDOM_SEED
            )),
        ]),
        "Hist Gradient Boosting": Pipeline([
            ("prep", _preprocessor()),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.07, max_iter=105, max_leaf_nodes=31,
                l2_regularization=0.3, random_state=RANDOM_SEED
            )),
        ]),
    }
    metrics = {
        "stockout": {}, "demand": {},
        "split": {
            "train_end": train_end, "validation_end": val_end,
            "train_rows": int(len(train)), "validation_rows": int(len(val)), "test_rows": int(len(test)),
        },
    }

    y_train_cls, y_test_cls = fit_train["Stockout_Next_7D"], test["Stockout_Next_7D"]
    trained_stock_models = []
    for name, model in stock_models.items():
        model.fit(x_train, y_train_cls)
        prob = model.predict_proba(x_test)[:, 1]
        metrics["stockout"][name] = stockout_metrics(y_test_cls, prob)
        joblib.dump(model, MODELS_DIR / f"stockout_{slug(name)}.joblib")
        trained_stock_models.append(name)

    # CatBoost is optional. Some Python/sklearn combinations are incompatible;
    # failure must never prevent the core application from running.
    try:
        from catboost import CatBoostClassifier
        cb = Pipeline([
            ("prep", _preprocessor()),
            ("model", CatBoostClassifier(
                iterations=120, depth=7, learning_rate=0.06, verbose=False,
                random_seed=RANDOM_SEED, auto_class_weights="Balanced", allow_writing_files=False
            )),
        ])
        cb.fit(x_train, y_train_cls)
        prob = cb.predict_proba(x_test)[:, 1]
        metrics["stockout"]["CatBoost"] = stockout_metrics(y_test_cls, prob)
        joblib.dump(cb, MODELS_DIR / "stockout_catboost.joblib")
        trained_stock_models.append("CatBoost")
    except Exception as exc:
        metrics["stockout_optional_note"] = f"CatBoost skipped: {type(exc).__name__}"

    y_train_reg, y_test_reg = fit_train["Demand_Next_7D"], test["Demand_Next_7D"]
    baseline = np.maximum(0, test["Avg_Daily_Sales_7D"].to_numpy() * 7.0)
    metrics["demand"]["Moving Average"] = demand_metrics(y_test_reg, baseline)

    trained_demand_models = ["Moving Average"]
    for name, model in demand_models.items():
        model.fit(x_train, y_train_reg)
        pred = np.maximum(0, model.predict(x_test))
        metrics["demand"][name] = demand_metrics(y_test_reg, pred)
        joblib.dump(model, MODELS_DIR / f"demand_{slug(name)}.joblib")
        trained_demand_models.append(name)

    # LightGBM is optional and guarded for environment compatibility.
    try:
        from lightgbm import LGBMRegressor
        lgbm = Pipeline([
            ("prep", _preprocessor()),
            ("model", LGBMRegressor(
                n_estimators=150, learning_rate=0.05, num_leaves=31,
                subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED, verbosity=-1
            )),
        ])
        lgbm.fit(x_train, y_train_reg)
        pred = np.maximum(0, lgbm.predict(x_test))
        metrics["demand"]["LightGBM"] = demand_metrics(y_test_reg, pred)
        joblib.dump(lgbm, MODELS_DIR / "demand_lightgbm.joblib")
        trained_demand_models.append("LightGBM")
    except Exception as exc:
        metrics["demand_optional_note"] = f"LightGBM skipped: {type(exc).__name__}"

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    meta = {
        "features": FEATURES,
        "stockout_models": trained_stock_models,
        "demand_models": trained_demand_models,
        "test_dates": [str(pd.to_datetime(test["Date"]).min().date()), str(pd.to_datetime(test["Date"]).max().date())],
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return metrics


def load_model(kind: str, name: str):
    if kind == "demand" and name == "Moving Average":
        return None
    return joblib.load(MODELS_DIR / f"{kind}_{slug(name)}.joblib")


def predict_stockout(df: pd.DataFrame, model_name: str) -> np.ndarray:
    return load_model("stockout", model_name).predict_proba(df[FEATURES])[:, 1]


def predict_demand7(df: pd.DataFrame, model_name: str) -> np.ndarray:
    if model_name == "Moving Average":
        return np.maximum(0, df["Avg_Daily_Sales_7D"].to_numpy() * 7.0)
    return np.maximum(0, load_model("demand", model_name).predict(df[FEATURES]))


def model_feature_importance(model, top_n: int = 12) -> pd.DataFrame:
    try:
        prep = model.named_steps["prep"]
        estimator = model.named_steps["model"]
        names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            values = np.abs(coef[0] if np.ndim(coef) > 1 else coef)
        else:
            return pd.DataFrame(columns=["Feature", "Importance"])
        out = pd.DataFrame({"Feature": names, "Importance": values})
        return out.sort_values("Importance", ascending=False).head(top_n)
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])
