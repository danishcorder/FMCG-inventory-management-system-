from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request

from config import ROOT
from core.features import risk_label, stockout_recommendation
from core.init_project import ensure_initialized
from core.modeling import predict_stockout
from core.storage import load_data

app = Flask(__name__)

DEFAULT_TOKEN = os.getenv("STOCKOUT_ALERTS_TOKEN", "demo-stockout-token")
MAX_STALE_DAYS = int(os.getenv("STOCKOUT_ALERTS_MAX_AGE_DAYS", "7"))


def _latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    latest_date = pd.Timestamp(df["Date"].max())
    return df[pd.to_datetime(df["Date"]) == latest_date].copy()


def _compute_status(df: pd.DataFrame) -> dict[str, Any]:
    latest_date = pd.Timestamp(pd.to_datetime(df["Date"]).max())
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    is_stale = (now_utc - latest_date).days > MAX_STALE_DAYS
    is_synthetic = True

    if is_synthetic:
        data_status = "SYNTHETIC"
    elif is_stale:
        data_status = "STALE"
    else:
        data_status = "LIVE"

    return {
        "data_status": data_status,
        "is_synthetic": is_synthetic,
        "is_stale": is_stale,
        "data_as_of": latest_date.date().isoformat(),
    }


def _build_alerts(df: pd.DataFrame, model_name: str) -> list[dict[str, Any]]:
    snapshot = _latest_snapshot(df)
    if snapshot.empty:
        raise ValueError("No stock data available for alert generation.")

    snapshot["Stockout_Probability"] = predict_stockout(snapshot, model_name)
    snapshot["Risk_Level"] = snapshot["Stockout_Probability"].map(risk_label)

    recomputed = snapshot.apply(lambda row: stockout_recommendation(row), axis=1)
    snapshot["Recommended_Action"] = recomputed.map(lambda item: item[0])
    snapshot["Risk_Reason"] = recomputed.map(lambda item: item[2])

    ranked = snapshot.sort_values(["Stockout_Probability", "Stock_Coverage_Days"], ascending=[False, True]).head(10)

    alerts: list[dict[str, Any]] = []
    for row in ranked.itertuples(index=False):
        alerts.append(
            {
                "SKU": row.SKU_ID,
                "product": row.Product_Name,
                "warehouse": row.Warehouse_Name,
                "probability": round(float(row.Stockout_Probability), 4),
                "risk_level": row.Risk_Level,
                "available_stock": int(row.Available_Stock),
                "stock_coverage": round(float(row.Stock_Coverage_Days), 2),
                "recommendation": row.Recommended_Action,
                "reason": row.Risk_Reason,
            }
        )

    return alerts


@app.get("/api/stockout-alerts")
def stockout_alerts():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Unauthorized"}), 401

    token = auth_header.split(" ", 1)[1].strip()
    if token != DEFAULT_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    ensure_initialized(force=False)
    df = load_data()

    status = _compute_status(df)
    model_name = "Hist Gradient Boosting"

    payload = {
        "data_status": status["data_status"],
        "is_synthetic": status["is_synthetic"],
        "is_stale": status["is_stale"],
        "data_as_of": status["data_as_of"],
        "model": model_name,
        "alerts": _build_alerts(df, model_name),
    }
    return jsonify(payload)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
