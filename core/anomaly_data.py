from __future__ import annotations

import numpy as np
import pandas as pd

from config import RANDOM_SEED

ANOMALY_TYPES = [
    "Stock Loss / Shrinkage",
    "Stock Mismatch",
    "Negative Inventory",
    "Abnormal Purchase Size",
    "Unusual Sales Day",
    "Suspicious Return",
    "Duplicate Transaction",
    "Late Adjustment",
    "Slow Moving / Dead Stock",
    "Expiry Risk",
    "Transfer Mismatch",
    "Warehouse Capacity Pressure",
    "Supplier Delay",
    "Short Delivery",
]

ANOMALY_CATEGORIES = {
    "Stock Loss / Shrinkage": "Quantity",
    "Stock Mismatch": "Quantity",
    "Negative Inventory": "Quantity",
    "Abnormal Purchase Size": "Quantity",
    "Unusual Sales Day": "Time-based",
    "Suspicious Return": "Transaction",
    "Duplicate Transaction": "Transaction",
    "Late Adjustment": "Time-based",
    "Slow Moving / Dead Stock": "Product",
    "Expiry Risk": "Product",
    "Transfer Mismatch": "Warehouse",
    "Warehouse Capacity Pressure": "Warehouse",
    "Supplier Delay": "Supplier",
    "Short Delivery": "Supplier",
}

ACTION_MAP = {
    "Stock Loss / Shrinkage": "Inspect warehouse movement, physical count and authorization trail.",
    "Stock Mismatch": "Reconcile opening, receipts, sales, transfers and adjustments.",
    "Negative Inventory": "Block posting and correct inventory master/transaction sequence.",
    "Abnormal Purchase Size": "Validate purchase order quantity, MOQ and approval.",
    "Unusual Sales Day": "Confirm promotion/event demand and inspect possible posting error.",
    "Suspicious Return": "Audit return documents, customer/source and batch condition.",
    "Duplicate Transaction": "Check duplicate document reference and reverse duplicate posting if confirmed.",
    "Late Adjustment": "Review user, timestamp and approval reason for the adjustment.",
    "Slow Moving / Dead Stock": "Reduce replenishment and review transfer, discount or liquidation options.",
    "Expiry Risk": "Prioritize FEFO dispatch, transfer, markdown or block risky batch.",
    "Transfer Mismatch": "Reconcile dispatch vs receipt between warehouses.",
    "Warehouse Capacity Pressure": "Review capacity utilization and rebalance stock.",
    "Supplier Delay": "Contact supplier, expedite delivery or revise lead-time assumptions.",
    "Short Delivery": "Verify GRN vs PO and raise supplier shortage claim.",
}

IMPACT_MAP = {
    "Stock Loss / Shrinkage": "Potential shrinkage, financial loss and service risk.",
    "Stock Mismatch": "Incorrect inventory valuation and unreliable replenishment decisions.",
    "Negative Inventory": "Data integrity failure and invalid availability signals.",
    "Abnormal Purchase Size": "Cash blockage, storage pressure or accidental over-order.",
    "Unusual Sales Day": "Possible lost demand signal, fraud/error or unplanned spike.",
    "Suspicious Return": "Margin erosion, stock distortion and possible abuse.",
    "Duplicate Transaction": "Double-counted movement and financial/inventory mismatch.",
    "Late Adjustment": "Control weakness and unexplained stock movement.",
    "Slow Moving / Dead Stock": "Working capital blocked with expiry/obsolescence risk.",
    "Expiry Risk": "Potential write-off, discounting cost and service disruption.",
    "Transfer Mismatch": "Inventory missing between source and destination locations.",
    "Warehouse Capacity Pressure": "Handling inefficiency, congestion and damage risk.",
    "Supplier Delay": "Higher stockout exposure and unstable replenishment.",
    "Short Delivery": "Lower-than-planned availability and supplier performance risk.",
}


def _stable_random_frame(n: int, seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def enrich_and_inject_anomalies(base: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a transaction-style anomaly dataset from the shared FMCG foundation.

    The original Stockout and Demand modules are not modified. This creates a separate
    enriched copy with batch/expiry/returns/adjustments/transfer/supplier fields and
    controlled synthetic anomalies for evaluation.
    """
    df = base.copy().sort_values(["SKU_ID", "Warehouse_ID", "Date"]).reset_index(drop=True)
    rng = _stable_random_frame(len(df), seed)
    n = len(df)

    df["Transaction_ID"] = [f"TXN-{i+1:08d}" for i in range(n)]
    df["Product_ID"] = df["SKU_ID"]
    df["Unit_Cost"] = (df["Unit_Price"] * rng.uniform(0.62, 0.82, n)).round(2)
    df["Selling_Price"] = df["Unit_Price"]
    df["Purchases"] = df["Received_Quantity"].astype(float)
    df["Daily_Sales"] = df["Daily_Sales"].astype(float)

    # Baseline transaction fields
    base_returns = rng.binomial(np.maximum(df["Daily_Sales"].astype(int), 0), 0.012)
    df["Returns"] = base_returns.astype(float)
    df["Stock_Adjustment"] = np.where(rng.random(n) < 0.018, rng.integers(-3, 4, n), 0).astype(float)
    df["Transfers_In"] = np.where(rng.random(n) < 0.012, rng.integers(5, 45, n), 0).astype(float)
    df["Transfers_Out"] = np.where(rng.random(n) < 0.012, rng.integers(5, 45, n), 0).astype(float)
    df["Damaged_Qty"] = np.where(rng.random(n) < 0.010, rng.integers(1, 8, n), 0).astype(float)

    wh_capacity = {
        "WH-MUL-C": 180000,
        "WH-MUL-N": 140000,
        "WH-MUL-S": 125000,
    }
    df["Warehouse_Capacity"] = df["Warehouse_ID"].map(wh_capacity).fillna(140000).astype(float)

    # Batch and expiry context. One synthetic batch per SKU-location-month.
    date = pd.to_datetime(df["Date"])
    df["Batch_ID"] = (
        "B-" + df["SKU_ID"].astype(str) + "-" + df["Warehouse_ID"].astype(str)
        + "-" + date.dt.strftime("%Y%m")
    )
    max_age = np.maximum(7, (df["Shelf_Life_Days"].to_numpy() * 0.85).astype(int))
    stock_age = np.array([rng.integers(1, max(2, m + 1)) for m in max_age], dtype=int)
    # Faster movers are generally younger stock.
    stock_age = np.where(df["Fast_Slow_Mover"].eq("Fast"), (stock_age * 0.55).astype(int), stock_age)
    stock_age = np.maximum(stock_age, 1)
    df["Stock_Age_Days"] = stock_age
    df["Expiry_Date"] = date + pd.to_timedelta(df["Shelf_Life_Days"] - df["Stock_Age_Days"], unit="D")
    df["Days_To_Expiry"] = (df["Expiry_Date"] - date).dt.days.astype(float)
    df["Expired_Qty"] = np.where(df["Days_To_Expiry"] < 0, np.minimum(df["Available_Stock"], 10), 0).astype(float)

    # Delivery context
    short_ratio = np.where(
        df["Received_Quantity"] > 0,
        np.clip(rng.beta(1.2, 30, n), 0, 0.18),
        0.0,
    )
    df["Short_Delivery_Ratio"] = short_ratio
    df["Supplier_Lead_Time_Deviation"] = (
        df["Actual_Avg_Lead_Time"] - df["Expected_Lead_Time_Days"]
    ).astype(float)

    # Expected stock equation before injected anomalies.
    df["Expected_Closing_Stock"] = (
        df["Opening_Stock"]
        + df["Purchases"]
        + df["Returns"]
        + df["Transfers_In"]
        - df["Daily_Sales"]
        - df["Transfers_Out"]
        - df["Damaged_Qty"]
        + df["Stock_Adjustment"]
    ).astype(float)
    df["Observed_Closing_Stock"] = df["Expected_Closing_Stock"].clip(lower=0)
    df["Duplicate_Transaction_Flag"] = 0
    df["Late_Adjustment_Flag"] = 0
    df["Transfer_Mismatch_Qty"] = 0.0
    df["Synthetic_Anomaly_Label"] = 0
    df["Injected_Anomaly_Type"] = "Normal"

    # Inject controlled examples. Keep total anomaly rate moderate.
    eligible = np.arange(n)
    rng.shuffle(eligible)
    cursor = 0

    def take(k: int):
        nonlocal cursor
        idx = eligible[cursor: cursor + k]
        cursor += k
        return idx

    counts = {
        "Stock Loss / Shrinkage": max(120, int(n * 0.010)),
        "Stock Mismatch": max(90, int(n * 0.008)),
        "Negative Inventory": max(45, int(n * 0.003)),
        "Abnormal Purchase Size": max(70, int(n * 0.005)),
        "Unusual Sales Day": max(100, int(n * 0.008)),
        "Suspicious Return": max(65, int(n * 0.005)),
        "Duplicate Transaction": max(55, int(n * 0.004)),
        "Late Adjustment": max(55, int(n * 0.004)),
        "Expiry Risk": max(80, int(n * 0.006)),
        "Transfer Mismatch": max(60, int(n * 0.004)),
        "Supplier Delay": max(90, int(n * 0.007)),
        "Short Delivery": max(60, int(n * 0.004)),
    }

    for anomaly_type, k in counts.items():
        idx = take(min(k, max(0, n - cursor)))
        if len(idx) == 0:
            continue
        df.loc[idx, "Synthetic_Anomaly_Label"] = 1
        df.loc[idx, "Injected_Anomaly_Type"] = anomaly_type

        if anomaly_type == "Stock Loss / Shrinkage":
            loss = np.maximum(10, (df.loc[idx, "Opening_Stock"].to_numpy() * rng.uniform(0.18, 0.55, len(idx))).astype(int))
            df.loc[idx, "Stock_Adjustment"] -= loss
            df.loc[idx, "Observed_Closing_Stock"] = (
                df.loc[idx, "Expected_Closing_Stock"].to_numpy() - loss
            )
        elif anomaly_type == "Stock Mismatch":
            mismatch = rng.choice([-1, 1], len(idx)) * rng.integers(15, 120, len(idx))
            df.loc[idx, "Observed_Closing_Stock"] = (
                df.loc[idx, "Expected_Closing_Stock"].to_numpy() + mismatch
            )
        elif anomaly_type == "Negative Inventory":
            df.loc[idx, "Observed_Closing_Stock"] = -rng.integers(1, 60, len(idx))
        elif anomaly_type == "Abnormal Purchase Size":
            df.loc[idx, "Purchases"] = (
                np.maximum(df.loc[idx, "Avg_Daily_Sales_28D"].to_numpy() * rng.uniform(35, 80, len(idx)), 500)
            )
        elif anomaly_type == "Unusual Sales Day":
            df.loc[idx, "Daily_Sales"] = (
                np.maximum(df.loc[idx, "Avg_Daily_Sales_28D"].to_numpy() * rng.uniform(3.5, 6.0, len(idx)), 25)
            )
        elif anomaly_type == "Suspicious Return":
            df.loc[idx, "Returns"] = np.maximum(
                10, df.loc[idx, "Daily_Sales"].to_numpy() * rng.uniform(0.35, 0.95, len(idx))
            )
        elif anomaly_type == "Duplicate Transaction":
            df.loc[idx, "Duplicate_Transaction_Flag"] = 1
        elif anomaly_type == "Late Adjustment":
            df.loc[idx, "Late_Adjustment_Flag"] = 1
            df.loc[idx, "Stock_Adjustment"] = rng.choice([-1, 1], len(idx)) * rng.integers(10, 90, len(idx))
        elif anomaly_type == "Expiry Risk":
            df.loc[idx, "Stock_Age_Days"] = np.maximum(
                1, df.loc[idx, "Shelf_Life_Days"].to_numpy() - rng.integers(1, 21, len(idx))
            )
            df.loc[idx, "Days_To_Expiry"] = rng.integers(1, 21, len(idx))
            df.loc[idx, "Expiry_Date"] = pd.to_datetime(df.loc[idx, "Date"]) + pd.to_timedelta(
                df.loc[idx, "Days_To_Expiry"], unit="D"
            )
        elif anomaly_type == "Transfer Mismatch":
            df.loc[idx, "Transfer_Mismatch_Qty"] = rng.integers(10, 120, len(idx))
        elif anomaly_type == "Supplier Delay":
            df.loc[idx, "Supplier_Lead_Time_Deviation"] = rng.integers(5, 14, len(idx))
            df.loc[idx, "Supplier_Delay_Rate"] = np.maximum(
                df.loc[idx, "Supplier_Delay_Rate"].to_numpy(), rng.uniform(0.35, 0.75, len(idx))
            )
        elif anomaly_type == "Short Delivery":
            df.loc[idx, "Short_Delivery_Ratio"] = rng.uniform(0.25, 0.70, len(idx))

    # Recompute expected/observed mismatch after anomaly injection.
    df["Stock_Mismatch_Qty"] = df["Observed_Closing_Stock"] - df["Expected_Closing_Stock"]
    df["Inventory_Value"] = df["Observed_Closing_Stock"].clip(lower=0) * df["Unit_Cost"]

    # Behavioral features
    keys = ["SKU_ID", "Warehouse_ID"]
    g = df.groupby(keys, sort=False)

    df["Daily_Stock_Change"] = g["Observed_Closing_Stock"].diff().fillna(0)
    prev_stock = g["Observed_Closing_Stock"].shift(1).abs().replace(0, np.nan)
    df["Pct_Stock_Change"] = (df["Daily_Stock_Change"] / prev_stock).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["Stock_Movement_Velocity"] = (
        df["Purchases"].abs()
        + df["Daily_Sales"].abs()
        + df["Returns"].abs()
        + df["Transfers_In"].abs()
        + df["Transfers_Out"].abs()
        + df["Stock_Adjustment"].abs()
    )

    df["Rolling_Sales_Mean_7"] = g["Daily_Sales"].transform(lambda s: s.rolling(7, min_periods=2).mean()).fillna(df["Daily_Sales"])
    df["Rolling_Sales_Std_28"] = g["Daily_Sales"].transform(lambda s: s.rolling(28, min_periods=3).std(ddof=0)).fillna(0)
    df["Demand_Variation"] = df["Rolling_Sales_Std_28"] / df["Rolling_Sales_Mean_7"].replace(0, np.nan)
    df["Demand_Variation"] = df["Demand_Variation"].replace([np.inf, -np.inf], np.nan).fillna(0)

    df["Stock_To_Sales_Ratio"] = df["Observed_Closing_Stock"].clip(lower=0) / df["Rolling_Sales_Mean_7"].replace(0, np.nan)
    df["Stock_To_Sales_Ratio"] = df["Stock_To_Sales_Ratio"].replace([np.inf, -np.inf], np.nan).fillna(999)
    df["Adjustment_Ratio"] = df["Stock_Adjustment"].abs() / np.maximum(df["Opening_Stock"].abs(), 1)
    df["Return_Percentage"] = df["Returns"] / np.maximum(df["Daily_Sales"].abs(), 1)
    df["Damage_Percentage"] = df["Damaged_Qty"] / np.maximum(df["Opening_Stock"].abs(), 1)
    df["Inventory_Turnover_Proxy"] = df["Daily_Sales"] / np.maximum(df["Observed_Closing_Stock"].clip(lower=0), 1)
    df["Product_Movement_Frequency"] = g["Daily_Sales"].transform(lambda s: s.rolling(28, min_periods=1).apply(lambda x: float((x > 0).mean()), raw=True))

    # Product/warehouse/category deviation features.
    df["SKU_Sales_Deviation"] = (
        df["Daily_Sales"] - g["Daily_Sales"].transform(lambda s: s.rolling(28, min_periods=3).mean())
    ) / g["Daily_Sales"].transform(lambda s: s.rolling(28, min_periods=3).std(ddof=0)).replace(0, np.nan)
    df["SKU_Sales_Deviation"] = df["SKU_Sales_Deviation"].replace([np.inf, -np.inf], np.nan).fillna(0)

    wh_date_mean = df.groupby(["Warehouse_ID", "Date"])["Daily_Stock_Change"].transform("mean")
    wh_date_std = df.groupby(["Warehouse_ID", "Date"])["Daily_Stock_Change"].transform("std").replace(0, np.nan)
    df["Warehouse_Level_Deviation"] = ((df["Daily_Stock_Change"] - wh_date_mean) / wh_date_std).replace([np.inf, -np.inf], np.nan).fillna(0)

    cat_date_mean = df.groupby(["Product_Category", "Date"])["Daily_Sales"].transform("mean")
    cat_date_std = df.groupby(["Product_Category", "Date"])["Daily_Sales"].transform("std").replace(0, np.nan)
    df["Category_Level_Deviation"] = ((df["Daily_Sales"] - cat_date_mean) / cat_date_std).replace([np.inf, -np.inf], np.nan).fillna(0)

    wh_total = df.groupby(["Warehouse_ID", "Date"])["Observed_Closing_Stock"].transform(lambda s: s.clip(lower=0).sum())
    df["Warehouse_Capacity_Utilization"] = wh_total / df["Warehouse_Capacity"].replace(0, np.nan)
    df["Warehouse_Capacity_Utilization"] = df["Warehouse_Capacity_Utilization"].fillna(0)

    df["Days_of_Inventory"] = df["Observed_Closing_Stock"].clip(lower=0) / df["Rolling_Sales_Mean_7"].replace(0, np.nan)
    df["Days_of_Inventory"] = df["Days_of_Inventory"].replace([np.inf, -np.inf], np.nan).fillna(999)

    # Rule score is transparent hybrid support.
    rule = np.zeros(n, dtype=float)
    rule += np.where(np.abs(df["Pct_Stock_Change"]) > 0.45, 18, 0)
    rule += np.where(np.abs(df["Stock_Mismatch_Qty"]) > np.maximum(20, df["Opening_Stock"] * 0.12), 22, 0)
    rule += np.where(df["Observed_Closing_Stock"] < 0, 35, 0)
    rule += np.where(df["Return_Percentage"] > 0.30, 18, 0)
    rule += np.where(df["Adjustment_Ratio"] > 0.15, 18, 0)
    rule += np.where(df["Duplicate_Transaction_Flag"] == 1, 25, 0)
    rule += np.where(df["Late_Adjustment_Flag"] == 1, 15, 0)
    rule += np.where(df["Days_To_Expiry"] <= 30, 18, 0)
    rule += np.where(df["Days_of_Inventory"] > 75, 16, 0)
    rule += np.where(df["Supplier_Lead_Time_Deviation"] >= 5, 16, 0)
    rule += np.where(df["Short_Delivery_Ratio"] >= 0.25, 18, 0)
    rule += np.where(df["Transfer_Mismatch_Qty"] >= 10, 20, 0)
    rule += np.where(df["Warehouse_Capacity_Utilization"] > 0.90, 14, 0)
    rule += np.where(np.abs(df["SKU_Sales_Deviation"]) >= 3.0, 14, 0)
    df["Business_Rule_Score"] = np.clip(rule, 0, 100)

    df["Stock_Status_Anomaly"] = np.select(
        [
            df["Days_To_Expiry"] <= 30,
            df["Observed_Closing_Stock"] <= df["Safety_Stock"],
            df["Days_of_Inventory"] > 60,
        ],
        ["EXPIRY RISK", "LOW", "OVERSTOCK"],
        default="NORMAL",
    )

    return df


def explain_anomaly_row(row: pd.Series) -> tuple[str, str, str, str]:
    """Return (type, category, reason, recommended action) from strongest visible signal."""
    candidates = []

    def add(priority, anomaly_type, reason):
        candidates.append((priority, anomaly_type, reason))

    if float(row.get("Observed_Closing_Stock", 0)) < 0:
        add(100, "Negative Inventory", "Observed closing stock is negative.")
    if abs(float(row.get("Stock_Mismatch_Qty", 0))) > max(20, abs(float(row.get("Opening_Stock", 0))) * 0.12):
        add(92, "Stock Mismatch", f"Closing stock differs from transaction-based expected stock by {row.get('Stock_Mismatch_Qty', 0):.0f} units.")
    if float(row.get("Pct_Stock_Change", 0)) < -0.45 and float(row.get("Adjustment_Ratio", 0)) > 0.10:
        add(90, "Stock Loss / Shrinkage", f"Stock dropped {abs(row.get('Pct_Stock_Change', 0)):.0%} with a large unexplained adjustment.")
    if int(row.get("Duplicate_Transaction_Flag", 0)) == 1:
        add(88, "Duplicate Transaction", "A duplicate transaction pattern is flagged for this record.")
    if float(row.get("Return_Percentage", 0)) > 0.30:
        add(84, "Suspicious Return", f"Returns equal {row.get('Return_Percentage', 0):.0%} of recorded sales.")
    if int(row.get("Late_Adjustment_Flag", 0)) == 1:
        add(82, "Late Adjustment", "An unusual inventory adjustment occurred outside normal transaction behavior.")
    if float(row.get("Transfer_Mismatch_Qty", 0)) >= 10:
        add(80, "Transfer Mismatch", f"Transfer mismatch of {row.get('Transfer_Mismatch_Qty', 0):.0f} units is present.")
    if float(row.get("Short_Delivery_Ratio", 0)) >= 0.25:
        add(78, "Short Delivery", f"Estimated short-delivery ratio is {row.get('Short_Delivery_Ratio', 0):.0%}.")
    if float(row.get("Supplier_Lead_Time_Deviation", 0)) >= 5:
        add(76, "Supplier Delay", f"Supplier lead time exceeds expectation by {row.get('Supplier_Lead_Time_Deviation', 0):.0f} days.")
    if float(row.get("Days_To_Expiry", 999)) <= 30 and float(row.get("Observed_Closing_Stock", 0)) > 0:
        add(74, "Expiry Risk", f"Stock has only {row.get('Days_To_Expiry', 0):.0f} days to expiry.")
    if float(row.get("Days_of_Inventory", 0)) > 75:
        add(70, "Slow Moving / Dead Stock", f"Inventory coverage is {row.get('Days_of_Inventory', 0):.0f} days.")
    if float(row.get("Warehouse_Capacity_Utilization", 0)) > 0.90:
        add(68, "Warehouse Capacity Pressure", f"Warehouse capacity utilization is {row.get('Warehouse_Capacity_Utilization', 0):.0%}.")
    if abs(float(row.get("SKU_Sales_Deviation", 0))) >= 3.0:
        add(66, "Unusual Sales Day", f"Sales are {abs(row.get('SKU_Sales_Deviation', 0)):.1f} standard deviations from recent SKU behavior.")
    if float(row.get("Purchases", 0)) > max(500, float(row.get("Avg_Daily_Sales_28D", 0)) * 35):
        add(64, "Abnormal Purchase Size", "Purchase quantity is unusually large relative to recent demand.")

    if not candidates:
        anomaly_type = str(row.get("Injected_Anomaly_Type", "Normal"))
        if anomaly_type != "Normal":
            reason = f"Synthetic test scenario indicates {anomaly_type.lower()} behavior."
        else:
            anomaly_type = "Normal"
            reason = "No strong business-rule anomaly signal is present."
    else:
        _, anomaly_type, reason = sorted(candidates, reverse=True)[0]

    category = ANOMALY_CATEGORIES.get(anomaly_type, "Normal")
    action = ACTION_MAP.get(anomaly_type, "Continue routine monitoring.")
    return anomaly_type, category, reason, action


def business_impact(anomaly_type: str) -> str:
    return IMPACT_MAP.get(anomaly_type, "No material business impact identified.")
