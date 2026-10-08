from __future__ import annotations
import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = [
    "Product_Category", "Brand", "Fast_Slow_Mover", "Warehouse_ID",
    "Supplier_ID", "Day_Of_Week"
]

NUMERIC_FEATURES = [
    "Unit_Price", "Shelf_Life_Days", "Available_Stock", "Reserved_Stock",
    "Safety_Stock", "Reorder_Point", "On_Order_Qty", "Incoming_Qty_7D",
    "Backorder_Qty", "Avg_Daily_Sales_7D", "Avg_Daily_Sales_28D",
    "Demand_Std_28D", "Demand_Growth_Rate", "Expected_Lead_Time_Days",
    "Actual_Avg_Lead_Time", "Supplier_Delay_Days", "Supplier_Delay_Rate",
    "Supplier_Reliability", "Next_PO_ETA_Days", "Month", "Weekend_Flag",
    "Promotion_Flag", "Discount_Pct", "Ramadan_Flag", "Eid_Flag",
    "Summer_Flag", "Local_Event_Flag", "Stock_Coverage_Days",
    "Inventory_Position", "Lead_Time_Demand", "Safety_Stock_Gap",
    "Inventory_To_Demand_Ratio", "Demand_Spike_Flag", "Stock_Consumption_Rate",
    "Demand_Lag_1", "Demand_Lag_7", "Demand_Lag_14", "Demand_Lag_28"
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def risk_label(p: float) -> str:
    if p < 0.30:
        return "LOW"
    if p < 0.60:
        return "MEDIUM"
    if p < 0.80:
        return "HIGH"
    return "CRITICAL"


def stock_status(row: pd.Series) -> str:
    available = float(row.get("Available_Stock", 0))
    safety = float(row.get("Safety_Stock", 0))
    rop = float(row.get("Reorder_Point", 0))
    coverage = float(row.get("Stock_Coverage_Days", 0))
    if available <= 0:
        return "OUT OF STOCK"
    if available < safety:
        return "CRITICAL STOCK"
    if available <= rop:
        return "BELOW REORDER POINT"
    if coverage > 60:
        return "POSSIBLE OVERSTOCK"
    return "HEALTHY"


def stockout_recommendation(row: pd.Series) -> tuple[str, int, str]:
    risk = float(row.get("Stockout_Probability", 0))
    coverage = float(row.get("Stock_Coverage_Days", 999))
    lead = max(1.0, float(row.get("Expected_Lead_Time_Days", 1)))
    on_order = float(row.get("On_Order_Qty", 0))
    eta = float(row.get("Next_PO_ETA_Days", 99))
    incoming = float(row.get("Incoming_Qty_7D", 0))
    available = float(row.get("Available_Stock", 0))
    avg = float(row.get("Avg_Daily_Sales_7D", 0))
    safety = float(row.get("Safety_Stock", 0))
    moq = max(1, int(row.get("MOQ", 1)))

    target = avg * lead + safety
    raw_qty = max(0.0, target - (available + on_order))
    qty = int(np.ceil(raw_qty / moq) * moq) if raw_qty > 0 else 0

    reasons = []
    if coverage < lead:
        reasons.append(f"stock coverage {coverage:.1f}d is below supplier lead time {lead:.0f}d")
    if float(row.get("Demand_Growth_Rate", 0)) > 0.20:
        reasons.append("recent demand is increasing")
    if float(row.get("Supplier_Delay_Rate", 0)) > 0.15:
        reasons.append("supplier delay rate is elevated")
    if incoming <= 0:
        reasons.append("no incoming stock is due within 7 days")
    if int(row.get("Demand_Spike_Flag", 0)) == 1:
        reasons.append("a demand spike is detected")

    if risk >= 0.80 and coverage < lead:
        action = "URGENT REORDER"
    elif risk >= 0.60 and on_order > 0 and eta > coverage:
        action = "EXPEDITE EXISTING PO"
    elif risk >= 0.60:
        action = "REORDER"
    elif int(row.get("Demand_Spike_Flag", 0)) == 1:
        action = "INVESTIGATE DEMAND SPIKE"
    elif coverage > 60:
        action = "MONITOR OVERSTOCK"
    else:
        action = "MONITOR" if risk >= 0.30 else "NO ACTION"

    if not reasons:
        reasons.append("current stock position is within the normal operating range")
    return action, qty, "; ".join(reasons)


def replenishment_values(row: pd.Series, forecast_units: float, horizon_days: int = 7, z: float = 1.65) -> dict:
    lead = max(1.0, float(row.get("Expected_Lead_Time_Days", 1)))
    avg_daily_forecast = float(forecast_units) / max(horizon_days, 1)
    variability = max(0.0, float(row.get("Demand_Std_28D", 0)))
    safety_stock = z * variability * np.sqrt(lead)
    lead_time_demand = avg_daily_forecast * lead
    reorder_point = lead_time_demand + safety_stock
    inventory_position = (
        float(row.get("Available_Stock", 0))
        + float(row.get("Incoming_Qty_7D", 0))
        - float(row.get("Reserved_Stock", 0))
    )
    raw_qty = max(0.0, float(forecast_units) + safety_stock - inventory_position)
    moq = max(1.0, float(row.get("MOQ", 1)))
    recommended = np.ceil(raw_qty / moq) * moq if raw_qty > 0 else 0.0
    return {
        "Lead_Time_Demand": lead_time_demand,
        "Dynamic_Safety_Stock": safety_stock,
        "Dynamic_Reorder_Point": reorder_point,
        "Inventory_Position_for_Plan": inventory_position,
        "Recommended_Order_Qty": recommended,
        "Order_Now": inventory_position <= reorder_point,
    }
