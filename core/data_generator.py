from __future__ import annotations
import math
import sqlite3
import numpy as np
import pandas as pd
from config import CSV_PATH, DB_PATH, RANDOM_SEED

CATEGORIES = [
    "Beverages", "Dairy", "Snacks", "Biscuits", "Cooking Oil", "Detergents",
    "Personal Care", "Tea", "Juices", "Bottled Water", "Household Cleaning", "Packaged Foods"
]
CATEGORY_BASE_DEMAND = {
    "Beverages": 52, "Dairy": 44, "Snacks": 38, "Biscuits": 35,
    "Cooking Oil": 28, "Detergents": 23, "Personal Care": 20, "Tea": 26,
    "Juices": 34, "Bottled Water": 48, "Household Cleaning": 18, "Packaged Foods": 31,
}
WAREHOUSES = [
    ("WH-MUL-C", "Multan Central Warehouse", 1.15),
    ("WH-MUL-N", "Multan North Warehouse", 0.95),
    ("WH-MUL-S", "Multan South Warehouse", 0.88),
]
BRANDS = ["Nova", "Prime", "FreshCo", "DailyChoice", "MetroMart", "PureLife", "Urban", "Family"]
PACKS = ["50g", "100g", "250g", "500g", "500ml", "1L", "2L", "5kg"]


def _ramadan_flag(d: pd.Timestamp) -> int:
    return int(pd.Timestamp("2025-03-01") <= d <= pd.Timestamp("2025-03-30"))


def _eid_flag(d: pd.Timestamp) -> int:
    return int(pd.Timestamp("2025-03-30") <= d <= pd.Timestamp("2025-04-04"))


def _summer_flag(d: pd.Timestamp) -> int:
    return int(d.month in (5, 6, 7, 8))


def _season_multiplier(category: str, d: pd.Timestamp) -> float:
    mult = 1.0
    if category in ("Beverages", "Juices", "Bottled Water") and _summer_flag(d):
        mult *= 1.35
    if category in ("Tea", "Biscuits", "Packaged Foods") and _ramadan_flag(d):
        mult *= 1.22
    if category in ("Cooking Oil", "Beverages", "Snacks", "Biscuits") and _eid_flag(d):
        mult *= 1.28
    return mult


def generate_synthetic_data(n_skus: int = 100, n_days: int = 365,
                            start_date: str = "2025-01-01", seed: int = RANDOM_SEED,
                            save: bool = True) -> pd.DataFrame:
    """Generate process-based FMCG inventory data.

    Demand, promotions, supplier delays, purchase orders, receipts and stock depletion
    are simulated together so the target is not a random label.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    suppliers = []
    for i in range(18):
        suppliers.append({
            "Supplier_ID": f"SUP-{i+1:03d}",
            "Supplier_Name": f"Supplier {i+1:02d}",
            "Expected_Lead_Time_Days": int(rng.integers(3, 10)),
            "Supplier_Reliability": float(rng.uniform(0.72, 0.96)),
            "MOQ": int(rng.choice([50, 100, 150, 200, 250, 500])),
        })
    suppliers_df = pd.DataFrame(suppliers)

    products = []
    for i in range(n_skus):
        category = CATEGORIES[i % len(CATEGORIES)]
        pack = str(rng.choice(PACKS))
        brand = str(rng.choice(BRANDS))
        base = CATEGORY_BASE_DEMAND[category] * float(rng.uniform(0.65, 1.45))
        supplier = suppliers[int(rng.integers(0, len(suppliers)))]
        products.append({
            "SKU_ID": f"{category[:3].upper()}-{pack.upper()}-{i+1:03d}",
            "Product_Name": f"{brand} {category} {pack}",
            "Product_Category": category,
            "Brand": brand,
            "Pack_Size": pack,
            "Unit_Price": round(float(rng.uniform(45, 650)), 2),
            "Shelf_Life_Days": int(rng.choice([60, 90, 120, 180, 270, 365, 540])),
            "Fast_Slow_Mover": "Fast" if base >= 34 else ("Medium" if base >= 24 else "Slow"),
            "Base_Demand": float(base),
            "Supplier_ID": supplier["Supplier_ID"],
        })
    products_df = pd.DataFrame(products).merge(suppliers_df, on="Supplier_ID", how="left")

    rows = []
    for _, prod in products_df.iterrows():
        for wh_id, wh_name, wh_mult in WAREHOUSES:
            stock = max(220, int(prod["Base_Demand"] * rng.uniform(8, 14)))
            open_orders: list[tuple[pd.Timestamp, int]] = []
            delayed_orders = 0
            total_orders = 0
            actual_lead_samples: list[int] = []
            safety_stock = max(45, int(prod["Base_Demand"] * rng.uniform(1.6, 3.0)))
            reorder_point = int(prod["Base_Demand"] * prod["Expected_Lead_Time_Days"] + safety_stock)
            order_cycle_days = int(rng.integers(6, 11))
            demand_hist, sales_hist = [], []

            for day_idx, d in enumerate(dates):
                opening_stock = stock
                received = 0
                pending = []
                for arrival, qty in open_orders:
                    if arrival <= d:
                        received += qty
                    else:
                        pending.append((arrival, qty))
                open_orders = pending
                stock += received

                dow_mult = 1.12 if d.dayofweek in (4, 5) else (0.92 if d.dayofweek == 0 else 1.0)
                season_mult = _season_multiplier(prod["Product_Category"], d)
                slope = float(rng.uniform(-0.04, 0.10))
                trend_mult = 1.0 + slope * day_idx / max(1, n_days - 1)
                promo_flag = int(rng.random() < 0.04)
                discount_pct = float(rng.choice([5, 10, 15, 20])) if promo_flag else 0.0
                promo_mult = 1.0 + (0.18 + discount_pct / 100.0) if promo_flag else 1.0
                local_event = int(rng.random() < 0.012)
                event_mult = 1.18 if local_event else 1.0

                demand_mean = prod["Base_Demand"] * wh_mult * dow_mult * season_mult * trend_mult * promo_mult * event_mult
                latent_demand = max(0, int(round(rng.normal(demand_mean, max(2.0, demand_mean * 0.16)))))
                reserved = int(round(latent_demand * rng.uniform(0.0, 0.08)))
                available_before_sales = max(stock - reserved, 0)
                sales = min(available_before_sales, latent_demand)
                lost_sales = max(latent_demand - available_before_sales, 0)
                stock = max(stock - sales, 0)

                # Imperfect replenishment policy creates realistic risk and occasional stockouts.
                on_order_before = sum(qty for _, qty in open_orders)
                inventory_position = stock + on_order_before
                due_for_review = day_idx % order_cycle_days == 0
                if (due_for_review or inventory_position <= reorder_point) and inventory_position < reorder_point * 1.12:
                    total_orders += 1
                    target_level = int(prod["Base_Demand"] * (prod["Expected_Lead_Time_Days"] + 7) + safety_stock)
                    raw_qty = max(int(prod["MOQ"]), target_level - inventory_position)
                    moq = int(prod["MOQ"])
                    order_qty = int(math.ceil(raw_qty / moq) * moq)
                    expected_lt = int(prod["Expected_Lead_Time_Days"])
                    late = rng.random() > float(prod["Supplier_Reliability"])
                    delay = int(rng.integers(1, 6)) if late else 0
                    if late:
                        delayed_orders += 1
                    actual_lt = expected_lt + delay
                    actual_lead_samples.append(actual_lt)
                    open_orders.append((d + pd.Timedelta(days=actual_lt), order_qty))

                on_order_qty = sum(qty for _, qty in open_orders)
                incoming_7d = sum(qty for arrival, qty in open_orders if 0 <= (arrival - d).days <= 7)
                next_eta = min([(arrival - d).days for arrival, _ in open_orders], default=99)
                delay_rate = delayed_orders / total_orders if total_orders else max(0.0, 1.0 - float(prod["Supplier_Reliability"]))
                actual_avg_lead = float(np.mean(actual_lead_samples)) if actual_lead_samples else float(prod["Expected_Lead_Time_Days"])

                demand_hist.append(latent_demand)
                sales_hist.append(sales)
                demand_series = pd.Series(demand_hist, dtype=float)
                sales_series = pd.Series(sales_hist, dtype=float)
                avg7 = float(demand_series.tail(7).mean())
                avg28 = float(demand_series.tail(28).mean())
                std28 = float(demand_series.tail(28).std(ddof=0)) if len(demand_series.tail(28)) > 1 else 0.0
                sales7 = float(sales_series.tail(7).sum())
                sales28 = float(sales_series.tail(28).sum())

                available = max(stock - reserved, 0)
                coverage = available / max(avg7, 0.1)
                lead_time_demand = avg7 * float(prod["Expected_Lead_Time_Days"])
                inv_position = available + on_order_qty
                inv_to_demand = inv_position / max(lead_time_demand, 1.0)
                demand_growth = (avg7 - avg28) / max(avg28, 1.0)
                demand_spike = int(avg7 > 1.30 * max(avg28, 1.0))
                stock_consumption = sales / max(opening_stock + received, 1)

                rows.append({
                    "Date": d,
                    "SKU_ID": prod["SKU_ID"], "Product_Name": prod["Product_Name"],
                    "Product_Category": prod["Product_Category"], "Brand": prod["Brand"],
                    "Pack_Size": prod["Pack_Size"], "Unit_Price": float(prod["Unit_Price"]),
                    "Shelf_Life_Days": int(prod["Shelf_Life_Days"]), "Fast_Slow_Mover": prod["Fast_Slow_Mover"],
                    "Warehouse_ID": wh_id, "Warehouse_Name": wh_name,
                    "Supplier_ID": prod["Supplier_ID"], "Supplier_Name": prod["Supplier_Name"],
                    "MOQ": int(prod["MOQ"]),
                    "Opening_Stock": int(opening_stock), "Received_Quantity": int(received),
                    "Demand_Qty": int(latent_demand), "Daily_Sales": int(sales), "Lost_Sales": int(lost_sales),
                    "Closing_Stock": int(stock), "Available_Stock": int(available), "Reserved_Stock": int(reserved),
                    "Safety_Stock": int(safety_stock), "Reorder_Point": int(reorder_point),
                    "On_Order_Qty": int(on_order_qty), "Incoming_Qty_7D": int(incoming_7d),
                    "Backorder_Qty": int(lost_sales), "Sales_Last_7D": sales7, "Sales_Last_28D": sales28,
                    "Avg_Daily_Sales_7D": avg7, "Avg_Daily_Sales_28D": avg28,
                    "Demand_Std_28D": std28, "Demand_Growth_Rate": demand_growth,
                    "Expected_Lead_Time_Days": int(prod["Expected_Lead_Time_Days"]),
                    "Actual_Avg_Lead_Time": actual_avg_lead,
                    "Supplier_Delay_Days": max(0.0, actual_avg_lead - float(prod["Expected_Lead_Time_Days"])),
                    "Supplier_Delay_Rate": float(delay_rate), "Supplier_Reliability": float(prod["Supplier_Reliability"]),
                    "Next_PO_ETA_Days": int(next_eta), "Day_Of_Week": d.day_name(), "Month": int(d.month),
                    "Weekend_Flag": int(d.dayofweek >= 5), "Promotion_Flag": promo_flag,
                    "Discount_Pct": discount_pct, "Ramadan_Flag": _ramadan_flag(d), "Eid_Flag": _eid_flag(d),
                    "Summer_Flag": _summer_flag(d), "Local_Event_Flag": local_event,
                    "Stock_Coverage_Days": coverage, "Inventory_Position": inv_position,
                    "Lead_Time_Demand": lead_time_demand, "Safety_Stock_Gap": available - safety_stock,
                    "Inventory_To_Demand_Ratio": inv_to_demand, "Demand_Spike_Flag": demand_spike,
                    "Stock_Consumption_Rate": stock_consumption,
                })

    df = pd.DataFrame(rows).sort_values(["SKU_ID", "Warehouse_ID", "Date"]).reset_index(drop=True)
    keys = ["SKU_ID", "Warehouse_ID"]
    grouped = df.groupby(keys, sort=False)

    for lag in [1, 7, 14, 28]:
        df[f"Demand_Lag_{lag}"] = grouped["Demand_Qty"].shift(lag)

    future_demand = pd.Series(0.0, index=df.index)
    future_stockout = pd.Series(False, index=df.index)
    for k in range(1, 8):
        future_demand += grouped["Demand_Qty"].shift(-k).fillna(0)
        future_stockout |= grouped["Lost_Sales"].shift(-k).fillna(0).gt(0)
    df["Demand_Next_7D"] = future_demand
    df["Stockout_Next_7D"] = future_stockout.astype(int)

    # Keep only rows with complete history and future horizon.
    reverse_count = df.groupby(keys).cumcount(ascending=False)
    df = df[reverse_count >= 7].copy()
    df = df.dropna(subset=["Demand_Lag_1", "Demand_Lag_7", "Demand_Lag_14", "Demand_Lag_28"]).reset_index(drop=True)

    if save:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        build_sqlite(df, products_df, suppliers_df)
    return df


def build_sqlite(df: pd.DataFrame, products_df: pd.DataFrame, suppliers_df: pd.DataFrame) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        products_df.drop(columns=["Base_Demand"], errors="ignore").to_sql("products", con, if_exists="replace", index=False)
        suppliers_df.to_sql("suppliers", con, if_exists="replace", index=False)
        pd.DataFrame([(a, b) for a, b, _ in WAREHOUSES], columns=["Warehouse_ID", "Warehouse_Name"]).to_sql(
            "warehouses", con, if_exists="replace", index=False
        )
        df.to_sql("inventory_daily", con, if_exists="replace", index=False)


if __name__ == "__main__":
    data = generate_synthetic_data()
    print(f"Generated {len(data):,} rows")
    print(f"7-day stockout positive rate: {data['Stockout_Next_7D'].mean():.2%}")
