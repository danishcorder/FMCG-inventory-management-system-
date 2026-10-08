from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from modules.common import (
    business_filters,
    stop_if_empty,
    metrics,
    title,
    model_filter,
    horizon_filter,
    as_of_filter,
    date_range_filter,
    apply_business_filter_state,
)
from core.modeling import predict_demand7, load_model, model_feature_importance
from core.features import replenishment_values


def snapshot_at(df: pd.DataFrame, as_of):
    return df[pd.to_datetime(df["Date"]).dt.date == as_of].copy()


def forecast_snapshot(df: pd.DataFrame, model_name: str, horizon: int, as_of):
    s = snapshot_at(df, as_of)
    if s.empty:
        return s

    s["Forecast_7D"] = predict_demand7(s, model_name)
    s["Forecast_Horizon"] = s["Forecast_7D"] * (horizon / 7.0)

    vals = s.apply(
        lambda r: replenishment_values(r, r["Forecast_Horizon"], horizon_days=horizon),
        axis=1,
    )
    for col in [
        "Lead_Time_Demand",
        "Dynamic_Safety_Stock",
        "Dynamic_Reorder_Point",
        "Inventory_Position_for_Plan",
        "Recommended_Order_Qty",
        "Order_Now",
    ]:
        s[col] = vals.map(lambda x, c=col: x[c])

    s["Overstock_Risk"] = s["Stock_Coverage_Days"] > 60
    s["Inventory_Value"] = s["Available_Stock"] * s["Unit_Price"]
    s["Projected_End_Stock"] = s["Inventory_Position_for_Plan"] - s["Forecast_Horizon"]
    s["Projected_Stockout_Risk"] = s["Projected_End_Stock"] < 0
    s["Excess_Inventory_Value"] = np.where(s["Overstock_Risk"], s["Inventory_Value"], 0.0)

    daily_forecast = s["Forecast_Horizon"] / max(horizon, 1)
    s["Days_To_Reorder_Point"] = np.where(
        daily_forecast > 0,
        (s["Inventory_Position_for_Plan"] - s["Dynamic_Reorder_Point"]) / daily_forecast,
        np.inf,
    )
    s["Order_Within_3_Days"] = (
        (~s["Order_Now"])
        & (s["Days_To_Reorder_Point"] >= 0)
        & (s["Days_To_Reorder_Point"] <= 3)
    )

    s["Replenishment_Alert"] = np.select(
        [
            s["Projected_Stockout_Risk"],
            s["Order_Now"],
            s["Order_Within_3_Days"],
            s["Overstock_Risk"],
        ],
        ["STOCKOUT RISK", "ORDER NOW", "ORDER WITHIN 3 DAYS", "OVERSTOCK RISK"],
        default="NO ORDER REQUIRED",
    )
    s["Recommended_Action"] = np.select(
        [
            s["Projected_Stockout_Risk"] & s["Order_Now"],
            s["Order_Now"],
            s["Order_Within_3_Days"],
            s["Overstock_Risk"],
        ],
        ["URGENT ORDER", "ORDER NOW", "PREPARE ORDER", "MONITOR OVERSTOCK"],
        default="WAIT",
    )
    return s


def dashboard(df, meta, all_metrics):
    title(
        "Demand Prediction & Replenishment Dashboard",
        "Enterprise demand forecasting and purchase planning. Model, date, horizon and business filters update the complete decision view.",
    )

    model = model_filter(
        "Forecast Model",
        meta["demand_models"],
        "demand_model",
        "The selected forecasting model drives forecast demand and replenishment calculations.",
    )
    horizon = horizon_filter("demand_horizon")
    range_start, range_end = date_range_filter(df, "demand_date_range", default_days=30)
    as_of = range_end

    s = forecast_snapshot(df, model, horizon, as_of)
    s = business_filters(
        s,
        "demand",
        show_sku=True,
        show_action=True,
        show_projected_risk=True,
        show_overstock=True,
    )
    stop_if_empty(s)

    m = all_metrics["demand"][model]

    metrics([
        (f"Forecast Demand ({horizon}d)", f"{s['Forecast_Horizon'].sum():,.0f}"),
        ("Recommended Purchase", f"{s['Recommended_Order_Qty'].sum():,.0f}"),
        ("Order-Now Locations", f"{int(s['Order_Now'].sum()):,}"),
        ("Order Within 3 Days", f"{int(s['Order_Within_3_Days'].sum()):,}"),
        ("Projected Stockout", f"{int(s['Projected_Stockout_Risk'].sum()):,}"),
        ("WMAPE", f"{m['WMAPE']:.1%}"),
    ])
    st.caption(
        f"Selected model: {model} · Forecast Bias {m['Forecast_Bias']:.1%} · "
        f"Inventory Value PKR {s['Inventory_Value'].sum():,.0f} · Overstock candidates {int(s['Overstock_Risk'].sum()):,} · As-of {as_of}"
    )

    as_of_ts = pd.Timestamp(range_end)
    recent = df[
        (pd.to_datetime(df["Date"]) >= as_of_ts - pd.Timedelta(days=29))
        & (pd.to_datetime(df["Date"]) <= as_of_ts)
    ].copy()
    recent = apply_business_filter_state(recent, "demand")
    total_demand = max(float(recent["Demand_Qty"].sum()), 1.0)
    fill_rate = float(recent["Daily_Sales"].sum()) / total_demand
    stockout_rate = float((recent["Lost_Sales"] > 0).mean())
    lost_sales_value = float((recent["Lost_Sales"] * recent["Unit_Price"]).sum())
    sales_value = float((recent["Daily_Sales"] * recent["Unit_Price"]).sum())
    avg_inv_value = max(float((recent["Available_Stock"] * recent["Unit_Price"]).mean()), 1.0)
    sku_locations = max(recent[["SKU_ID", "Warehouse_ID"]].drop_duplicates().shape[0], 1)
    turnover_30d = sales_value / max(avg_inv_value * sku_locations, 1.0)

    st.markdown("#### Recent 30-Day Operating KPIs")
    metrics([
        ("Fill Rate", f"{fill_rate:.1%}"),
        ("Observed Stockout Rate", f"{stockout_rate:.1%}"),
        ("Lost Sales Estimate", f"PKR {lost_sales_value:,.0f}"),
        ("Approx. Inventory Turnover", f"{turnover_30d:.2f}x"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        top = s.nlargest(15, "Forecast_Horizon").sort_values("Forecast_Horizon")
        st.plotly_chart(
            px.bar(
                top,
                x="Forecast_Horizon",
                y="Product_Name",
                orientation="h",
                color="Product_Category",
                hover_data=["SKU_ID", "Warehouse_Name", "Available_Stock"],
                title=f"Top Forecast Demand — Next {horizon} Days",
            ),
            use_container_width=True,
        )
    with c2:
        d = (
            s.groupby("Product_Category", as_index=False)["Forecast_Horizon"]
            .sum()
            .sort_values("Forecast_Horizon")
        )
        st.plotly_chart(
            px.bar(
                d,
                x="Forecast_Horizon",
                y="Product_Category",
                orientation="h",
                title="Forecast Demand by Product Category",
            ),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        top = s.nlargest(15, "Recommended_Order_Qty").sort_values("Recommended_Order_Qty")
        st.plotly_chart(
            px.bar(
                top,
                x="Recommended_Order_Qty",
                y="Product_Name",
                orientation="h",
                color="Recommended_Action",
                hover_data=["SKU_ID", "Warehouse_Name", "Dynamic_Reorder_Point"],
                title="Purchase Priority",
            ),
            use_container_width=True,
        )
    with c4:
        d = s.groupby("Supplier_Name", as_index=False).agg(
            Avg_Lead=("Expected_Lead_Time_Days", "mean"),
            Avg_Delay=("Supplier_Delay_Rate", "mean"),
            Purchase_Qty=("Recommended_Order_Qty", "sum"),
        )
        fig = px.scatter(
            d,
            x="Avg_Delay",
            y="Purchase_Qty",
            size="Avg_Lead",
            hover_name="Supplier_Name",
            title="Supplier Delay vs Purchase Exposure",
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fig = px.scatter(
            s,
            x="Stock_Coverage_Days",
            y="Expected_Lead_Time_Days",
            size=np.clip(s["Forecast_Horizon"], 1, None),
            color="Recommended_Action",
            hover_name="Product_Name",
            hover_data=["SKU_ID", "Warehouse_Name", "Projected_End_Stock"],
            title="Coverage vs Lead Time — Replenishment Pressure",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        w = s.groupby("Warehouse_Name", as_index=False).agg(
            Forecast_Demand=("Forecast_Horizon", "sum"),
            Recommended_Order=("Recommended_Order_Qty", "sum"),
            At_Risk=("Projected_Stockout_Risk", "sum"),
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(x=w["Warehouse_Name"], y=w["Forecast_Demand"], name="Forecast Demand"))
        fig.add_trace(go.Bar(x=w["Warehouse_Name"], y=w["Recommended_Order"], name="Recommended Order"))
        fig.update_layout(barmode="group", title="Warehouse Demand vs Recommended Purchase")
        st.plotly_chart(fig, use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        action = s["Recommended_Action"].value_counts().reset_index()
        action.columns = ["Action", "Count"]
        st.plotly_chart(
            px.bar(
                action.sort_values("Count"),
                x="Count",
                y="Action",
                orientation="h",
                title="Decision Mix",
            ),
            use_container_width=True,
        )
    with c8:
        risk = pd.DataFrame({
            "Status": ["Projected Stockout", "Overstock", "Normal"],
            "Count": [
                int(s["Projected_Stockout_Risk"].sum()),
                int(s["Overstock_Risk"].sum()),
                int((~s["Projected_Stockout_Risk"] & ~s["Overstock_Risk"]).sum()),
            ],
        })
        st.plotly_chart(
            px.bar(risk, x="Status", y="Count", color="Status", title="Inventory Planning Risk"),
            use_container_width=True,
        )

    c9, c10 = st.columns(2)
    with c9:
        movers = s.groupby("Fast_Slow_Mover", as_index=False).agg(
            SKU_Locations=("SKU_ID", "count"),
            Inventory_Value=("Inventory_Value", "sum"),
            Forecast_Demand=("Forecast_Horizon", "sum"),
        )
        st.plotly_chart(
            px.bar(
                movers,
                x="Fast_Slow_Mover",
                y="Inventory_Value",
                color="Fast_Slow_Mover",
                title="Inventory Value by Mover Class",
            ),
            use_container_width=True,
        )
    with c10:
        slow = s[s["Fast_Slow_Mover"].isin(["Slow", "Medium"])].nlargest(15, "Stock_Coverage_Days")
        if not slow.empty:
            st.plotly_chart(
                px.bar(
                    slow.sort_values("Stock_Coverage_Days"),
                    x="Stock_Coverage_Days",
                    y="Product_Name",
                    orientation="h",
                    color="Fast_Slow_Mover",
                    title="Slow / Medium-Moving SKUs with Highest Coverage",
                ),
                use_container_width=True,
            )
        else:
            st.info("No slow/medium-moving SKUs in the filtered scope.")

    st.markdown("### Forecast Validation Trend")
    trend_scope = df[
        (pd.to_datetime(df["Date"]).dt.date >= range_start)
        & (pd.to_datetime(df["Date"]).dt.date <= range_end)
    ].copy()
    if not trend_scope.empty:
        trend_scope = apply_business_filter_state(trend_scope, "demand")
        if not trend_scope.empty:
            trend_scope["Predicted_7D"] = predict_demand7(trend_scope, model)
            by_date = trend_scope.groupby("Date", as_index=False).agg(
                Actual_Next_7D=("Demand_Next_7D", "sum"),
                Predicted_Next_7D=("Predicted_7D", "sum"),
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=by_date["Date"], y=by_date["Actual_Next_7D"],
                name="Actual Next-7D Demand", mode="lines"
            ))
            fig.add_trace(go.Scatter(
                x=by_date["Date"], y=by_date["Predicted_Next_7D"],
                name=f"{model} Prediction", mode="lines"
            ))
            fig.update_layout(title="Selected Model — Aggregate Forecast vs Actual")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Replenishment Priority Table")
    cols = [
        "SKU_ID", "Product_Name", "Product_Category", "Warehouse_Name", "Supplier_Name",
        "Forecast_Horizon", "Available_Stock", "Incoming_Qty_7D",
        "Dynamic_Safety_Stock", "Dynamic_Reorder_Point", "Projected_End_Stock",
        "Projected_Stockout_Risk", "Order_Now", "Order_Within_3_Days", "Replenishment_Alert", "Recommended_Action",
        "Recommended_Order_Qty", "Stock_Coverage_Days",
    ]
    out = s.sort_values(
        ["Order_Now", "Projected_Stockout_Risk", "Recommended_Order_Qty"],
        ascending=[False, False, False],
    )[cols]
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Replenishment Plan",
        out.to_csv(index=False).encode("utf-8"),
        "replenishment_plan.csv",
        "text/csv",
        use_container_width=True,
    )


def planner(df, meta):
    title(
        "Replenishment Planner",
        "Purchase decision workspace: what to order, how much to order, when to act and why.",
    )

    model = model_filter("Forecast Model", meta["demand_models"], "planner_model")
    horizon = horizon_filter("planner_horizon")
    as_of = as_of_filter(df, "planner_date", recent_days=90)

    s = forecast_snapshot(df, model, horizon, as_of)
    s = business_filters(
        s,
        "planner",
        show_sku=True,
        show_action=True,
        show_projected_risk=True,
        show_overstock=True,
    )
    stop_if_empty(s)

    metrics([
        ("Filtered SKU-Locations", f"{len(s):,}"),
        ("Order Now", f"{int(s['Order_Now'].sum()):,}"),
        ("Projected Stockout", f"{int(s['Projected_Stockout_Risk'].sum()):,}"),
        ("Recommended Qty", f"{s['Recommended_Order_Qty'].sum():,.0f}"),
        ("Purchase Value", f"PKR {(s['Recommended_Order_Qty'] * s['Unit_Price']).sum():,.0f}"),
    ])

    cols = [
        "SKU_ID", "Product_Name", "Product_Category", "Warehouse_Name", "Supplier_Name",
        "Forecast_Horizon", "Available_Stock", "Incoming_Qty_7D", "Dynamic_Safety_Stock",
        "Dynamic_Reorder_Point", "MOQ", "Projected_End_Stock", "Order_Now", "Order_Within_3_Days",
        "Replenishment_Alert", "Recommended_Action", "Recommended_Order_Qty",
    ]
    out = s.sort_values(
        ["Order_Now", "Projected_Stockout_Risk", "Recommended_Order_Qty"],
        ascending=[False, False, False],
    )[cols]
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.download_button(
        "Export Purchase Recommendations",
        out.to_csv(index=False).encode("utf-8"),
        "purchase_recommendations.csv",
        "text/csv",
        use_container_width=True,
    )

    st.markdown("### Recommendation Explanation")
    sku = st.selectbox("SKU for Explanation", sorted(s["SKU_ID"].unique()), key="planner_sku")
    wh = st.selectbox(
        "Warehouse for Explanation",
        sorted(s[s["SKU_ID"] == sku]["Warehouse_Name"].unique()),
        key="planner_wh",
    )
    row = s[(s["SKU_ID"] == sku) & (s["Warehouse_Name"] == wh)].iloc[0]

    st.info(
        f"**{row['Product_Name']}** — forecast for next {horizon} days: **{row['Forecast_Horizon']:.0f} units**. "
        f"Available: **{row['Available_Stock']:.0f}**, incoming: **{row['Incoming_Qty_7D']:.0f}**, "
        f"dynamic safety stock: **{row['Dynamic_Safety_Stock']:.0f}**, "
        f"dynamic reorder point: **{row['Dynamic_Reorder_Point']:.0f}**, "
        f"supplier lead time: **{row['Expected_Lead_Time_Days']:.0f} days**, MOQ: **{row['MOQ']:.0f}**. "
        f"Alert: **{row['Replenishment_Alert']}**. Decision: **{row['Recommended_Action']}**. "
        f"Recommended order: **{row['Recommended_Order_Qty']:.0f} units**."
    )


def sku_forecast(df, meta):
    title(
        "SKU Demand Forecast",
        "Historical demand and model-driven forward demand planning for one SKU-location.",
    )

    model = model_filter("Forecast Model", meta["demand_models"], "sku_fc_model")
    as_of = as_of_filter(df, "sku_fc_date", recent_days=180)

    base = df[pd.to_datetime(df["Date"]).dt.date <= as_of].copy()
    sku = st.selectbox("SKU", sorted(base["SKU_ID"].unique()), key="sku_fc_sku")
    d = base[base["SKU_ID"] == sku]
    wh = st.selectbox("Warehouse", sorted(d["Warehouse_Name"].unique()), key="sku_fc_wh")
    d = d[d["Warehouse_Name"] == wh].sort_values("Date")

    latest = d.tail(1).copy()
    f7 = float(predict_demand7(latest, model)[0])
    daily = f7 / 7.0

    hist = d.tail(90)
    dow = hist.groupby(hist["Date"].dt.dayofweek)["Demand_Qty"].mean()
    overall = max(hist["Demand_Qty"].mean(), 1.0)
    mult = {i: float(dow.get(i, overall) / overall) for i in range(7)}

    future_dates = pd.date_range(d["Date"].max() + pd.Timedelta(days=1), periods=30, freq="D")
    raw = np.array([daily * mult[x.dayofweek] for x in future_dates], dtype=float)
    raw[:7] *= f7 / max(raw[:7].sum(), 1e-9)
    fc = pd.DataFrame({"Date": future_dates, "Forecast_Demand": np.maximum(0, raw)})

    metrics([
        ("Selected Model", model),
        ("Next 7D Forecast", f"{f7:,.0f}"),
        ("Avg Daily Forecast", f"{daily:,.1f}"),
        ("Current Stock", f"{latest.iloc[0]['Available_Stock']:,.0f}"),
        ("Current Coverage", f"{latest.iloc[0]['Stock_Coverage_Days']:.1f} days"),
    ])

    fig = go.Figure()
    hist60 = d.tail(60)
    fig.add_trace(go.Scatter(x=hist60["Date"], y=hist60["Demand_Qty"], name="Historical Demand"))
    fig.add_trace(go.Scatter(x=fc["Date"], y=fc["Forecast_Demand"], name="Forward Forecast"))
    fig.update_layout(title="Historical vs Forward Demand")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        weekly = d.tail(90).copy()
        weekly["Week"] = weekly["Date"].dt.to_period("W").astype(str)
        weekly = weekly.groupby("Week", as_index=False)["Demand_Qty"].sum().tail(12)
        st.plotly_chart(
            px.bar(weekly, x="Week", y="Demand_Qty", title="Recent Weekly Demand"),
            use_container_width=True,
        )
    with c2:
        day_profile = hist.groupby(hist["Date"].dt.day_name(), as_index=False)["Demand_Qty"].mean()
        st.plotly_chart(
            px.bar(day_profile, x="Date", y="Demand_Qty", title="Day-of-Week Demand Profile"),
            use_container_width=True,
        )

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        y = d.set_index("Date")["Demand_Qty"].asfreq("D").fillna(0)
        if len(y) >= 60:
            hw = ExponentialSmoothing(
                y, trend="add", seasonal="add", seasonal_periods=7
            ).fit(optimized=True)
            pred = hw.forecast(14)
            h = pd.DataFrame({"Date": pred.index, "Holt_Winters": pred.values})
            st.plotly_chart(
                px.line(h, x="Date", y="Holt_Winters", title="Holt-Winters Statistical Baseline — 14 Days"),
                use_container_width=True,
            )
    except Exception:
        st.caption("Holt-Winters baseline is unavailable in this environment.")


def model_lab(df, meta, all_metrics):
    title(
        "Demand Forecast Model Lab",
        "Compare model error and bias globally or within a filtered business segment.",
    )

    model = model_filter("Model", meta["demand_models"], "demand_lab_model")
    dates = sorted(pd.to_datetime(df["Date"]).unique())
    e = df[pd.to_datetime(df["Date"]) >= dates[int(len(dates) * 0.85)]].copy()
    e = business_filters(e, "demand_lab", show_date_range=True, show_sku=True)
    stop_if_empty(e)

    pred = predict_demand7(e, model)
    actual = e["Demand_Next_7D"].to_numpy(dtype=float)
    pred = np.maximum(0, np.asarray(pred, dtype=float))

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    wmape = np.abs(actual - pred).sum() / max(np.abs(actual).sum(), 1.0)
    bias = (pred - actual).sum() / max(np.abs(actual).sum(), 1.0)
    r2 = r2_score(actual, pred)

    metrics([
        ("Evaluation Rows", f"{len(e):,}"),
        ("MAE", f"{mae:.2f}"),
        ("RMSE", f"{rmse:.2f}"),
        ("WMAPE", f"{wmape:.1%}"),
        ("Forecast Bias", f"{bias:.1%}"),
        ("R²", f"{r2:.3f}"),
    ])

    comp = pd.DataFrame(all_metrics["demand"]).T.reset_index().rename(columns={"index": "Model"})
    st.markdown("### Global Model Comparison")
    st.dataframe(comp, use_container_width=True, hide_index=True)
    long = comp.melt(
        id_vars="Model",
        value_vars=["WMAPE", "Forecast_Bias"],
        var_name="Metric",
        value_name="Value",
    )
    st.plotly_chart(
        px.bar(long, x="Model", y="Value", color="Metric", barmode="group", title="Forecast Error Comparison"),
        use_container_width=True,
    )

    e = e.copy()
    e["Predicted_7D"] = pred
    by_date = (
        e.groupby("Date", as_index=False)
        .agg(Actual_7D=("Demand_Next_7D", "mean"), Predicted_7D=("Predicted_7D", "mean"))
        .tail(40)
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=by_date["Date"], y=by_date["Actual_7D"], name="Actual Next-7D Demand"))
    fig.add_trace(go.Scatter(x=by_date["Date"], y=by_date["Predicted_7D"], name="Predicted Next-7D Demand"))
    fig.update_layout(title="Forecast vs Actual — Filtered Evaluation Scope")
    st.plotly_chart(fig, use_container_width=True)

    if model != "Moving Average":
        imp = model_feature_importance(load_model("demand", model))
        if not imp.empty:
            st.plotly_chart(
                px.bar(
                    imp.sort_values("Importance"),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Global Demand Model Feature Importance",
                ),
                use_container_width=True,
            )


def supplier_performance(df):
    title(
        "Supplier Performance",
        "Enterprise supplier intelligence covering lead time, reliability, delay rate, incoming stock and backorder exposure.",
    )

    as_of = as_of_filter(df, "supplier_date", recent_days=90)
    s = snapshot_at(df, as_of)
    s = business_filters(s, "supplier_perf", show_sku=True)
    stop_if_empty(s)

    d = s.groupby("Supplier_Name", as_index=False).agg(
        Avg_Lead_Time=("Expected_Lead_Time_Days", "mean"),
        Avg_Actual_Lead=("Actual_Avg_Lead_Time", "mean"),
        Delay_Rate=("Supplier_Delay_Rate", "mean"),
        Reliability=("Supplier_Reliability", "mean"),
        Incoming_7D=("Incoming_Qty_7D", "sum"),
        Backorders=("Backorder_Qty", "sum"),
        Inventory_Exposure=("Available_Stock", "sum"),
    )

    metrics([
        ("Suppliers", str(len(d))),
        ("Avg Delay Rate", f"{d['Delay_Rate'].mean():.1%}"),
        ("Avg Reliability", f"{d['Reliability'].mean():.1%}"),
        ("Incoming in 7D", f"{d['Incoming_7D'].sum():,.0f}"),
        ("Backorder Units", f"{d['Backorders'].sum():,.0f}"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.bar(
                d.sort_values("Delay_Rate"),
                x="Delay_Rate",
                y="Supplier_Name",
                orientation="h",
                title="Supplier Delay Rate",
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.scatter(
                d,
                x="Avg_Lead_Time",
                y="Reliability",
                size=np.clip(d["Inventory_Exposure"], 1, None),
                color="Delay_Rate",
                hover_name="Supplier_Name",
                title="Lead Time vs Reliability",
            ),
            use_container_width=True,
        )

    st.dataframe(d.sort_values("Delay_Rate", ascending=False), use_container_width=True, hide_index=True)
