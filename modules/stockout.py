from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from modules.common import (
    RISK_COLOR,
    STOCK_STATUS_COLOR,
    business_filters,
    stop_if_empty,
    metrics,
    title,
    model_filter,
    as_of_filter,
    date_range_filter,
    apply_business_filter_state,
)
from core.features import risk_label, stock_status, stockout_recommendation
from core.modeling import predict_stockout, load_model, model_feature_importance


def snapshot(df: pd.DataFrame, model_name: str, as_of):
    s = df[pd.to_datetime(df["Date"]).dt.date == as_of].copy()
    if s.empty:
        return s

    s["Stockout_Probability"] = predict_stockout(s, model_name)
    s["Risk_Level"] = s["Stockout_Probability"].map(risk_label)
    s["Stock_Status"] = s.apply(stock_status, axis=1)

    recs = s.apply(stockout_recommendation, axis=1)
    s["Recommended_Action"] = recs.map(lambda x: x[0])
    s["Recommended_Order_Qty"] = recs.map(lambda x: x[1])
    s["Risk_Reason"] = recs.map(lambda x: x[2])

    s["Estimated_Stockout_Days"] = np.where(
        s["Avg_Daily_Sales_7D"] > 0,
        s["Available_Stock"] / s["Avg_Daily_Sales_7D"],
        np.nan,
    )
    s["Estimated_Stockout_Date"] = pd.to_datetime(s["Date"]) + pd.to_timedelta(
        np.ceil(s["Estimated_Stockout_Days"].fillna(999)), unit="D"
    )
    s["Inventory_Value"] = s["Available_Stock"] * s["Unit_Price"]
    return s


def dashboard(df, meta, all_metrics):
    title(
        "Stockout Risk Command Center",
        "Enterprise 7-day stockout intelligence. Change the ML model or any business filter and every KPI, chart and queue refreshes.",
    )

    model = model_filter(
        "ML Model",
        meta["stockout_models"],
        "stock_model",
        "Each model generates its own stockout probabilities and risk ranking.",
    )
    range_start, range_end = date_range_filter(df, "stock_date_range", default_days=30)
    as_of = range_end

    s = snapshot(df, model, as_of)
    s = business_filters(
        s,
        "stock",
        show_sku=True,
        show_stock_status=True,
        show_risk=True,
        show_action=True,
    )
    stop_if_empty(s)

    model_metric = all_metrics["stockout"][model]

    metrics([
        ("Active SKU-Locations", f"{len(s):,}"),
        ("High Risk", f"{(s['Risk_Level'] == 'HIGH').sum():,}"),
        ("Critical Risk", f"{(s['Risk_Level'] == 'CRITICAL').sum():,}"),
        ("Average Risk", f"{s['Stockout_Probability'].mean():.1%}"),
        ("Below Reorder Point", f"{(s['Available_Stock'] <= s['Reorder_Point']).sum():,}"),
        ("Model Recall", f"{model_metric['Recall']:.1%}"),
    ])

    st.caption(
        f"Selected model: {model} · PR-AUC {model_metric['PR_AUC']:.3f} · "
        f"Precision {model_metric['Precision']:.1%} · As-of {as_of}"
    )

    c1, c2 = st.columns(2)
    with c1:
        d = (
            s["Risk_Level"]
            .value_counts()
            .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
            .fillna(0)
            .reset_index()
        )
        d.columns = ["Risk", "Count"]
        fig = px.bar(
            d, x="Risk", y="Count", color="Risk", color_discrete_map=RISK_COLOR,
            title="Risk Distribution"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top = s.nlargest(15, "Stockout_Probability").sort_values("Stockout_Probability")
        fig = px.bar(
            top,
            x="Stockout_Probability",
            y="Product_Name",
            orientation="h",
            color="Risk_Level",
            color_discrete_map=RISK_COLOR,
            hover_data=["SKU_ID", "Warehouse_Name", "Available_Stock", "Stock_Coverage_Days"],
            title="Top Stockout Risk SKU-Locations",
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(
            s,
            x="Stock_Coverage_Days",
            y="Expected_Lead_Time_Days",
            size=np.clip(s["Avg_Daily_Sales_7D"], 1, None),
            color="Risk_Level",
            color_discrete_map=RISK_COLOR,
            hover_name="Product_Name",
            hover_data={
                "SKU_ID": True,
                "Warehouse_Name": True,
                "Stockout_Probability": ":.1%",
                "Available_Stock": True,
                "Stock_Coverage_Days": ":.1f",
                "Expected_Lead_Time_Days": ":.0f",
            },
            title="Stock Coverage vs Supplier Lead Time",
        )
        lim = max(
            float(s["Stock_Coverage_Days"].clip(upper=60).max() or 1),
            float(s["Expected_Lead_Time_Days"].max() or 1),
        )
        fig.add_shape(
            type="line", x0=0, y0=0, x1=lim, y1=lim,
            line=dict(color="rgba(245,158,11,.65)", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=min(lim * 0.65, 30), y=min(lim * 0.72, 25),
            text="Below diagonal = coverage < lead time",
            showarrow=False,
            font=dict(color="#F59E0B"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        d = (
            s.groupby("Product_Category", as_index=False)["Stockout_Probability"]
            .mean()
            .sort_values("Stockout_Probability")
        )
        fig = px.bar(
            d, x="Stockout_Probability", y="Product_Category",
            orientation="h", title="Average Stockout Risk by Category"
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        d = s.groupby(["Warehouse_Name", "Risk_Level"]).size().reset_index(name="SKUs")
        fig = px.bar(
            d, x="Warehouse_Name", y="SKUs", color="Risk_Level",
            color_discrete_map=RISK_COLOR, barmode="stack", title="Risk by Warehouse"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        d = (
            s.groupby("Supplier_Name", as_index=False)
            .agg(
                Delay_Rate=("Supplier_Delay_Rate", "mean"),
                At_Risk=("Stockout_Probability", lambda x: int((x >= 0.60).sum())),
                Avg_Lead=("Expected_Lead_Time_Days", "mean"),
            )
            .sort_values(["At_Risk", "Delay_Rate"], ascending=False)
            .head(15)
        )
        fig = px.scatter(
            d, x="Delay_Rate", y="At_Risk", size="Avg_Lead",
            hover_name="Supplier_Name", title="Supplier Risk Exposure"
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        status = s["Stock_Status"].value_counts().reset_index()
        status.columns = ["Stock_Status", "Count"]
        fig = px.bar(
            status, x="Stock_Status", y="Count", color="Stock_Status",
            color_discrete_map=STOCK_STATUS_COLOR, title="Current Stock Status"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c8:
        action = s["Recommended_Action"].value_counts().reset_index()
        action.columns = ["Recommended_Action", "Count"]
        fig = px.bar(
            action.sort_values("Count"),
            x="Count", y="Recommended_Action", orientation="h",
            title="Recommended Action Mix"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Model Risk Trend")
    trend_scope = df[
        (pd.to_datetime(df["Date"]).dt.date >= range_start)
        & (pd.to_datetime(df["Date"]).dt.date <= range_end)
    ].copy()
    # Reapply the same business scope used in the snapshot where practical.
    # Predict in one batch so the selected model genuinely drives the trend.
    if not trend_scope.empty:
        trend_scope["Stockout_Probability"] = predict_stockout(trend_scope, model)
        trend_scope["Risk_Level"] = trend_scope["Stockout_Probability"].map(risk_label)
        trend_scope["Stock_Status"] = trend_scope.apply(stock_status, axis=1)
        trend_recs = trend_scope.apply(stockout_recommendation, axis=1)
        trend_scope["Recommended_Action"] = trend_recs.map(lambda x: x[0])
        trend_scope = apply_business_filter_state(
            trend_scope,
            "stock",
            show_stock_status=True,
            show_risk=True,
            show_action=True,
        )
        if not trend_scope.empty:
            trend_daily = trend_scope.groupby("Date", as_index=False).agg(
                Average_Risk=("Stockout_Probability", "mean"),
                High_Critical=("Risk_Level", lambda x: int(pd.Series(x).isin(["HIGH", "CRITICAL"]).sum())),
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_daily["Date"], y=trend_daily["Average_Risk"],
                name="Average Stockout Risk", mode="lines+markers"
            ))
            fig.add_trace(go.Bar(
                x=trend_daily["Date"], y=trend_daily["High_Critical"],
                name="High + Critical SKU-Locations", opacity=0.32, yaxis="y2"
            ))
            fig.update_layout(
                title="Selected Model Risk Trend Across Date Range",
                yaxis=dict(title="Average Risk", tickformat=".0%"),
                yaxis2=dict(title="High + Critical", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Priority SKU Queue")
    cols = [
        "SKU_ID", "Product_Name", "Product_Category", "Warehouse_Name", "Supplier_Name",
        "Available_Stock", "Avg_Daily_Sales_7D", "Stock_Coverage_Days",
        "Expected_Lead_Time_Days", "Incoming_Qty_7D", "Stockout_Probability",
        "Risk_Level", "Stock_Status", "Estimated_Stockout_Date",
        "Recommended_Action", "Recommended_Order_Qty",
    ]
    out = s.sort_values(["Stockout_Probability", "Stock_Coverage_Days"], ascending=[False, True])[cols]
    st.dataframe(
        out,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Stockout_Probability": st.column_config.ProgressColumn(
                "Stockout Probability", min_value=0.0, max_value=1.0, format="percent"
            ),
            "Stock_Coverage_Days": st.column_config.NumberColumn("Coverage Days", format="%.1f"),
            "Estimated_Stockout_Date": st.column_config.DateColumn("Estimated Stockout"),
        },
    )
    st.download_button(
        "Download Filtered Risk Report",
        out.to_csv(index=False).encode("utf-8"),
        "stockout_risk_report.csv",
        "text/csv",
        use_container_width=True,
    )


def stock_report(df, meta):
    title(
        "Full Stock Report",
        "Enterprise inventory report combining current stock status with model-driven stockout risk.",
    )

    model = model_filter("ML Model", meta["stockout_models"], "stock_report_model")
    as_of = as_of_filter(df, "stock_report_date", recent_days=90)

    s = snapshot(df, model, as_of)
    s = business_filters(
        s,
        "stock_report",
        show_sku=True,
        show_stock_status=True,
        show_risk=True,
        show_action=True,
    )
    stop_if_empty(s)

    metrics([
        ("Inventory Units", f"{s['Available_Stock'].sum():,.0f}"),
        ("Inventory Value", f"PKR {s['Inventory_Value'].sum():,.0f}"),
        ("Below Safety Stock", f"{(s['Available_Stock'] < s['Safety_Stock']).sum():,}"),
        ("Below Reorder Point", f"{(s['Available_Stock'] <= s['Reorder_Point']).sum():,}"),
        ("High + Critical Risk", f"{s['Risk_Level'].isin(['HIGH','CRITICAL']).sum():,}"),
        ("Backorder Units", f"{s['Backorder_Qty'].sum():,.0f}"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        d = (
            s.groupby("Product_Category", as_index=False)["Inventory_Value"]
            .sum()
            .sort_values("Inventory_Value", ascending=False)
        )
        st.plotly_chart(
            px.bar(d, x="Product_Category", y="Inventory_Value", title="Inventory Value by Category"),
            use_container_width=True,
        )
    with c2:
        d = s.groupby("Warehouse_Name", as_index=False)["Inventory_Value"].sum()
        st.plotly_chart(
            px.bar(d, x="Warehouse_Name", y="Inventory_Value", title="Inventory Value by Warehouse"),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        d = s["Stock_Status"].value_counts().reset_index()
        d.columns = ["Status", "Count"]
        st.plotly_chart(
            px.bar(
                d, x="Status", y="Count", color="Status",
                color_discrete_map=STOCK_STATUS_COLOR, title="Current Stock Status"
            ),
            use_container_width=True,
        )
    with c4:
        low = s.nsmallest(20, "Stock_Coverage_Days").sort_values("Stock_Coverage_Days")
        st.plotly_chart(
            px.bar(
                low, x="Stock_Coverage_Days", y="Product_Name",
                orientation="h", color="Risk_Level", color_discrete_map=RISK_COLOR,
                title="Lowest Stock Coverage"
            ),
            use_container_width=True,
        )

    cols = [
        "SKU_ID", "Product_Name", "Product_Category", "Warehouse_Name", "Supplier_Name",
        "Available_Stock", "Safety_Stock", "Reorder_Point", "Stock_Coverage_Days",
        "On_Order_Qty", "Incoming_Qty_7D", "Backorder_Qty", "Inventory_Value",
        "Stock_Status", "Stockout_Probability", "Risk_Level", "Recommended_Action",
    ]
    out = s[cols].sort_values(
        ["Stockout_Probability", "Stock_Coverage_Days"], ascending=[False, True]
    )
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Full Stock Report",
        out.to_csv(index=False).encode("utf-8"),
        "full_stock_report.csv",
        "text/csv",
        use_container_width=True,
    )


def sku_analysis(df, meta):
    title(
        "SKU Analysis",
        "Drill into a single SKU-location and inspect its model-specific risk, history and business action.",
    )
    model = model_filter("ML Model", meta["stockout_models"], "sku_stock_model")
    as_of = as_of_filter(df, "sku_stock_date", recent_days=180)

    base = df[pd.to_datetime(df["Date"]).dt.date <= as_of].copy()
    sku = st.selectbox("SKU", sorted(base["SKU_ID"].unique()), key="sku_stock_sku")
    d = base[base["SKU_ID"] == sku]
    wh = st.selectbox("Warehouse", sorted(d["Warehouse_Name"].unique()), key="sku_stock_wh")
    d = d[d["Warehouse_Name"] == wh].sort_values("Date")

    latest = d.tail(1).copy()
    latest["Stockout_Probability"] = predict_stockout(latest, model)
    latest["Risk_Level"] = latest["Stockout_Probability"].map(risk_label)
    latest["Stock_Status"] = latest.apply(stock_status, axis=1)

    row = latest.iloc[0]
    action, qty, reason = stockout_recommendation(row)

    metrics([
        ("Stockout Probability", f"{row['Stockout_Probability']:.1%}"),
        ("Risk", row["Risk_Level"]),
        ("Current Status", row["Stock_Status"]),
        ("Available Stock", f"{row['Available_Stock']:,.0f}"),
        ("Coverage", f"{row['Stock_Coverage_Days']:.1f} days"),
        ("Recommended Qty", f"{qty:,.0f}"),
    ])

    hist = d.tail(90)
    fig = go.Figure()
    for col, name in [
        ("Available_Stock", "Available Stock"),
        ("Daily_Sales", "Daily Sales"),
        ("Avg_Daily_Sales_7D", "7D Avg Demand"),
        ("Safety_Stock", "Safety Stock"),
        ("Reorder_Point", "Reorder Point"),
    ]:
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist[col], name=name))
    fig.update_layout(title="Inventory and Demand History — Last 90 Days")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Why is this SKU at risk?")
        for item in reason.split("; "):
            st.write(f"- {item}")
    with c2:
        st.markdown("### Decision Support")
        st.info(
            f"**Recommended Action:** {action}\n\n"
            f"**Decision-support order quantity:** {qty:,.0f} units\n\n"
            f"**Supplier Lead Time:** {row['Expected_Lead_Time_Days']:.0f} days\n\n"
            f"**Incoming in 7 Days:** {row['Incoming_Qty_7D']:,.0f} units"
        )


def model_lab(df, meta, all_metrics):
    title(
        "Stockout ML Model Lab",
        "Evaluate model behavior by business segment. Threshold changes recalculate precision, recall, F1 and the confusion matrix.",
    )

    model_name = model_filter("Model", meta["stockout_models"], "stock_lab_model")
    with st.sidebar:
        threshold = st.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.05, key="stock_lab_thr")

    dates = sorted(pd.to_datetime(df["Date"]).unique())
    e = df[pd.to_datetime(df["Date"]) >= dates[int(len(dates) * 0.85)]].copy()
    e = business_filters(e, "stock_lab", show_date_range=True, show_sku=True)
    stop_if_empty(e)

    p = predict_stockout(e, model_name)
    y = e["Stockout_Next_7D"].to_numpy()
    pred = (p >= threshold).astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    pr = precision_score(y, pred, zero_division=0)
    rc = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    base = all_metrics["stockout"][model_name]

    metrics([
        ("Evaluation Rows", f"{len(e):,}"),
        ("Precision", f"{pr:.3f}"),
        ("Recall", f"{rc:.3f}"),
        ("F1", f"{f1:.3f}"),
        ("ROC-AUC", f"{base['ROC_AUC']:.3f}" if base["ROC_AUC"] is not None else "n/a"),
        ("PR-AUC", f"{base['PR_AUC']:.3f}" if base["PR_AUC"] is not None else "n/a"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        cm_df = pd.DataFrame(cm, index=["Actual Safe", "Actual Stockout"], columns=["Pred Safe", "Pred Stockout"])
        st.plotly_chart(
            px.imshow(cm_df, text_auto=True, title=f"Confusion Matrix @ Threshold {threshold:.2f}"),
            use_container_width=True,
        )
    with c2:
        imp = model_feature_importance(load_model("stockout", model_name))
        if not imp.empty:
            st.plotly_chart(
                px.bar(
                    imp.sort_values("Importance"), x="Importance", y="Feature",
                    orientation="h", title="Global Feature Importance"
                ),
                use_container_width=True,
            )
        else:
            st.info("Feature importance is unavailable for this model.")

    comp = pd.DataFrame(all_metrics["stockout"]).T.reset_index().rename(columns={"index": "Model"})
    st.markdown("### Global Model Comparison")
    st.dataframe(
        comp[["Model", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]],
        use_container_width=True,
        hide_index=True,
    )
    long = comp.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1"],
        var_name="Metric",
        value_name="Score",
    )
    st.plotly_chart(
        px.bar(long, x="Model", y="Score", color="Metric", barmode="group", title="Precision / Recall / F1"),
        use_container_width=True,
    )
