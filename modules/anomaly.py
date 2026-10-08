from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from modules.common import title, metrics, stop_if_empty
from core.anomaly_modeling import (
    score_column,
    severity_column,
    severity_label,
)

SEVERITY_COLORS = {
    "NORMAL": "#10B981",
    "MODERATE": "#F59E0B",
    "HIGH": "#F97316",
    "CRITICAL": "#EF4444",
}

CATEGORY_COLORS = {
    "Transaction": "#7C3AED",
    "Quantity": "#EF4444",
    "Time-based": "#F59E0B",
    "Product": "#10B981",
    "Warehouse": "#8B5CF6",
    "Supplier": "#F97316",
    "Normal": "#64748B",
}


def _unique(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().astype(str).unique().tolist())


def anomaly_filters(df: pd.DataFrame, model_name: str, key: str):
    out = df.copy()
    score_col = score_column(model_name)
    sev_col = severity_column(model_name)
    out["Anomaly_Score"] = out[score_col]
    out["Severity"] = out[sev_col]
    out["Detected_Anomaly"] = out["Anomaly_Score"] > 30

    with st.sidebar:
        st.markdown("### Anomaly Filters")
        st.markdown(
            '<div class="unicorn-filter-caption">Every KPI, chart, alert and table uses the selected ML model plus these cumulative filters.</div>',
            unsafe_allow_html=True,
        )

        dates = pd.to_datetime(out["Date"])
        min_d, max_d = dates.min().date(), dates.max().date()
        default_start = max(min_d, (pd.Timestamp(max_d) - pd.Timedelta(days=29)).date())
        selected = st.date_input(
            "Date Range",
            value=(default_start, max_d),
            min_value=min_d,
            max_value=max_d,
            key=f"{key}_date",
        )

        warehouse = st.selectbox("Warehouse", ["All"] + _unique(out, "Warehouse_Name"), key=f"{key}_wh")
        category = st.selectbox("Category", ["All"] + _unique(out, "Product_Category"), key=f"{key}_cat")
        brand = st.selectbox("Brand", ["All"] + _unique(out, "Brand"), key=f"{key}_brand")
        sku = st.selectbox("SKU", ["All"] + _unique(out, "SKU_ID"), key=f"{key}_sku")
        supplier = st.selectbox("Supplier", ["All"] + _unique(out, "Supplier_Name"), key=f"{key}_sup")
        batch = st.selectbox("Batch", ["All"] + _unique(out, "Batch_ID"), key=f"{key}_batch")
        anomaly_type = st.selectbox(
            "Anomaly Type",
            ["All"] + [x for x in _unique(out, "Anomaly_Type") if x != "Normal"],
            key=f"{key}_type",
        )
        severity = st.selectbox(
            "Severity",
            ["All", "NORMAL", "MODERATE", "HIGH", "CRITICAL"],
            key=f"{key}_severity",
        )
        stock_status = st.selectbox(
            "Stock Status",
            ["All", "NORMAL", "LOW", "OVERSTOCK", "EXPIRY RISK"],
            key=f"{key}_stock",
        )
        detected_only = st.checkbox("Detected anomalies only", value=False, key=f"{key}_detected")

    if isinstance(selected, (list, tuple)) and len(selected) == 2:
        start, end = selected
        d = pd.to_datetime(out["Date"]).dt.date
        out = out[(d >= start) & (d <= end)]

    if warehouse != "All":
        out = out[out["Warehouse_Name"].astype(str) == warehouse]
    if category != "All":
        out = out[out["Product_Category"].astype(str) == category]
    if brand != "All":
        out = out[out["Brand"].astype(str) == brand]
    if sku != "All":
        out = out[out["SKU_ID"].astype(str) == sku]
    if supplier != "All":
        out = out[out["Supplier_Name"].astype(str) == supplier]
    if batch != "All":
        out = out[out["Batch_ID"].astype(str) == batch]
    if anomaly_type != "All":
        out = out[out["Anomaly_Type"].astype(str) == anomaly_type]
    if severity != "All":
        out = out[out["Severity"].astype(str) == severity]
    if stock_status != "All":
        out = out[out["Stock_Status_Anomaly"].astype(str) == stock_status]
    if detected_only:
        out = out[out["Detected_Anomaly"]]

    return out


def _model_selector(meta: dict, key: str):
    with st.sidebar:
        st.markdown("### ML Controls")
        return st.selectbox(
            "Anomaly Model",
            meta["models"],
            key=key,
            help="Switching the model changes the anomaly score, severity, detected count, ranking, charts and risk value.",
        )


def _at_risk_value(df: pd.DataFrame) -> float:
    d = df[df["Anomaly_Score"] > 30]
    return float(d["Inventory_Value"].clip(lower=0).sum())


def dashboard(df: pd.DataFrame, meta: dict, model_metrics: dict):
    title(
        "Inventory Anomaly Detection Command Center",
        "Detect abnormal stock behavior, rank severity, explain the reason and support corrective action across FMCG inventory.",
    )

    model = _model_selector(meta, "anomaly_dashboard_model")
    f = anomaly_filters(df, model, "anomaly_dashboard")
    stop_if_empty(f)

    detected = f[f["Anomaly_Score"] > 30].copy()
    critical = f[f["Severity"] == "CRITICAL"].copy()
    expiry = f[(f["Days_To_Expiry"] <= 30) & (f["Observed_Closing_Stock"] > 0)]
    slow = f[(f["Days_of_Inventory"] > 75) & (f["Observed_Closing_Stock"] > 0)]
    suspicious_adjustments = f[
        (f["Adjustment_Ratio"] > 0.15) | (f["Late_Adjustment_Flag"] == 1)
    ]

    metrics([
        ("Inventory Value at Risk", f"PKR {_at_risk_value(f):,.0f}"),
        ("Detected Anomalies", f"{len(detected):,}"),
        ("Critical SKU-Locations", f"{critical[['SKU_ID','Warehouse_ID']].drop_duplicates().shape[0]:,}"),
        ("Expiry Risk", f"{len(expiry):,}"),
        ("Slow / Dead Stock", f"{len(slow):,}"),
        ("Suspicious Adjustments", f"{len(suspicious_adjustments):,}"),
    ])

    mm = model_metrics[model]
    st.caption(
        f"Selected model: {model} · Precision {mm['Precision']:.1%} · Recall {mm['Recall']:.1%} · "
        f"F1 {mm['F1']:.1%} · Evaluation execution time {mm['Execution_Time_Seconds']:.2f}s"
    )

    c1, c2 = st.columns(2)
    with c1:
        daily = (
            detected.groupby("Date", as_index=False)
            .agg(
                Anomalies=("Transaction_ID", "count"),
                Avg_Score=("Anomaly_Score", "mean"),
            )
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["Date"], y=daily["Anomalies"], name="Detected Anomalies"))
        fig.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily["Avg_Score"],
                name="Average Risk Score",
                yaxis="y2",
                mode="lines+markers",
            )
        )
        fig.update_layout(
            title="Anomalies Over Time",
            yaxis=dict(title="Detected anomalies"),
            yaxis2=dict(title="Avg score", overlaying="y", side="right", range=[0, 100], showgrid=False),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sev = (
            f["Severity"]
            .value_counts()
            .reindex(["NORMAL", "MODERATE", "HIGH", "CRITICAL"])
            .fillna(0)
            .reset_index()
        )
        sev.columns = ["Severity", "Count"]
        st.plotly_chart(
            px.bar(
                sev,
                x="Severity",
                y="Count",
                color="Severity",
                color_discrete_map=SEVERITY_COLORS,
                title="Severity Distribution",
            ),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        top = (
            detected.groupby(["SKU_ID", "Product_Name"], as_index=False)
            .agg(Max_Score=("Anomaly_Score", "max"), Anomaly_Count=("Transaction_ID", "count"))
            .nlargest(15, "Max_Score")
            .sort_values("Max_Score")
        )
        st.plotly_chart(
            px.bar(
                top,
                x="Max_Score",
                y="Product_Name",
                orientation="h",
                color="Anomaly_Count",
                hover_data=["SKU_ID", "Anomaly_Count"],
                title="Top Anomalous SKUs",
                range_x=[0, 100],
            ),
            use_container_width=True,
        )
    with c4:
        cat = (
            detected.groupby("Product_Category", as_index=False)
            .agg(Anomalies=("Transaction_ID", "count"), Avg_Score=("Anomaly_Score", "mean"))
            .sort_values("Anomalies")
        )
        st.plotly_chart(
            px.bar(
                cat,
                x="Anomalies",
                y="Product_Category",
                orientation="h",
                color="Avg_Score",
                title="Category-wise Anomalies",
            ),
            use_container_width=True,
        )

    c5, c6 = st.columns(2)
    with c5:
        wh = (
            f.groupby("Warehouse_Name", as_index=False)
            .agg(
                Detected=("Detected_Anomaly", "sum"),
                Average_Score=("Anomaly_Score", "mean"),
                Value_At_Risk=("Inventory_Value", lambda s: float(s.sum())),
            )
        )
        st.plotly_chart(
            px.bar(
                wh,
                x="Warehouse_Name",
                y="Detected",
                color="Average_Score",
                title="Warehouse Anomaly Comparison",
            ),
            use_container_width=True,
        )
    with c6:
        supplier = (
            f.groupby("Supplier_Name", as_index=False)
            .agg(
                Avg_Anomaly_Score=("Anomaly_Score", "mean"),
                Lead_Time_Deviation=("Supplier_Lead_Time_Deviation", "mean"),
                Short_Delivery=("Short_Delivery_Ratio", "mean"),
            )
            .sort_values("Avg_Anomaly_Score", ascending=False)
            .head(18)
        )
        st.plotly_chart(
            px.scatter(
                supplier,
                x="Lead_Time_Deviation",
                y="Avg_Anomaly_Score",
                size=np.clip(supplier["Short_Delivery"], 0.01, None),
                hover_name="Supplier_Name",
                color="Short_Delivery",
                title="Supplier Delivery Anomalies",
            ),
            use_container_width=True,
        )

    c7, c8 = st.columns(2)
    with c7:
        typ = (
            detected["Anomaly_Type"]
            .value_counts()
            .reset_index()
        )
        typ.columns = ["Anomaly_Type", "Count"]
        st.plotly_chart(
            px.bar(
                typ.sort_values("Count"),
                x="Count",
                y="Anomaly_Type",
                orientation="h",
                title="Detected Anomaly Types",
            ),
            use_container_width=True,
        )
    with c8:
        category = (
            detected["Anomaly_Category"]
            .value_counts()
            .reset_index()
        )
        category.columns = ["Anomaly_Category", "Count"]
        st.plotly_chart(
            px.bar(
                category,
                x="Anomaly_Category",
                y="Count",
                color="Anomaly_Category",
                color_discrete_map=CATEGORY_COLORS,
                title="Anomaly Taxonomy",
            ),
            use_container_width=True,
        )

    # Stock vs expected trend for the currently most anomalous SKU-location in scope.
    if not detected.empty:
        focus = detected.nlargest(1, "Anomaly_Score").iloc[0]
        hist = df[
            (df["SKU_ID"] == focus["SKU_ID"])
            & (df["Warehouse_ID"] == focus["Warehouse_ID"])
            & (pd.to_datetime(df["Date"]) <= pd.to_datetime(focus["Date"]))
            & (pd.to_datetime(df["Date"]) >= pd.to_datetime(focus["Date"]) - pd.Timedelta(days=45))
        ].copy()

        st.markdown("### Stock vs Expected Stock Trend")
        st.caption(
            f"Auto-focus: {focus['Product_Name']} · {focus['Warehouse_Name']} · "
            f"highest filtered anomaly score {focus['Anomaly_Score']:.1f}"
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Observed_Closing_Stock"], name="Observed Stock"))
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Expected_Closing_Stock"], name="Expected Stock"))
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Daily_Sales"], name="Sales", opacity=0.55))
        fig.update_layout(title="Observed vs Expected Inventory Behavior")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Anomaly Alert Queue")
    queue_cols = [
        "Date", "Transaction_ID", "SKU_ID", "Product_Name", "Product_Category",
        "Warehouse_Name", "Supplier_Name", "Batch_ID", "Anomaly_Type",
        "Anomaly_Category", "Anomaly_Score", "Severity", "Stock_Status_Anomaly",
        "Inventory_Value", "Anomaly_Reason", "Business_Impact", "Recommended_Action",
    ]
    queue = (
        f[f["Anomaly_Score"] > 30]
        .sort_values(["Anomaly_Score", "Inventory_Value"], ascending=[False, False])[queue_cols]
    )
    if queue.empty:
        st.success("No model-detected anomalies match the current filters.")
    else:
        st.dataframe(
            queue,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Anomaly_Score": st.column_config.ProgressColumn(
                    "Risk Score", min_value=0, max_value=100, format="%d"
                ),
                "Inventory_Value": st.column_config.NumberColumn("Inventory Value", format="PKR %.0f"),
            },
        )
        st.download_button(
            "Download Filtered Anomaly Report",
            queue.to_csv(index=False).encode("utf-8"),
            "inventory_anomaly_report.csv",
            "text/csv",
            use_container_width=True,
        )


def investigation(df: pd.DataFrame, meta: dict):
    title(
        "Anomaly Investigation & Explainability",
        "Open a detected anomaly, inspect context against normal SKU behavior, understand impact and decide corrective action.",
    )

    model = _model_selector(meta, "anomaly_investigation_model")
    f = anomaly_filters(df, model, "anomaly_investigation")
    f = f[f["Anomaly_Score"] > 30].copy()
    stop_if_empty(f)

    f = f.sort_values("Anomaly_Score", ascending=False).head(1000)
    labels = (
        f["Transaction_ID"].astype(str)
        + " | " + f["SKU_ID"].astype(str)
        + " | " + pd.to_datetime(f["Date"]).dt.strftime("%Y-%m-%d")
        + " | " + f["Anomaly_Type"].astype(str)
        + " | " + f["Severity"].astype(str)
    )
    selected_label = st.selectbox("Select anomaly to investigate", labels.tolist())
    row = f.loc[labels[labels == selected_label].index[0]]

    metrics([
        ("Anomaly Score", f"{row['Anomaly_Score']:.1f}/100"),
        ("Severity", row["Severity"]),
        ("Type", row["Anomaly_Type"]),
        ("Observed Stock", f"{row['Observed_Closing_Stock']:,.0f}"),
        ("Expected Stock", f"{row['Expected_Closing_Stock']:,.0f}"),
        ("Inventory Value", f"PKR {max(row['Inventory_Value'],0):,.0f}"),
    ])

    a, b = st.columns([1, 1])
    with a:
        st.markdown("### Investigation Context")
        st.write(f"**Product:** {row['Product_Name']} ({row['SKU_ID']})")
        st.write(f"**Category / Brand:** {row['Product_Category']} / {row['Brand']}")
        st.write(f"**Batch:** {row['Batch_ID']}")
        st.write(f"**Warehouse:** {row['Warehouse_Name']}")
        st.write(f"**Supplier:** {row['Supplier_Name']}")
        st.write(f"**Date:** {pd.to_datetime(row['Date']).date()}")
        st.write(f"**Stock status:** {row['Stock_Status_Anomaly']}")
        st.write(f"**Days to expiry:** {row['Days_To_Expiry']:.0f}")

    with b:
        st.markdown("### Explainability")
        st.warning(f"**Possible reason:** {row['Anomaly_Reason']}")
        st.error(f"**Business impact:** {row['Business_Impact']}")
        st.success(f"**Recommended action:** {row['Recommended_Action']}")
        st.caption(f"Model used: {model} · Model risk score: {row['Anomaly_Score']:.1f}")

    focus_date = pd.to_datetime(row["Date"])
    hist = df[
        (df["SKU_ID"] == row["SKU_ID"])
        & (df["Warehouse_ID"] == row["Warehouse_ID"])
        & (pd.to_datetime(df["Date"]) >= focus_date - pd.Timedelta(days=60))
        & (pd.to_datetime(df["Date"]) <= focus_date + pd.Timedelta(days=7))
    ].copy()
    hist["Selected_Model_Score"] = hist[score_column(model)]

    st.markdown("### Historical Comparison with Normal Behavior")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Observed_Closing_Stock"], name="Observed Stock"))
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Expected_Closing_Stock"], name="Expected Stock"))
    fig.add_vline(x=focus_date, line_dash="dash", line_color="#EF4444")
    fig.update_layout(title="Actual vs Expected Stock")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hist["Date"], y=hist["Daily_Sales"], name="Daily Sales"))
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Rolling_Sales_Mean_7"], name="7D Normal Sales"))
        fig.add_vline(x=focus_date, line_dash="dash", line_color="#EF4444")
        fig.update_layout(title="Sales vs SKU Normal Behavior")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Selected_Model_Score"], name="Anomaly Score"))
        fig.add_hline(y=30, line_dash="dot", line_color="#F59E0B")
        fig.add_hline(y=60, line_dash="dot", line_color="#F97316")
        fig.add_hline(y=80, line_dash="dot", line_color="#EF4444")
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(title="Model Risk Score History")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature Signals for Selected Record")
    signal_data = pd.DataFrame({
        "Signal": [
            "Absolute Stock Change %",
            "Stock Mismatch Ratio",
            "Adjustment Ratio",
            "Return %",
            "Supplier Lead-Time Deviation",
            "Short Delivery %",
            "Warehouse Capacity %",
            "Days-to-Expiry Pressure",
        ],
        "Value": [
            min(abs(float(row["Pct_Stock_Change"])) * 100, 100),
            min(abs(float(row["Stock_Mismatch_Qty"])) / max(abs(float(row["Opening_Stock"])), 1) * 100, 100),
            min(float(row["Adjustment_Ratio"]) * 100, 100),
            min(float(row["Return_Percentage"]) * 100, 100),
            min(float(row["Supplier_Lead_Time_Deviation"]) / 14 * 100, 100),
            min(float(row["Short_Delivery_Ratio"]) * 100, 100),
            min(float(row["Warehouse_Capacity_Utilization"]) * 100, 100),
            max(0, min((30 - float(row["Days_To_Expiry"])) / 30 * 100, 100)),
        ],
    })
    st.plotly_chart(
        px.bar(
            signal_data.sort_values("Value"),
            x="Value",
            y="Signal",
            orientation="h",
            title="Normalized Business Signal Strength",
            range_x=[0, 100],
        ),
        use_container_width=True,
    )


def model_lab(df: pd.DataFrame, meta: dict, model_metrics: dict):
    title(
        "Anomaly Model Comparison Lab",
        "Compare Isolation Forest, LOF, One-Class SVM, Autoencoder, statistical rules and the recommended hybrid using synthetic anomaly labels.",
    )

    selected_model = _model_selector(meta, "anomaly_lab_model")
    f = anomaly_filters(df, selected_model, "anomaly_lab")
    stop_if_empty(f)

    # Recalculate selected-model metrics for current filtered scope.
    y = f["Synthetic_Anomaly_Label"].astype(int).to_numpy()
    pred = (f["Anomaly_Score"].to_numpy() > 30).astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(fp + tn, 1)

    metrics([
        ("Filtered Rows", f"{len(f):,}"),
        ("Detected", f"{int(pred.sum()):,}"),
        ("Precision", f"{precision:.1%}"),
        ("Recall", f"{recall:.1%}"),
        ("F1", f"{f1:.1%}"),
        ("False Positive Rate", f"{fpr:.1%}"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Normal", "Actual Anomaly"],
            columns=["Pred Normal", "Pred Anomaly"],
        )
        st.plotly_chart(
            px.imshow(cm_df, text_auto=True, title=f"{selected_model} — Confusion Matrix"),
            use_container_width=True,
        )
    with c2:
        score_dist = f.copy()
        score_dist["Ground_Truth"] = np.where(
            score_dist["Synthetic_Anomaly_Label"] == 1, "Injected Anomaly", "Normal"
        )
        st.plotly_chart(
            px.histogram(
                score_dist,
                x="Anomaly_Score",
                color="Ground_Truth",
                nbins=35,
                barmode="overlay",
                opacity=0.65,
                title="Risk Score Separation",
            ),
            use_container_width=True,
        )

    st.markdown("### Global Model Comparison")
    rows = []
    for model_name in meta["models"]:
        m = model_metrics[model_name]
        rows.append({
            "Model": model_name,
            "Detected Anomalies": m["Detected_Anomalies"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1 Score": m["F1"],
            "False Positive Rate": m["False_Positive_Rate"],
            "Execution Time (s)": m["Execution_Time_Seconds"],
        })
    comp = pd.DataFrame(rows)
    st.dataframe(comp, use_container_width=True, hide_index=True)

    long = comp.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score",
    )
    st.plotly_chart(
        px.bar(
            long,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Precision / Recall / F1 Comparison",
        ),
        use_container_width=True,
    )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            px.bar(
                comp.sort_values("Detected Anomalies"),
                x="Detected Anomalies",
                y="Model",
                orientation="h",
                title="Detected Anomalies by Model",
            ),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            px.bar(
                comp.sort_values("Execution Time (s)"),
                x="Execution Time (s)",
                y="Model",
                orientation="h",
                title="Execution Time by Model",
            ),
            use_container_width=True,
        )

    # Warehouse/category heatmap changes with selected model.
    heat = (
        f.groupby(["Warehouse_Name", "Product_Category"], as_index=False)["Anomaly_Score"]
        .mean()
    )
    pivot = heat.pivot(index="Warehouse_Name", columns="Product_Category", values="Anomaly_Score").fillna(0)
    st.plotly_chart(
        px.imshow(
            pivot,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="Plasma",
            title=f"{selected_model} — Average Risk Heatmap",
            zmin=0,
            zmax=100,
        ),
        use_container_width=True,
    )


def data_explorer(df: pd.DataFrame, meta: dict):
    title(
        "Anomaly Data Explorer",
        "Audit synthetic anomaly injection, transaction fields, engineered behavior signals and model-ready inventory records.",
    )

    model = _model_selector(meta, "anomaly_data_model")
    f = anomaly_filters(df, model, "anomaly_data")
    stop_if_empty(f)

    metrics([
        ("Rows", f"{len(f):,}"),
        ("SKUs", f"{f['SKU_ID'].nunique():,}"),
        ("Warehouses", f"{f['Warehouse_ID'].nunique():,}"),
        ("Suppliers", f"{f['Supplier_ID'].nunique():,}"),
        ("Batches", f"{f['Batch_ID'].nunique():,}"),
        ("Injected Anomaly Rate", f"{f['Synthetic_Anomaly_Label'].mean():.1%}"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        truth = f["Injected_Anomaly_Type"].value_counts().reset_index()
        truth.columns = ["Injected_Anomaly_Type", "Rows"]
        st.plotly_chart(
            px.bar(
                truth.head(15).sort_values("Rows"),
                x="Rows",
                y="Injected_Anomaly_Type",
                orientation="h",
                title="Controlled Synthetic Anomaly Injection",
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.scatter(
                f.sample(min(3500, len(f)), random_state=42),
                x="Expected_Closing_Stock",
                y="Observed_Closing_Stock",
                color="Severity",
                color_discrete_map=SEVERITY_COLORS,
                hover_data=["SKU_ID", "Warehouse_Name", "Anomaly_Type"],
                title="Observed vs Expected Stock",
            ),
            use_container_width=True,
        )

    st.markdown("### Transaction / Inventory Sample")
    cols = [
        "Date", "Transaction_ID", "SKU_ID", "Product_Name", "Product_Category",
        "Brand", "Warehouse_Name", "Supplier_Name", "Batch_ID", "Opening_Stock",
        "Purchases", "Daily_Sales", "Returns", "Stock_Adjustment", "Transfers_In",
        "Transfers_Out", "Expected_Closing_Stock", "Observed_Closing_Stock",
        "Stock_Mismatch_Qty", "Stock_Age_Days", "Days_To_Expiry", "Anomaly_Type",
        "Anomaly_Score", "Severity", "Synthetic_Anomaly_Label",
    ]
    st.dataframe(f[cols].head(1000), use_container_width=True, hide_index=True)

    st.download_button(
        "Download Filtered Anomaly Dataset",
        f.to_csv(index=False).encode("utf-8"),
        "synthetic_inventory_anomaly_data.csv",
        "text/csv",
        use_container_width=True,
    )
