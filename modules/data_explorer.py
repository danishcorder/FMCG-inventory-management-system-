from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px

from modules.common import title, metrics, business_filters, stop_if_empty


def render(df: pd.DataFrame):
    title(
        "Data Explorer",
        "Inspect the synthetic FMCG data foundation, apply business/date filters and audit class balance and data quality.",
    )

    filtered = business_filters(df, "data_explorer", show_date_range=True, show_sku=True)
    stop_if_empty(filtered)

    metrics([
        ("Rows", f"{len(filtered):,}"),
        ("Columns", f"{filtered.shape[1]:,}"),
        ("SKUs", f"{filtered['SKU_ID'].nunique():,}"),
        ("Warehouses", f"{filtered['Warehouse_ID'].nunique():,}"),
        ("Suppliers", f"{filtered['Supplier_ID'].nunique():,}"),
        ("Stockout Rate", f"{filtered['Stockout_Next_7D'].mean():.1%}"),
    ])

    st.caption(
        f"Filtered date range: {pd.to_datetime(filtered['Date']).min().date()} → "
        f"{pd.to_datetime(filtered['Date']).max().date()}"
    )

    c1, c2 = st.columns(2)
    with c1:
        d = filtered["Stockout_Next_7D"].value_counts().rename_axis("Target").reset_index(name="Rows")
        d["Target"] = d["Target"].map({0: "No Stockout", 1: "Stockout"})
        st.plotly_chart(
            px.bar(d, x="Target", y="Rows", color="Target", title="7-Day Stockout Target Distribution"),
            use_container_width=True,
        )
    with c2:
        d = (
            filtered.groupby("Product_Category", as_index=False)["Demand_Qty"]
            .mean()
            .sort_values("Demand_Qty", ascending=False)
        )
        st.plotly_chart(
            px.bar(d, x="Product_Category", y="Demand_Qty", title="Average Daily Demand by Category"),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        daily = filtered.groupby("Date", as_index=False)["Demand_Qty"].sum().tail(120)
        st.plotly_chart(
            px.line(daily, x="Date", y="Demand_Qty", title="Aggregate Demand Trend"),
            use_container_width=True,
        )
    with c4:
        wh = filtered.groupby("Warehouse_Name", as_index=False)["Available_Stock"].sum()
        st.plotly_chart(
            px.bar(wh, x="Warehouse_Name", y="Available_Stock", title="Available Stock by Warehouse"),
            use_container_width=True,
        )

    st.markdown("### Sample Data")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)

    miss = filtered.isna().sum().reset_index()
    miss.columns = ["Column", "Missing"]
    st.markdown("### Data Quality — Missing Values")
    if (miss["Missing"] > 0).any():
        st.dataframe(miss[miss["Missing"] > 0], use_container_width=True, hide_index=True)
    else:
        st.success("No missing values in the filtered dataset.")

    st.download_button(
        "Download Filtered Synthetic Dataset",
        filtered.to_csv(index=False).encode("utf-8"),
        "synthetic_fmcg_inventory_filtered.csv",
        "text/csv",
        use_container_width=True,
    )
