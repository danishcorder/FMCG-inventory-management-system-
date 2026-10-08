from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd
import streamlit as st

RISK_COLOR = {
    "LOW": "#10B981",
    "MEDIUM": "#F59E0B",
    "HIGH": "#F97316",
    "CRITICAL": "#EF4444",
}

STOCK_STATUS_COLOR = {
    "OUT OF STOCK": "#EF4444",
    "CRITICAL STOCK": "#F97316",
    "BELOW REORDER POINT": "#F59E0B",
    "HEALTHY": "#10B981",
    "POSSIBLE OVERSTOCK": "#60A5FA",
}

def title(text: str, caption: str = "", eyebrow: str = "UNICORN ENTERPRISE UI"):
    st.markdown(
        f"""
        <div class="unicorn-page-header">
          <div class="eyebrow">{eyebrow}</div>
          <div class="headline">{text}</div>
          <div class="subline">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def metrics(items):
    cols = st.columns(len(items))
    for c, item in zip(cols, items):
        if len(item) == 2:
            label, value = item
            delta = None
        else:
            label, value, delta = item
        c.metric(label, value, delta)

def _safe_unique(df: pd.DataFrame, col: str) -> list:
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().astype(str).unique().tolist())

def business_filters(
    df: pd.DataFrame,
    key: str,
    *,
    show_date_range: bool = False,
    show_sku: bool = True,
    show_stock_status: bool = False,
    show_risk: bool = False,
    show_action: bool = False,
    show_projected_risk: bool = False,
    show_overstock: bool = False,
):
    out = df.copy()
    with st.sidebar:
        st.markdown("### Filters")
        st.markdown(
            '<div class="unicorn-filter-caption">Filters are cumulative. Every KPI, chart and table on this page uses the filtered scope.</div>',
            unsafe_allow_html=True
        )

        if show_date_range and "Date" in out.columns:
            d = pd.to_datetime(out["Date"])
            min_d, max_d = d.min().date(), d.max().date()
            selected = st.date_input(
                "Date Range",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
                key=f"{key}_date_range",
            )
            if isinstance(selected, (list, tuple)) and len(selected) == 2:
                start, end = selected
                out = out[(pd.to_datetime(out["Date"]).dt.date >= start) & (pd.to_datetime(out["Date"]).dt.date <= end)]

        wh = st.selectbox("Warehouse", ["All"] + _safe_unique(out, "Warehouse_Name"), key=f"{key}_wh")
        cat = st.selectbox("Product Category", ["All"] + _safe_unique(out, "Product_Category"), key=f"{key}_cat")

        if show_sku:
            sku_options = ["All"] + _safe_unique(out, "SKU_ID")
            sku = st.selectbox("SKU", sku_options, key=f"{key}_sku")
        else:
            sku = "All"

        sup = st.selectbox("Supplier", ["All"] + _safe_unique(out, "Supplier_Name"), key=f"{key}_sup")
        mover = st.selectbox("Mover Class", ["All"] + _safe_unique(out, "Fast_Slow_Mover"), key=f"{key}_move")
        promo = st.selectbox("Promotion", ["All", "Active", "No Promotion"], key=f"{key}_promo")

        stock_status = "All"
        if show_stock_status and "Stock_Status" in out.columns:
            stock_status = st.selectbox(
                "Stock Status", ["All"] + _safe_unique(out, "Stock_Status"), key=f"{key}_stock_status"
            )

        risk = "All"
        if show_risk and "Risk_Level" in out.columns:
            risk = st.selectbox(
                "Risk Level", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"], key=f"{key}_risk"
            )

        action = "All"
        if show_action and "Recommended_Action" in out.columns:
            action = st.selectbox(
                "Recommended Action", ["All"] + _safe_unique(out, "Recommended_Action"), key=f"{key}_action"
            )

        projected = "All"
        if show_projected_risk and "Projected_Stockout_Risk" in out.columns:
            projected = st.selectbox(
                "Projected Stockout", ["All", "At Risk", "Not At Risk"], key=f"{key}_projected"
            )

        overstock = "All"
        if show_overstock and "Overstock_Risk" in out.columns:
            overstock = st.selectbox(
                "Overstock", ["All", "Overstock Candidate", "Normal"], key=f"{key}_overstock"
            )

    if wh != "All":
        out = out[out["Warehouse_Name"].astype(str) == wh]
    if cat != "All":
        out = out[out["Product_Category"].astype(str) == cat]
    if sku != "All":
        out = out[out["SKU_ID"].astype(str) == sku]
    if sup != "All":
        out = out[out["Supplier_Name"].astype(str) == sup]
    if mover != "All":
        out = out[out["Fast_Slow_Mover"].astype(str) == mover]
    if promo == "Active":
        out = out[out["Promotion_Flag"] == 1]
    elif promo == "No Promotion":
        out = out[out["Promotion_Flag"] == 0]
    if stock_status != "All":
        out = out[out["Stock_Status"].astype(str) == stock_status]
    if risk != "All":
        out = out[out["Risk_Level"].astype(str) == risk]
    if action != "All":
        out = out[out["Recommended_Action"].astype(str) == action]
    if projected == "At Risk":
        out = out[out["Projected_Stockout_Risk"] == True]
    elif projected == "Not At Risk":
        out = out[out["Projected_Stockout_Risk"] == False]
    if overstock == "Overstock Candidate":
        out = out[out["Overstock_Risk"] == True]
    elif overstock == "Normal":
        out = out[out["Overstock_Risk"] == False]
    return out


def apply_business_filter_state(
    df: pd.DataFrame,
    key: str,
    *,
    show_stock_status: bool = False,
    show_risk: bool = False,
    show_action: bool = False,
    show_projected_risk: bool = False,
    show_overstock: bool = False,
):
    """Apply already-rendered sidebar filter values to another dataframe without creating widgets."""
    out = df.copy()

    wh = st.session_state.get(f"{key}_wh", "All")
    cat = st.session_state.get(f"{key}_cat", "All")
    sku = st.session_state.get(f"{key}_sku", "All")
    sup = st.session_state.get(f"{key}_sup", "All")
    mover = st.session_state.get(f"{key}_move", "All")
    promo = st.session_state.get(f"{key}_promo", "All")

    if wh != "All" and "Warehouse_Name" in out:
        out = out[out["Warehouse_Name"].astype(str) == wh]
    if cat != "All" and "Product_Category" in out:
        out = out[out["Product_Category"].astype(str) == cat]
    if sku != "All" and "SKU_ID" in out:
        out = out[out["SKU_ID"].astype(str) == sku]
    if sup != "All" and "Supplier_Name" in out:
        out = out[out["Supplier_Name"].astype(str) == sup]
    if mover != "All" and "Fast_Slow_Mover" in out:
        out = out[out["Fast_Slow_Mover"].astype(str) == mover]
    if promo == "Active" and "Promotion_Flag" in out:
        out = out[out["Promotion_Flag"] == 1]
    elif promo == "No Promotion" and "Promotion_Flag" in out:
        out = out[out["Promotion_Flag"] == 0]

    if show_stock_status and "Stock_Status" in out:
        status = st.session_state.get(f"{key}_stock_status", "All")
        if status != "All":
            out = out[out["Stock_Status"].astype(str) == status]

    if show_risk and "Risk_Level" in out:
        risk = st.session_state.get(f"{key}_risk", "All")
        if risk != "All":
            out = out[out["Risk_Level"].astype(str) == risk]

    if show_action and "Recommended_Action" in out:
        action = st.session_state.get(f"{key}_action", "All")
        if action != "All":
            out = out[out["Recommended_Action"].astype(str) == action]

    if show_projected_risk and "Projected_Stockout_Risk" in out:
        projected = st.session_state.get(f"{key}_projected", "All")
        if projected == "At Risk":
            out = out[out["Projected_Stockout_Risk"] == True]
        elif projected == "Not At Risk":
            out = out[out["Projected_Stockout_Risk"] == False]

    if show_overstock and "Overstock_Risk" in out:
        overstock = st.session_state.get(f"{key}_overstock", "All")
        if overstock == "Overstock Candidate":
            out = out[out["Overstock_Risk"] == True]
        elif overstock == "Normal":
            out = out[out["Overstock_Risk"] == False]

    return out

def stop_if_empty(df: pd.DataFrame):
    if df.empty:
        st.warning("No records match the selected filters. Broaden one or more filters.")
        st.stop()

def model_filter(label: str, options: list[str], key: str, help_text: str | None = None) -> str:
    with st.sidebar:
        st.markdown("### ML Controls")
        return st.selectbox(label, options, key=key, help=help_text)

def horizon_filter(key: str, options=(7, 14, 30)) -> int:
    with st.sidebar:
        return st.selectbox("Planning Horizon", list(options), key=key)


def date_range_filter(df: pd.DataFrame, key: str, default_days: int = 30):
    dates = pd.to_datetime(df["Date"])
    min_d, max_d = dates.min().date(), dates.max().date()
    default_start = max(min_d, (pd.Timestamp(max_d) - pd.Timedelta(days=default_days - 1)).date())
    with st.sidebar:
        selected = st.date_input(
            "Date Range",
            value=(default_start, max_d),
            min_value=min_d,
            max_value=max_d,
            key=key,
            help="The range end becomes the dashboard as-of date; the full range is used for trend analysis.",
        )
    if isinstance(selected, (list, tuple)) and len(selected) == 2:
        return selected[0], selected[1]
    return default_start, max_d

def as_of_filter(df: pd.DataFrame, key: str, recent_days: int | None = None):
    dates = sorted(pd.to_datetime(df["Date"]).dt.date.unique())
    if recent_days:
        dates = dates[-recent_days:]
    with st.sidebar:
        return st.selectbox("As-of Date", dates, index=len(dates)-1, key=key)
