from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="FMCG Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.theme import apply_unicorn_enterprise_theme
apply_unicorn_enterprise_theme()

from core.init_project import ensure_initialized
from core.anomaly_modeling import (
    load_anomaly_data,
    load_anomaly_metrics,
    load_anomaly_meta,
)
from core.storage import load_data, load_metrics, load_meta
from core.auth import authenticate
from modules import stockout, demand, anomaly, data_explorer


@st.cache_resource(show_spinner=True)
def initialize():
    ensure_initialized(force=False)
    return True


@st.cache_data(show_spinner=False)
def data():
    return load_data()


@st.cache_data(show_spinner=False)
def metrics_data():
    return load_metrics()


@st.cache_data(show_spinner=False)
def meta_data():
    return load_meta()

@st.cache_data(show_spinner=False)
def anomaly_data():
    return load_anomaly_data()


@st.cache_data(show_spinner=False)
def anomaly_metrics_data():
    return load_anomaly_metrics()


@st.cache_data(show_spinner=False)
def anomaly_meta_data():
    return load_anomaly_meta()


def login():
    st.markdown(
        """
        <div class="unicorn-page-header">
          <div class="eyebrow">UNICORN ENTERPRISE UI</div>
          <div class="headline">FMCG Inventory Intelligence Platform</div>
          <div class="subline">Connected machine-learning decision support for stockout risk, demand planning, replenishment and inventory anomaly detection.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b = st.columns([1.25, 1])
    with a:
        st.markdown("### Enterprise Intelligence Modules")
        st.markdown(
            """
            **Stockout Risk Prediction**  
            Predict SKU-location stockout risk for the next 7 days, compare ML models, explain risk drivers and prioritize corrective action.

            **Demand Prediction & Replenishment**  
            Forecast SKU demand and convert the forecast into dynamic safety stock, reorder points and MOQ-aware purchase recommendations.

            **Inventory Anomaly Detection**  
            Detect unusual stock movements, mismatch, expiry risk, suspicious returns/adjustments, warehouse issues and supplier anomalies.

            **Shared data foundation**  
            All three modules use connected product, warehouse, supplier, inventory, sales and business context.
            """
        )
        st.info("Version 3 connects three ML systems under the same Unicorn Enterprise UI and shared FMCG data foundation.")
    with b:
        with st.form("login"):
            st.subheader("Secure Demo Login")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            if st.form_submit_button("Sign in", use_container_width=True):
                user = authenticate(username, password)
                if user:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        st.caption("Demo platform login: platform_admin / admin123 · anomaly_admin / admin123")


def main():
    initialize()
    df = data()
    all_metrics = metrics_data()
    meta = meta_data()

    if "user" not in st.session_state:
        login()
        return

    user = st.session_state["user"]

    st.sidebar.markdown('<span class="unicorn-badge">UNICORN ENTERPRISE</span>', unsafe_allow_html=True)
    st.sidebar.title("FMCG Intelligence")
    st.sidebar.caption(f"Signed in as {user['username']}")

    modules = []
    if "stockout" in user["modules"]:
        modules.append("Stockout Risk Prediction")
    if "demand" in user["modules"]:
        modules.append("Demand Prediction & Replenishment")
    if "anomaly" in user["modules"]:
        modules.append("Inventory Anomaly Detection")

    module = st.sidebar.radio("Module", modules)

    if module == "Stockout Risk Prediction":
        page = st.sidebar.radio(
            "Workspace",
            ["Command Center", "Full Stock Report", "SKU Analysis", "ML Model Lab", "Data Explorer", "About"],
        )
        if page == "Command Center":
            stockout.dashboard(df, meta, all_metrics)
        elif page == "Full Stock Report":
            stockout.stock_report(df, meta)
        elif page == "SKU Analysis":
            stockout.sku_analysis(df, meta)
        elif page == "ML Model Lab":
            stockout.model_lab(df, meta, all_metrics)
        elif page == "Data Explorer":
            data_explorer.render(df)
        else:
            st.title("About — Stockout Risk Prediction")
            st.write(
                "Predicts the probability that an SKU-location will stock out in the next 7 days using inventory position, "
                "demand velocity, supplier lead time/reliability, promotions, calendar signals and engineered coverage features. "
                "Predictions are decision support, not automatic purchase orders."
            )
    elif module == "Demand Prediction & Replenishment":
        page = st.sidebar.radio(
            "Workspace",
            ["Demand Dashboard", "Replenishment Planner", "SKU Demand Forecast", "Demand Model Lab",
             "Supplier Performance", "Data Explorer", "About"],
        )
        if page == "Demand Dashboard":
            demand.dashboard(df, meta, all_metrics)
        elif page == "Replenishment Planner":
            demand.planner(df, meta)
        elif page == "SKU Demand Forecast":
            demand.sku_forecast(df, meta)
        elif page == "Demand Model Lab":
            demand.model_lab(df, meta, all_metrics)
        elif page == "Supplier Performance":
            demand.supplier_performance(df)
        elif page == "Data Explorer":
            data_explorer.render(df)
        else:
            st.title("About — Demand Prediction & Replenishment")
            st.write(
                "Forecasts SKU-level demand and converts it into replenishment decisions using lead-time demand, dynamic safety stock, "
                "reorder point, incoming stock, supplier lead time and MOQ-aware order quantities."
            )
    else:
        adf = anomaly_data()
        ametrics = anomaly_metrics_data()
        ameta = anomaly_meta_data()
        page = st.sidebar.radio(
            "Workspace",
            ["Anomaly Dashboard", "Investigation", "Model Comparison", "Anomaly Data Explorer", "About"],
        )
        if page == "Anomaly Dashboard":
            anomaly.dashboard(adf, ameta, ametrics)
        elif page == "Investigation":
            anomaly.investigation(adf, ameta)
        elif page == "Model Comparison":
            anomaly.model_lab(adf, ameta, ametrics)
        elif page == "Anomaly Data Explorer":
            anomaly.data_explorer(adf, ameta)
        else:
            st.title("About — Inventory Anomaly Detection")
            st.write(
                "Uses unsupervised ML plus transparent business rules to detect abnormal FMCG inventory behavior. "
                "It scores records from 0–100, assigns severity, identifies anomaly type, explains business impact, "
                "and recommends an investigation or corrective action."
            )

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.pop("user", None)
        st.rerun()


if __name__ == "__main__":
    main()
