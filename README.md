---
title: FMCG Inventory Intelligence
emoji: 📦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.0.0"
app_file: app.py
pinned: false
---



# FMCG Inventory Intelligence Platform — Version 3

A connected, synthetic-first FMCG Machine Learning platform with three separate business dashboards under one shared Unicorn Enterprise UI.

## Implemented Modules

### 1. Stockout Risk Prediction
Predicts 7-day SKU × warehouse stockout risk.

Main capabilities:
- Logistic Regression, Random Forest, HistGradientBoosting, optional CatBoost
- Model selector that changes probabilities, KPIs, rankings and charts
- Risk levels: Low / Medium / High / Critical
- Full Stock Report
- SKU risk investigation
- stock coverage vs lead time
- supplier exposure
- recommended action and decision-support order quantity
- Stockout ML Model Lab

### 2. Demand Prediction & Replenishment
Forecasts SKU demand and converts forecasts into purchase decisions.

Main capabilities:
- Moving Average baseline
- Random Forest Regressor
- HistGradientBoostingRegressor
- optional LightGBM
- Holt-Winters SKU baseline
- 7 / 14 / 30 day planning views
- dynamic safety stock
- dynamic reorder point
- MOQ-aware recommended order quantity
- ORDER NOW / ORDER WITHIN 3 DAYS / STOCKOUT RISK / OVERSTOCK RISK alerts
- supplier performance
- Demand ML Model Lab

### 3. Inventory Anomaly Detection
Detects abnormal FMCG inventory behavior and explains what a manager should investigate.

Models:
- **Hybrid: Isolation Forest + Rules**
- Isolation Forest
- Local Outlier Factor
- One-Class SVM
- Autoencoder
- Statistical Rules

The hybrid is the recommended default because it combines unsupervised pattern detection with transparent inventory-control logic.

Anomaly types include:
- stock loss / shrinkage
- stock mismatch
- negative inventory
- abnormal purchase size
- unusual sales day
- suspicious return
- duplicate transaction
- late adjustment
- slow/dead stock
- expiry risk
- transfer mismatch
- warehouse capacity pressure
- supplier delay
- short delivery

Anomaly dashboard:
- Inventory Value at Risk
- Detected Anomalies
- Critical SKU-Locations
- Expiry Risk
- Slow / Dead Stock
- Suspicious Adjustments
- anomalies over time
- severity distribution
- top anomalous SKUs
- category-wise anomalies
- warehouse comparison
- observed vs expected stock
- supplier delivery anomalies
- anomaly taxonomy
- alert queue
- CSV report export

Required filters:
- ML model
- date range
- warehouse
- category
- brand
- SKU
- supplier
- batch
- anomaly type
- severity
- stock status
- detected anomalies only

Investigation screen:
- selected anomaly context
- SKU / batch / warehouse / supplier
- observed vs expected stock
- historical normal behavior
- sales vs normal behavior
- model risk score history
- possible reason
- business impact
- recommended action

Model comparison:
- detected anomalies
- precision
- recall
- F1
- false positive rate
- execution time
- confusion matrix
- score separation
- model comparison charts
- warehouse/category risk heatmap

---

## Shared Data Architecture

All three modules share the same business foundation:
- products
- SKUs
- categories / brands / pack sizes
- warehouses
- suppliers
- sales
- inventory
- purchase/receipt behavior
- promotions
- calendar signals

The anomaly module creates a separate enriched transaction view from the same foundation with:
- Transaction_ID
- Batch_ID
- Unit_Cost
- Returns
- Stock_Adjustment
- Transfers_In / Transfers_Out
- Observed vs Expected Closing Stock
- Stock age
- Expiry date
- Damaged / expired quantity
- Supplier delivery deviation
- Warehouse capacity

This keeps the systems connected without duplicating unrelated master data.

---

## Anomaly Score

The anomaly module converts each model output to a 0–100 business risk score.

- **0–30** → NORMAL
- **31–60** → MODERATE
- **61–80** → HIGH
- **81–100** → CRITICAL

Every detected anomaly includes:
- type
- category
- severity
- SKU
- warehouse
- date
- possible reason
- business impact
- recommended action

---

## ML Discipline

Stockout / Demand:
- chronological train / validation / test split
- past-only features
- no random time-series holdout

Anomaly Detection:
- chronological unsupervised training
- models learn mainly from older normal records
- controlled synthetic anomalies are used for evaluation
- injected anomaly labels are **not** used as supervised targets
- model comparison uses Precision / Recall / F1 / False Positive Rate

---

## Synthetic Prototype Scale

Shared FMCG foundation:
- 100 SKUs
- 3 Multan warehouses
- 18 suppliers
- approximately 99,000 daily SKU-location records

Anomaly prototype:
- recent contiguous history from the shared data
- about 48,600 transaction-style records
- 90 SKUs
- 3 warehouses
- 18 suppliers
- controlled synthetic anomalies for validation

Synthetic data is only for prototype, testing and academic demonstration. Replace it with real ERP/POS/warehouse data for industrial deployment.

---

## Demo Logins

| User | Password | Access |
|---|---|---|
| `platform_admin` | `admin123` | All three modules |
| `stockout_admin` | `admin123` | Stockout only |
| `demand_admin` | `admin123` | Demand / Replenishment only |
| `anomaly_admin` | `admin123` | Anomaly Detection only |

These are local academic demo credentials, not production authentication.

---

## Installation

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate on Windows

```bash
.venv\Scripts\activate
```

### 3. Install core requirements

```bash
pip install -r requirements.txt
```

Optional advanced packages for the first two modules:

```bash
pip install -r optional_requirements.txt
```

### 4. Initialize data + all ML models

```bash
python -m core.init_project
```

The first initialization:
1. generates the shared synthetic FMCG dataset,
2. trains Stockout models,
3. trains Demand models,
4. builds the anomaly transaction dataset,
5. injects controlled anomaly scenarios,
6. trains all anomaly models,
7. writes model metrics and artifacts.

### 5. Start the dashboard

```bash
streamlit run app.py
```

Or on Windows:

```bash
run_windows.bat
```

---

## Technology Stack

- Python
- Pandas / NumPy
- scikit-learn
- Streamlit
- Plotly
- SQLite
- Joblib
- Statsmodels
- optional CatBoost
- optional LightGBM

The anomaly Autoencoder uses a compact `MLPRegressor` reconstruction architecture so TensorFlow/PyTorch is not required in Version 1.

---

## Enterprise UI

All modules use the centralized `core/theme.py` **Unicorn Enterprise UI**:
- dark navy SaaS layout
- purple/cyan accents
- semantic risk colors
- professional KPI cards
- consistent filters
- enterprise Plotly theme
- responsive wide layout
- styled tables and exports
- shared module navigation

---

## Project Structure

```text
FMCG_Inventory_Intelligence_v3_Anomaly/
│
├── app.py
├── config.py
├── requirements.txt
├── optional_requirements.txt
├── START_HERE.txt
├── README.md
├── ANOMALY_IMPLEMENTATION_AUDIT.md
│
├── core/
│   ├── data_generator.py
│   ├── features.py
│   ├── modeling.py
│   ├── anomaly_data.py
│   ├── anomaly_modeling.py
│   ├── init_project.py
│   ├── storage.py
│   ├── auth.py
│   └── theme.py
│
├── modules/
│   ├── stockout.py
│   ├── demand.py
│   ├── anomaly.py
│   ├── data_explorer.py
│   └── common.py
│
├── tests/
│   ├── smoke_test.py
│   ├── filter_audit.py
│   └── anomaly_audit.py
│
├── data/
└── models/
```

---

## Audits

Read:
- `IMPLEMENTATION_AUDIT.md`
- `FILTER_DASHBOARD_AUDIT.md`
- `ANOMALY_IMPLEMENTATION_AUDIT.md`

The third module is designed around the business workflow:

**Detect → Score → Explain → Investigate → Act**
