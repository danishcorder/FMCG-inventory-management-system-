# Filter & Dashboard Audit — Version 2 Enterprise

## Result
The dashboard/filter requirements from the implementation brief were re-audited and upgraded.

### Stockout Risk Command Center — PASS
Required controls now present:
- ML Model selector
- Date Range (range end is used as the as-of date)
- Warehouse
- Product Category
- SKU
- Supplier
- Risk Level
- Fast / Medium / Slow mover class
- Promotion status

Additional enterprise filters:
- Current Stock Status
- Recommended Action

Model switching is functional, not decorative:
- stockout probabilities change
- risk classes change
- high/critical counts change
- top-risk ranking changes
- stockout charts change
- model risk trend changes
- model recall/precision context changes
- recommendations can change because they consume the selected model probability

### Full Stock Report — PASS
Controls:
- ML Model
- As-of Date
- Warehouse
- Product Category
- SKU
- Supplier
- Mover Class
- Promotion
- Stock Status
- Risk Level
- Recommended Action

The report combines operational stock state and future ML risk instead of showing only a static inventory table.

### SKU Analysis — PASS
Controls:
- ML Model
- As-of Date
- SKU
- Warehouse

Outputs:
- stockout probability
- risk
- current stock status
- available stock
- coverage
- recommended order quantity
- 90-day inventory/demand history
- explanation and action

### Stockout ML Model Lab — PASS
Controls:
- Model
- Decision threshold
- Date Range
- Warehouse
- Product Category
- SKU
- Supplier
- Mover Class
- Promotion

Outputs recalculate on the filtered evaluation set:
- precision
- recall
- F1
- confusion matrix
- evaluation rows
- forecast/model comparison context

---

## Demand Prediction & Replenishment Dashboard — PASS
Required/important controls:
- Forecast Model
- Date Range
- Planning Horizon: 7 / 14 / 30 days
- Warehouse
- Product Category
- SKU
- Supplier
- Mover Class
- Promotion

Additional decision filters:
- Recommended Action
- Projected Stockout
- Overstock

Changing the forecast model updates:
- forecast demand
- recommended purchase quantity
- projected ending stock
- order-now decisions
- purchase priority
- warehouse demand/purchase chart
- supplier exposure
- forecast-vs-actual validation trend

### Replenishment Planner — PASS
Controls:
- Forecast Model
- Planning Horizon
- As-of Date
- Warehouse
- Product Category
- SKU
- Supplier
- Mover Class
- Promotion
- Recommended Action
- Projected Stockout
- Overstock

### SKU Demand Forecast — PASS
Controls:
- Forecast Model
- As-of Date
- SKU
- Warehouse

Outputs:
- historical demand
- forward forecast
- weekly demand
- day-of-week profile
- Holt-Winters baseline when available

### Demand Model Lab — PASS
Controls:
- Forecast Model
- Date Range
- Warehouse
- Product Category
- SKU
- Supplier
- Mover Class
- Promotion

Metrics are recalculated for the filtered evaluation scope.

### Supplier Performance — PASS
Controls:
- As-of Date
- Warehouse
- Product Category
- SKU
- Supplier
- Mover Class
- Promotion

---

## Shared Data Explorer — PASS
Controls:
- Date Range
- Warehouse
- Product Category
- SKU
- Supplier
- Mover Class
- Promotion

All KPIs, charts, sample rows and CSV export use the filtered dataset.

---

# Enterprise Styling Audit — PASS

Version 2 uses a shared `Unicorn Enterprise UI` theme in `core/theme.py`.

Implemented:
- enterprise dark navy background
- purple/cyan Unicorn accent system
- consistent KPI cards
- styled filter controls
- styled buttons and download actions
- styled data tables
- styled Plotly charts
- unified typography
- professional page headers
- consistent spacing and borders
- sidebar module workspace
- semantic stockout risk colors
- responsive wide layout
- hidden default Streamlit chrome for a cleaner SaaS presentation

The styling is centralized so future Inventory Anomaly Detection pages can reuse the same visual system.

---

# Technical Verification

- Python syntax compilation: PASS
- Synthetic data generator smoke test: PASS
- Stockout model training smoke test: PASS
- Demand model training smoke test: PASS
- Chronological train/validation/test split retained
- Optional CatBoost remains guarded
- Optional LightGBM remains guarded
- Live Streamlit rendering should be visually checked after installing requirements on the target machine.

## Additional Replenishment Compliance
- ORDER WITHIN 3 DAYS alert is implemented.
- STOCKOUT RISK / ORDER NOW / ORDER WITHIN 3 DAYS / OVERSTOCK RISK / NO ORDER REQUIRED alert types are implemented.
- Slow/medium-moving SKU coverage is visualized on the demand dashboard.
