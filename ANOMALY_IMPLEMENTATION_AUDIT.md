# Inventory Anomaly Detection — Implementation Audit

## Source Requirements Covered

The third module implements the Inventory Anomaly Detection plan as a connected part of the FMCG Inventory Intelligence Platform.

### Business Objective — PASS
- Detect unusual stock behavior before losses occur.
- Support inventory, warehouse, supply-chain, procurement and finance users.
- Rank anomaly severity and business impact.
- Explain likely reason and recommended corrective action.

### Anomaly Taxonomy — PASS
Implemented business categories:
- Transaction anomalies
- Quantity anomalies
- Time-based anomalies
- Product-level anomalies
- Warehouse anomalies
- Supplier anomalies

Implemented anomaly types include:
- Stock Loss / Shrinkage
- Stock Mismatch
- Negative Inventory
- Abnormal Purchase Size
- Unusual Sales Day
- Suspicious Return
- Duplicate Transaction
- Late Adjustment
- Slow Moving / Dead Stock
- Expiry Risk
- Transfer Mismatch
- Warehouse Capacity Pressure
- Supplier Delay
- Short Delivery

### Dataset / Attributes — PASS
The anomaly module extends the shared FMCG dataset with:
- Transaction_ID
- Product_ID / SKU
- Unit_Cost / Selling_Price
- Opening stock
- Purchases
- Sales
- Returns
- Stock adjustments
- Transfers in/out
- Observed vs expected closing stock
- Supplier and lead-time behavior
- Warehouse capacity
- Batch ID
- Stock age
- Expiry date / days to expiry
- Damaged / expired quantity
- Promotion / calendar context

The Stockout and Demand modules continue to use the same shared SKU, supplier and warehouse foundation.

### Feature Engineering — PASS
Implemented:
- Daily stock change
- Percentage stock change
- Stock movement velocity
- Rolling sales average
- Rolling sales standard deviation
- Demand variation
- Stock-to-sales ratio
- Adjustment ratio
- Return percentage
- Damage percentage
- Inventory turnover proxy
- Inventory value
- Product movement frequency
- Supplier lead-time deviation
- Short-delivery ratio
- Warehouse-level deviation
- Category-level deviation
- Warehouse capacity utilization
- Stock age
- Days to expiry
- Days of inventory
- Transfer mismatch
- Duplicate / late-adjustment indicators

### ML Models — PASS
Dashboard model selector includes:
1. Hybrid: Isolation Forest + Rules
2. Isolation Forest
3. Local Outlier Factor
4. One-Class SVM
5. Autoencoder
6. Statistical Rules

Recommended first version is implemented as the default hybrid design: Isolation Forest + transparent business rules.

The Autoencoder is implemented as a lightweight sklearn MLP reconstruction model so no deep-learning framework is required for Version 1.

### Time / Leakage Discipline — PASS
- Models fit on older chronological training dates.
- Newer records are used for evaluation.
- Unsupervised models learn primarily from synthetic normal records.
- Synthetic anomaly labels are used for evaluation, not as supervised training targets.

### Synthetic Anomaly Injection — PASS
Controlled test anomalies are injected for:
- sudden stock loss
- stock mismatch
- negative stock
- unusual purchase size
- unusual sales day
- suspicious returns
- duplicate transaction
- late adjustment
- expiry risk
- transfer mismatch
- supplier delay
- short delivery

### 0–100 Scoring — PASS
Severity:
- 0–30: NORMAL
- >30–60: MODERATE
- >60–80: HIGH
- >80–100: CRITICAL

Model score + transparent business rules are available.

### Dashboard KPIs — PASS
- Inventory Value at Risk
- Detected Anomalies
- Critical SKU-Locations
- Expiry Risk
- Slow / Dead Stock
- Suspicious Adjustments

### Dashboard Charts — PASS
- Anomalies over time
- Severity distribution
- Top anomalous SKUs
- Category-wise anomalies
- Warehouse anomaly comparison
- Stock vs expected stock trend
- Supplier delivery anomalies
- Anomaly type distribution
- Taxonomy distribution

### Required Filters — PASS
- Date range
- Category
- Brand
- SKU
- Supplier
- Warehouse
- Batch
- Anomaly Type
- Severity
- Stock Status
- ML Model
- Detected anomalies only

The selected model genuinely changes:
- anomaly score
- severity
- detected count
- inventory value at risk
- rankings
- charts
- alert queue

### Model Comparison — PASS
Table includes:
- Model
- Detected anomalies
- Precision
- Recall
- F1 score
- False positive rate
- Execution time

Also includes:
- confusion matrix
- score separation
- model metric chart
- detected-anomaly comparison
- execution-time comparison
- warehouse/category risk heatmap

### Investigation / Explainability — PASS
The manager can open an anomaly and see:
- Product / SKU
- Batch
- Warehouse
- Supplier
- Date
- Model used
- 0–100 anomaly score
- Severity
- Observed stock
- Expected stock
- Historical normal behavior
- Sales vs normal trend
- Business signal strengths
- Possible reason
- Business impact
- Recommended action

### Enterprise Integration — PASS
- Separate `anomaly_admin` login
- `platform_admin` can access all three modules
- Shared Unicorn Enterprise UI
- Shared FMCG master-data concept
- Independent dashboard/workspace
- CSV anomaly report export
- CSV anomaly dataset export

## Full Prototype Validation
A full initialization test completed successfully with:
- 99,000 shared FMCG base rows
- 48,600 anomaly transaction rows
- 90 SKUs in anomaly prototype
- 3 warehouses
- 18 suppliers
- approximately 6.8% controlled injected anomalies
- all six anomaly model options trained/scored

The anomaly prototype intentionally stays simple and synthetic-first while preserving a SaaS-ready architecture.
