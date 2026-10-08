from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

module = (ROOT / "modules" / "anomaly.py").read_text(encoding="utf-8")
modeling = (ROOT / "core" / "anomaly_modeling.py").read_text(encoding="utf-8")
data = (ROOT / "core" / "anomaly_data.py").read_text(encoding="utf-8")
app = (ROOT / "app.py").read_text(encoding="utf-8")
config = (ROOT / "config.py").read_text(encoding="utf-8")

for token in [
    '"Date Range"',
    '"Warehouse"',
    '"Category"',
    '"Brand"',
    '"SKU"',
    '"Supplier"',
    '"Batch"',
    '"Anomaly Type"',
    '"Severity"',
    '"Stock Status"',
    '"Anomaly Model"',
]:
    assert token in module, f"Missing anomaly filter/control: {token}"

for model in [
    "Hybrid: Isolation Forest + Rules",
    "Isolation Forest",
    "Local Outlier Factor",
    "One-Class SVM",
    "Autoencoder",
    "Statistical Rules",
]:
    assert model in modeling, f"Missing anomaly model: {model}"

for requirement in [
    "Inventory Value at Risk",
    "Detected Anomalies",
    "Critical SKU-Locations",
    "Expiry Risk",
    "Slow / Dead Stock",
    "Suspicious Adjustments",
    "Anomalies Over Time",
    "Severity Distribution",
    "Top Anomalous SKUs",
    "Warehouse Anomaly Comparison",
    "Supplier Delivery Anomalies",
    "Anomaly Investigation & Explainability",
    "Global Model Comparison",
]:
    assert requirement in module, f"Missing dashboard item: {requirement}"

assert '"anomaly_admin"' in config
assert '"anomaly"' in config
assert "Inventory Anomaly Detection" in app
assert "Business_Rule_Score" in data
assert "Synthetic_Anomaly_Label" in data

print("Anomaly implementation static audit passed.")
