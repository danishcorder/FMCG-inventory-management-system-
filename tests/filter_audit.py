from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

common = (ROOT / "modules" / "common.py").read_text(encoding="utf-8")
stock = (ROOT / "modules" / "stockout.py").read_text(encoding="utf-8")
demand = (ROOT / "modules" / "demand.py").read_text(encoding="utf-8")
theme = (ROOT / "core" / "theme.py").read_text(encoding="utf-8")

required_common = [
    '"Warehouse"',
    '"Product Category"',
    '"SKU"',
    '"Supplier"',
    '"Mover Class"',
    '"Promotion"',
    '"Date Range"',
]
for token in required_common:
    assert token in common, f"Missing filter control: {token}"

for token in ['"Risk Level"', '"Stock Status"', '"Recommended Action"']:
    assert token in common, f"Missing stockout enterprise filter: {token}"

assert 'model_filter(' in stock
assert 'date_range_filter(' in stock
assert 'show_risk=True' in stock
assert 'show_stock_status=True' in stock
assert 'show_action=True' in stock

assert 'model_filter(' in demand
assert 'date_range_filter(' in demand
assert 'horizon_filter(' in demand
assert 'show_projected_risk=True' in demand
assert 'show_overstock=True' in demand

assert "Unicorn Enterprise UI" in theme or "UNICORN" in theme
assert "unicorn-page-header" in theme

print("Filter/dashboard static audit passed.")
