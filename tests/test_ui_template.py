"""Regression tests for the UI template: a displayed score must never
outlive the selection it was computed for (stale-result mismatch)."""
from pathlib import Path

TEMPLATE = (Path(__file__).resolve().parents[1] / "templates" / "index.html").read_text(encoding="utf-8")


def test_clear_result_defined_and_hides_result_panel():
    assert "function clearResult()" in TEMPLATE
    assert "document.getElementById('result').style.display = 'none';" in TEMPLATE
    assert "document.getElementById('prob').textContent = '';" in TEMPLATE
    assert "document.getElementById('meta').innerHTML = '';" in TEMPLATE


def test_market_change_clears_result_after_rebuilding_commodities():
    assert "marketSel.addEventListener('change', () => { updateCommodities(); clearResult(); });" in TEMPLATE


def test_commodity_change_clears_result():
    assert "commoditySel.addEventListener('change', clearResult);" in TEMPLATE
