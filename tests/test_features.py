"""Feature pipeline tests, including the no-leakage guarantee."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import add_series_features


def _toy_series():
    # monthly prices Jan-Jun 2024, with March missing (gap)
    months = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-04-01",
                             "2024-05-01", "2024-06-01"])
    return pd.DataFrame({
        "market": ["National Average"] * 5,
        "commodity": ["Rice"] * 5,
        "month": months,
        "price": [10.0, 20.0, 40.0, 50.0, 60.0],
    })


def test_lags_are_calendar_exact_not_row_shift():
    g = add_series_features(_toy_series())
    april = g[g["month"] == "2024-04-01"].iloc[0]
    # February is two calendar months before April
    assert april["price_lag_2"] == 20.0
    # March (one month before) was not observed -> NaN, never the Feb value
    assert np.isnan(april["price_lag_1"])


def test_lag1_equals_previous_observed_month_price():
    g = add_series_features(_toy_series())
    june = g[g["month"] == "2024-06-01"].iloc[0]
    assert june["price_lag_1"] == 50.0  # May price, strictly in the past


def test_no_feature_uses_current_price():
    g = add_series_features(_toy_series())
    feature_cols = [c for c in g.columns if c not in
                    ("market", "commodity", "month", "price")]
    # double every price: lag features must stay identical for early rows
    g2 = _toy_series()
    g2.loc[g2["month"] == "2024-06-01", "price"] = 999.0
    g2 = add_series_features(g2)
    june_a = g[g["month"] == "2024-06-01"].iloc[0]
    june_b = g2[g["month"] == "2024-06-01"].iloc[0]
    for col in feature_cols:
        a, b = june_a[col], june_b[col]
        both_nan = pd.isna(a) and pd.isna(b)
        assert both_nan or a == b, f"{col} changed when current price changed"
