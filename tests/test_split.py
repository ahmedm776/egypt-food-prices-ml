"""Chronological window and date-boundary guarantees."""
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import (
    CALIB_START,
    EARLIEST_ASOF_MONTH,
    EVAL_START,
    build_features,
    load_series,
    model_frame,
    split_chronological,
)
from app import (
    CALIBRATION_LAST_MONTH,
    EARLIEST_ASOF_MONTH as APP_EARLIEST,
    TRAIN_LAST_MONTH,
    build_prediction_row,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "wfp_food_prices_egy.csv"


def test_chronological_windows_disjoint_and_global():
    feat = build_features(load_series(DATA))
    tr, ca, ev = split_chronological(feat)
    assert tr["month"].max() < CALIB_START
    assert ca["month"].min() >= CALIB_START and ca["month"].max() < EVAL_START
    assert ev["month"].min() >= EVAL_START
    assert not ((ca["month"] < CALIB_START).any())


def test_artifact_dates_precede_served_dates():
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    train_last = pd.Timestamp(bundle["train_last_month"])
    calib_last = pd.Timestamp(bundle["calibration_last_month"])
    earliest = pd.Timestamp(bundle["earliest_asof_month"])
    assert train_last < calib_last < earliest
    assert earliest == EARLIEST_ASOF_MONTH == APP_EARLIEST
    assert train_last == TRAIN_LAST_MONTH and calib_last == CALIBRATION_LAST_MONTH


def test_prediction_alignment_against_real_prices():
    market, commodity = "Cairo (national average)", "Rice"
    target = pd.Timestamp("2026-06-01")
    x, row, tm = build_prediction_row(market, commodity, target)
    assert tm == target and tm >= EARLIEST_ASOF_MONTH
    df = load_series(DATA)
    g = df[(df["market"] == market) & (df["commodity"] == commodity)]
    prices = dict(zip(g["month"], g["price"]))
    for k in (1, 2, 3):
        expect = prices.get(target - pd.offsets.MonthBegin(k))
        got = row[f"price_lag_{k}"]
        assert (pd.isna(got) if expect is None else got == expect)


def test_history_check_refuses_in_sample_months():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    # series A ends 2022-10: every month is inside training -> must refuse
    r = client.post("/api/history-check", json={
        "market": "National Average", "commodity": "Rice"})
    assert r.status_code == 422
    assert "training or calibration window" in r.json()["detail"]
