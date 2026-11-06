"""FastAPI serving app for the price-spike early-warning model.

Serves a small web UI and a JSON API. For a chosen (market, commodity)
series the app predicts the probability that the average monthly price
rises more than 2% in the month after the series' last observed month,
using only prices that would have been known at that point.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from features import DERIVED_COLS, add_series_features, load_series

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "wfp_food_prices_egy.csv"
MODELS = ROOT / "models"

app = FastAPI(title="Egypt Food Price Spike Early Warning")

_bundle = joblib.load(MODELS / "model.joblib")
MODEL = _bundle["model"]
CALIBRATOR = _bundle["calibrator"]
FEATURE_COLUMNS = _bundle["feature_columns"]
SPIKE_THRESHOLD = _bundle["spike_threshold"]
TRAIN_LAST_MONTH = pd.Timestamp(_bundle["train_last_month"])
CALIBRATION_LAST_MONTH = pd.Timestamp(_bundle["calibration_last_month"])
EARLIEST_ASOF_MONTH = pd.Timestamp(_bundle["earliest_asof_month"])
METRICS = json.loads((MODELS / "metrics.json").read_text())

_df = load_series(DATA)
SERIES = sorted(
    ({"market": m, "commodity": c}
     for m, c in _df[["market", "commodity"]].drop_duplicates().itertuples(index=False)),
    key=lambda s: (s["market"], s["commodity"]),
)


class PredictRequest(BaseModel):
    market: str
    commodity: str


def build_prediction_row(market: str, commodity: str, target_month=None):
    """Feature row for `target_month` (default: month after the series' last
    observed month), built only from prices strictly before that month."""
    g = _df[(_df["market"] == market) & (_df["commodity"] == commodity)]
    if g.empty:
        raise HTTPException(status_code=404, detail="unknown market/commodity series")
    last_month = g["month"].max()
    if target_month is None:
        target_month = last_month + pd.offsets.MonthBegin(1)
    target_month = pd.Timestamp(target_month)
    extra = pd.DataFrame(
        {"market": [market], "commodity": [commodity],
         "month": [target_month], "price": [np.nan]}
    )
    g = pd.concat([g, extra], ignore_index=True)
    g = add_series_features(g)
    if target_month <= last_month:
        # historical (as-of) scoring: pretend we stand at target_month - 1;
        # recompute features after dropping the target month's own price
        g = g[g["month"] < target_month]
        g = pd.concat([g, extra], ignore_index=True)
        g = add_series_features(g)
    row = g[g["month"] == target_month].iloc[-1]
    if pd.isna(row["price_lag_1"]):
        raise HTTPException(
            status_code=422,
            detail="series has no observation in the month before the target; cannot score",
        )
    x = pd.DataFrame([{c: row.get(c, np.nan) for c in DERIVED_COLS}])
    for col in FEATURE_COLUMNS:
        if col.startswith("mkt_") or col.startswith("com_"):
            x[col] = 1.0 if col in (f"mkt_{market}", f"com_{commodity}") else 0.0
    x = x[FEATURE_COLUMNS]
    return x, row, target_month


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "templates" / "index.html").read_text()


@app.get("/health")
def health():
    return {"status": "ok", "n_series": len(SERIES)}


@app.get("/api/series")
def list_series():
    return {"series": SERIES, "metrics": METRICS}


def _score_row(x):
    raw = float(MODEL.predict_proba(x)[0, 1])
    return raw, float(CALIBRATOR.predict([raw])[0])


def _actual_outcome(market, commodity, target_month):
    """Known spike outcome for a historical target month, else None."""
    g = _df[(_df["market"] == market) & (_df["commodity"] == commodity)]
    cur = g[g["month"] == target_month]["price"]
    prev = g[g["month"] == target_month - pd.offsets.MonthBegin(1)]["price"]
    if len(cur) and len(prev):
        return bool(cur.iloc[0] / prev.iloc[0] - 1 > SPIKE_THRESHOLD)
    return None


@app.post("/api/predict")
def predict(req: PredictRequest):
    """As-of score for the month after the series' last observation
    (Oct 2026, dataset ends Sep 2026). Valid as-of: the model was trained
    on months through Sep 2024 and calibrated on Oct 2024 - Sep 2025, both
    strictly before the scored month. Retrospective reconstruction on
    development-exposed data, not a live forecast or a verified
    probability."""
    x, row, target_month = build_prediction_row(req.market, req.commodity)
    assert target_month > CALIBRATION_LAST_MONTH  # enforced serving rule
    raw, cal = _score_row(x)
    return {
        "market": req.market,
        "commodity": req.commodity,
        "prediction_for_month": str(target_month.date()),
        "as_of": str((target_month - pd.offsets.MonthBegin(1)).date()),
        "mode": "retrospective_asof_demo",
        "model_trained_through": str(TRAIN_LAST_MONTH.date()),
        "model_calibrated_through": str(CALIBRATION_LAST_MONTH.date()),
        "risk_score_calibrated": round(cal, 3),
        "risk_score_raw": round(raw, 3),
        "rank_signal_raw_score": round(raw, 3),
        "probability_estimate_calibrated": round(cal, 3),
        "channels_note": (
            "Two channels: (1) ranking signal = raw model score (unitless, not a probability), use it to rank "
            "which series is riskier; (2) probability estimate = isotonic-calibrated "
            "score, a rough and unverified probability. On windows that drift from "
            "the calibration regime (Oct 2024 - Sep 2025), isotonic flat-segment ties "
            "can distort ranking - measured on the 2016-10 origin robustness check "
            "(development-exposed): raw-score AUC 0.694 vs isotonic 0.644 - so ranking "
            "follows the raw score. Retrospective backtest, not a live forecast; no "
            "independent final test exists."
        ),
        "score_note": ("calibrated model risk score (isotonic, fit on "
                       "2024-10..2025-09); retrospective demo on "
                       "development-exposed data - not a verified "
                       "probability and no independent final test exists"),
        "predicted_label": "elevated" if cal >= 0.5 else "low",
        "spike_definition": f"average monthly price rises > {SPIKE_THRESHOLD:.0%}",
        "last_observed_month": str((target_month - pd.offsets.MonthBegin(1)).date()),
        "last_observed_price_egp": round(float(row["price_lag_1"]), 2),
        "model_evaluation": METRICS,
    }


@app.get("/api/rank")
def rank_market(market: str):
    """Rank ALL commodities in one market by the raw rank signal for the
    month after the market's last observed month. Same as-of serving rule
    as /api/predict: retrospective reconstruction on development-exposed
    data, not a live forecast. The raw score is unitless and used only to
    order the list; the calibrated estimate is shown alongside, labelled
    rough and unverified. No new metric is computed by this endpoint."""
    commodities = sorted({s["commodity"] for s in SERIES if s["market"] == market})
    if not commodities:
        raise HTTPException(status_code=404, detail="unknown market")
    # One COMMON target month for the whole list: the month after the market's
    # last observation. Ranking only compares scores for the same target
    # month - scoring each series for its own last month+1 would silently mix
    # target months inside one ranked list (fixed 2026-11-06, v7). Series
    # without an observation in the month before the common target cannot be
    # scored for it and are listed as skipped, with the reason - a stale
    # series is never presented as a current score.
    market_last = _df[_df["market"] == market]["month"].max()
    target_month = market_last + pd.offsets.MonthBegin(1)
    assert target_month > CALIBRATION_LAST_MONTH  # enforced serving rule
    as_of = target_month - pd.offsets.MonthBegin(1)
    ranked, skipped = [], []
    for c in commodities:
        try:
            x, row, tm = build_prediction_row(market, c, target_month)
            raw, cal = _score_row(x)
            ranked.append({
                "commodity": c,
                "rank_signal_raw_score": round(raw, 3),
                "probability_estimate_calibrated": round(cal, 3),
                "predicted_label": "elevated" if cal >= 0.5 else "low",
                "last_observed_price_egp": round(float(row["price_lag_1"]), 2),
                "prediction_for_month": str(tm.date()),
                "as_of": str((tm - pd.offsets.MonthBegin(1)).date()),
            })
        except HTTPException:
            g = _df[(_df["market"] == market) & (_df["commodity"] == c)]
            last = str(g["month"].max().date()) if len(g) else "no data"
            skipped.append({
                "commodity": c,
                "reason": ("no observation in the month before the common "
                           f"target month {target_month.date()}; last observed {last}"),
            })
    ranked.sort(key=lambda r: r["rank_signal_raw_score"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return {
        "market": market,
        "prediction_for_month": str(target_month.date()),
        "as_of": str((target_month - pd.offsets.MonthBegin(1)).date()),
        "mode": "retrospective_asof_demo",
        "model_trained_through": str(TRAIN_LAST_MONTH.date()),
        "model_calibrated_through": str(CALIBRATION_LAST_MONTH.date()),
        "ranked": ranked,
        "skipped_unscoreable": skipped,
        "channels_note": (
            "Ranked by the raw model score (unitless, not a probability): higher means "
            "riskier. The calibrated percentage next to each row is a rough, unverified "
            "probability estimate; on windows that drift from the calibration regime "
            "(Oct 2024 - Sep 2025), calibration ties can distort ranking, so the order "
            "follows the raw score (2016-origin robustness check, development-exposed: "
            "raw-score AUC 0.694 vs isotonic 0.644). Retrospective reconstruction on "
            "development-exposed data, not a live forecast; no independent final test "
            "exists. Ranking commodities is the model's documented strength (temporal "
            "frozen-protocol AUC 0.759); this view does not change any metric."
        ),
        "spike_definition": f"average monthly price rises > {SPIKE_THRESHOLD:.0%}",
    }


@app.post("/api/history-check")
def history_check(req: PredictRequest):
    """Score the series' LAST observed month as-of the previous one and
    show what actually happened - a verifiable historical check."""
    g = _df[(_df["market"] == req.market) & (_df["commodity"] == req.commodity)]
    if g.empty:
        raise HTTPException(status_code=404, detail="unknown market/commodity series")
    # most recent month whose previous month is also observed (scoreable
    # as-of and with a known outcome)
    months = sorted((pd.Timestamp(m) for m in g["month"].unique()), reverse=True)
    have = set(months)
    target_month = next(
        (m for m in months if m - pd.offsets.MonthBegin(1) in have), None
    )
    if target_month is None:
        raise HTTPException(status_code=422, detail="no consecutive month pair in this series")
    if target_month < EARLIEST_ASOF_MONTH:
        # scoring this month would be in-sample for training or calibration
        supported = [m for m in months if m >= EARLIEST_ASOF_MONTH]
        raise HTTPException(
            status_code=422,
            detail=(
                f"refused: {target_month.date()} is inside the training or "
                f"calibration window. As-of history checks are supported "
                f"from {EARLIEST_ASOF_MONTH.date()} onward; latest scoreable "
                f"month for this series: "
                f"{(supported[0].date() if supported else 'none')}."
            ),
        )
    x, row, _ = build_prediction_row(req.market, req.commodity, target_month)
    raw, cal = _score_row(x)
    return {
        "market": req.market,
        "commodity": req.commodity,
        "scored_month": str(target_month.date()),
        "as_of": str((target_month - pd.offsets.MonthBegin(1)).date()),
        "model_trained_through": str(TRAIN_LAST_MONTH.date()),
        "model_calibrated_through": str(CALIBRATION_LAST_MONTH.date()),
        "risk_score_calibrated": round(cal, 3),
        "risk_score_raw": round(raw, 3),
        "rank_signal_raw_score": round(raw, 3),
        "probability_estimate_calibrated": round(cal, 3),
        "channels_note": (
            "Two channels: (1) ranking signal = raw model score (unitless, not a probability), use it to rank "
            "which series is riskier; (2) probability estimate = isotonic-calibrated "
            "score, a rough and unverified probability. On windows that drift from "
            "the calibration regime (Oct 2024 - Sep 2025), isotonic flat-segment ties "
            "can distort ranking - measured on the 2016-10 origin robustness check "
            "(development-exposed): raw-score AUC 0.694 vs isotonic 0.644 - so ranking "
            "follows the raw score. Retrospective backtest, not a live forecast; no "
            "independent final test exists."
        ),
        "actual_outcome": ("spike" if _actual_outcome(req.market, req.commodity, target_month)
                           else "no spike"),
    }
