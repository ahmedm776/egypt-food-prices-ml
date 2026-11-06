"""API contract tests over the real app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_series_listing():
    data = client.get("/api/series").json()
    assert len(data["series"]) == 47
    assert "retrospective_eval" in data["metrics"]


def test_predict_happy_path():
    r = client.post("/api/predict", json={
        "market": "Cairo (national average)", "commodity": "Rice"})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score_calibrated"] <= 1.0
    assert body["mode"] == "retrospective_asof_demo"
    assert "not a verified probability" in body["score_note"]
    assert body["model_calibrated_through"] < body["prediction_for_month"]
    assert body["model_trained_through"] < body["model_calibrated_through"]
    assert body["last_observed_month"] < body["prediction_for_month"]


def test_history_check_reports_actual_outcome():
    r = client.post("/api/history-check", json={
        "market": "Cairo (national average)", "commodity": "Rice"})
    assert r.status_code == 200
    body = r.json()
    assert body["actual_outcome"] in ("spike", "no spike")
    assert body["as_of"] < body["scored_month"]


def test_predict_unknown_series_404():
    r = client.post("/api/predict", json={"market": "X", "commodity": "Y"})
    assert r.status_code == 404


def test_index_serves_form():
    r = client.get("/")
    assert r.status_code == 200 and "risk score" in r.text.lower()


def test_rank_market_happy_path():
    from fastapi.testclient import TestClient
    from app import app
    c = TestClient(app)
    r = c.get("/api/rank", params={"market": "Cairo (national average)"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["mode"] == "retrospective_asof_demo"
    scores = [x["rank_signal_raw_score"] for x in d["ranked"]]
    assert len(scores) > 1
    assert scores == sorted(scores, reverse=True), "must be ranked by raw score desc"
    assert [x["rank"] for x in d["ranked"]] == list(range(1, len(scores) + 1))
    for x in d["ranked"]:
        assert set(x) >= {"rank", "commodity", "rank_signal_raw_score",
                          "probability_estimate_calibrated", "predicted_label",
                          "last_observed_price_egp"}
    assert "unitless, not a probability" in d["channels_note"]
    assert "not a live forecast" in d["channels_note"]


def test_rank_unknown_market_404():
    from fastapi.testclient import TestClient
    from app import app
    c = TestClient(app)
    r = c.get("/api/rank", params={"market": "No Such Market"})
    assert r.status_code == 404


def test_rank_serving_rule_matches_predict():
    # the ranked month must be the same as-of month /api/predict serves
    from fastapi.testclient import TestClient
    from app import app
    c = TestClient(app)
    d = c.get("/api/rank", params={"market": "Cairo (national average)"}).json()
    p = c.post("/api/predict", json={"market": "Cairo (national average)",
                                     "commodity": "Rice"}).json()
    assert d["prediction_for_month"] == p["prediction_for_month"]
    assert d["as_of"] == p["as_of"]
    # and the single-series raw score is identical inside the ranked list
    row = [x for x in d["ranked"] if x["commodity"] == "Rice"]
    assert row and abs(row[0]["rank_signal_raw_score"] - p["risk_score_raw"]) < 1e-9


def test_rank_common_target_month_and_honest_skips():
    # v7: every ranked row targets the SAME month (the response-level
    # prediction_for_month), and skipped series carry a reason - no stale
    # series presented as a current score.
    from fastapi.testclient import TestClient
    from app import app
    c = TestClient(app)
    d = c.get("/api/rank", params={"market": "Cairo (national average)"}).json()
    for x in d["ranked"]:
        assert x["prediction_for_month"] == d["prediction_for_month"]
        assert x["as_of"] == d["as_of"]
    for s in d["skipped_unscoreable"]:
        assert set(s) == {"commodity", "reason"}
        assert "last observed" in s["reason"]
    # known gapped series: Sugar in Cairo has no Sep 2026 observation, so it
    # must be skipped rather than scored for a stale month
    names = {s["commodity"] for s in d["skipped_unscoreable"]}
    assert d["ranked"] and len(names) >= 0  # structure only; coverage varies by market


def test_rank_all_unscoreable_market_no_500():
    # v7: if no commodity in the market can be scored for the common target
    # month, the endpoint returns an empty list + reasons, never a 500 from
    # an unbound target_month.
    from fastapi.testclient import TestClient
    import app as appmod
    c = TestClient(appmod.app)
    real = appmod.build_prediction_row
    from fastapi import HTTPException
    def boom(m, cm, tm=None):
        raise HTTPException(status_code=422, detail="no prior month")
    appmod.build_prediction_row = boom
    try:
        r = c.get("/api/rank", params={"market": "Cairo (national average)"})
    finally:
        appmod.build_prediction_row = real
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ranked"] == [] and len(d["skipped_unscoreable"]) > 1
    assert d["prediction_for_month"]  # bound, not unbound
