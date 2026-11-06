import json, subprocess, sys, shutil
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "prospective")); sys.path.insert(0, str(ROOT / "src"))
import score_ledger as S

LEDGER = json.loads((ROOT / "prospective" / "ledger_2026-10_late-issued.json").read_text())

def test_ledger_hash_valid_and_untampered():
    assert S.verify_ledger(LEDGER)
    t = json.loads(json.dumps(LEDGER)); t["predictions"][0]["calibrated"] = 0.99
    assert not S.verify_ledger(t)

def test_late_issued_ledger_is_not_labelled_prospective():
    assert LEDGER["evidence_class"] == "late_issued_backfill"
    assert LEDGER["frozen_at_utc"][:10] > LEDGER["target_month"][:10]

def test_ledger_targets_month_after_data_and_only_past_inputs():
    assert LEDGER["target_month"] == "2026-10-01" and LEDGER["latest_observed_month_in_data"] == "2026-09-01"
    assert all(0 <= p["calibrated"] <= 1 for p in LEDGER["predictions"])

def test_scoring_with_current_data_reports_no_outcome_yet():
    out = S.score(LEDGER, ROOT / "data" / "wfp_food_prices_egy.csv")
    assert out["n_with_outcome"] == 0 and out["pre_target_rows_changed_since_freeze"] is False

def test_scoring_runs_on_synthetic_outcome(tmp_path):
    """Mechanics only: append a fabricated Oct-2026 row to a COPY and check metrics compute."""
    raw = (ROOT / "data" / "wfp_food_prices_egy.csv").read_text().splitlines()
    hdr = raw[0].split(","); rows = []
    p = LEDGER["predictions"][0]
    import csv, io
    r = list(csv.DictReader(io.StringIO("\n".join(raw))))
    base = [x for x in r[1:] if x["market"] == p["market"] and x["commodity"] == p["commodity"] and x["date"].startswith("2026-09")]
    assert base
    new = dict(base[0]); new["date"] = "2026-10-15"; new["price"] = str(float(base[0]["price"]) * 1.10)
    out = tmp_path / "copy.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); [w.writerow(x) for x in r]; w.writerow(new)
    res = S.score(LEDGER, out)
    assert res["n_with_outcome"] >= 1 and res["n_spikes"] >= 1
    assert res["n_prev_month_rule"] >= 1 and "acc_prev_month_spiked_rule" in res  # baseline computable from stored as-of values
