"""Freeze a PROSPECTIVE prediction ledger for the first month after the data ends.

The shipped retrospective evaluation is development-exposed. A ledger only counts
as prospective if it is issued BEFORE its target month begins; the script labels each
ledger automatically (evidence_class). Issued after the month began, it is a
late-issued backfill and proves nothing about unseen outcomes. This script scores every scoreable series for
the target month (default: month after the latest observation) with the SHIPPED,
unchanged model, and writes a ledger with hashes of the model file, the data
file and the data rows the predictions used. It changes nothing else.

Limits (stated in prospective/README.md): the timestamp is self-attested unless
the ledger hash is published somewhere external; one month is ~47 series at
most; no accuracy claim is made in advance of the outcome.
"""
import hashlib, json, sys, datetime as dt
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import app as A  # noqa: E402  (loads the shipped model; read-only use)

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(out="prospective/ledger_2026-10_late-issued.json"):
    df = A._df
    last = df["month"].max()
    target = last + pd.offsets.MonthBegin(1)
    preds, skipped = [], []
    for s in A.SERIES:
        try:
            x, row, tm = A.build_prediction_row(s["market"], s["commodity"], target)
        except HTTPException as e:
            skipped.append({**s, "reason": e.detail}); continue
        raw, cal = A._score_row(x)
        g = df[(df.market == s["market"]) & (df.commodity == s["commodity"]) & (df.month == last)]
        preds.append({**s, "raw": round(raw, 6), "calibrated": round(cal, 6),
                      "prev_month_price": float(g.price.iloc[0]) if len(g) else None,
                      "prev_month_spiked_as_of_target": (None if pd.isna(row.get("r1")) else bool(row["r1"] > A.SPIKE_THRESHOLD))})
    now = dt.datetime.now(dt.timezone.utc)
    evidence_class = "prospective" if now.replace(tzinfo=None) < target.to_pydatetime() else "late_issued_backfill"
    used = df[df.month < target]
    body = {
        "purpose": "frozen-prediction ledger for the one-month-ahead task (horizon and task unchanged)",
        "evidence_class": evidence_class,
        "evidence_class_rule": "prospective only if frozen_at is before the first day of the target month; otherwise late_issued_backfill (tamper evidence only: hashes do not prove the outcome was unseen during development)",
        "target_month": str(target.date()),
        "latest_observed_month_in_data": str(last.date()),
        "frozen_at_utc": now.isoformat(timespec="seconds"),
        "model_sha256": sha256(ROOT / "models" / "model.joblib"),
        "data_csv_sha256": sha256(ROOT / "data" / "wfp_food_prices_egy.csv"),
        "monthly_rows_used_sha256": hashlib.sha256(used.to_csv(index=False).encode()).hexdigest(),
        "spike_definition": f"price[m]/price[m-1]-1 > {A.SPIKE_THRESHOLD}",
        "decision_threshold": 0.5,
        "pre_declared_metrics": ["ROC AUC (calibrated score)", "Brier (calibrated)",
            "accuracy at 0.5 vs majority-class and previous-month-spiked baselines",
            "report n and the number of spikes; AUC undefined if one class only"],
        "predictions": preds, "skipped_series": skipped,
    }
    body["ledger_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    Path(ROOT / out).write_text(json.dumps(body, indent=2))
    print(f"wrote {out}: {len(preds)} predictions, {len(skipped)} skipped, ledger sha256 {body['ledger_sha256']}")

if __name__ == "__main__":
    main(*sys.argv[1:])
