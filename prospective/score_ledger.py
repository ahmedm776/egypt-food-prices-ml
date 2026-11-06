"""Score a frozen ledger against a newer WFP CSV. Run once the target month is published.

Usage: python prospective/score_ledger.py LEDGER.json NEW_WFP_CSV
Refuses to score if the ledger hash or the model hash do not match, and warns if the
pre-target rows of the new CSV differ from what the predictions used (WFP revisions).
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import load_series, SPIKE_THRESHOLD

def verify_ledger(L):
    body = {k: v for k, v in L.items() if k != "ledger_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == L["ledger_sha256"]

def score(L, csv):
    if not verify_ledger(L):
        raise SystemExit("ledger hash mismatch - ledger was modified after freeze")
    mh = hashlib.sha256((ROOT / "models" / "model.joblib").read_bytes()).hexdigest()
    if mh != L["model_sha256"]:
        raise SystemExit("model file differs from the one that made the ledger")
    df = load_series(csv)
    target = pd.Timestamp(L["target_month"]); prev = target - pd.offsets.MonthBegin(1)
    used = df[df.month < target]
    drift = hashlib.sha256(used.to_csv(index=False).encode()).hexdigest() != L["monthly_rows_used_sha256"]
    rows = []
    for p in L["predictions"]:
        g = df[(df.market == p["market"]) & (df.commodity == p["commodity"])]
        cur = g[g.month == target].price; pr = g[g.month == prev].price
        if len(cur) and len(pr):
            rows.append({**p, "spike": bool(cur.iloc[0] / pr.iloc[0] - 1 > SPIKE_THRESHOLD)})
    n = len(rows)
    out = {"target_month": L["target_month"], "n_predictions_in_ledger": len(L["predictions"]),
           "n_with_outcome": n, "pre_target_rows_changed_since_freeze": bool(drift)}
    if n == 0:
        out["status"] = "target month not yet published for any ledger series; nothing scored"; return out
    y = np.array([r["spike"] for r in rows]); c = np.array([r["calibrated"] for r in rows])
    out.update(n_spikes=int(y.sum()), brier_calibrated=round(float(brier_score_loss(y, c)), 4),
               acc_model_0p5=round(float(((c >= .5) == y).mean()), 4),
               acc_majority=round(float(max(y.mean(), 1 - y.mean())), 4))
    out["auc"] = round(float(roc_auc_score(y, c)), 4) if 0 < y.sum() < n else None
    pm = [r for r in rows if r["prev_month_spiked_as_of_target"] is not None]
    if pm:
        ypm = np.array([r["spike"] for r in pm]); b = np.array([r["prev_month_spiked_as_of_target"] for r in pm])
        out["acc_prev_month_spiked_rule"] = round(float((b == ypm).mean()), 4); out["n_prev_month_rule"] = len(pm)
    out["note"] = "one month, small n; retrospective-vs-prospective gap is informative only if n and spike count are not tiny"
    return out

if __name__ == "__main__":
    print(json.dumps(score(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2]), indent=2))
