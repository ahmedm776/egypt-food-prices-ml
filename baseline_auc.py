"""Baseline ranking comparison on the retrospective window (v4).

The headline 'ranking signal' claim is only meaningful against a temporal
baseline. This computes ROC AUC of a prev-month-spike rule (score = 1 if the
previous month of the same series spiked) on the retrospective predictions,
restricted to rows where a previous month exists, and adds the result to
models/metrics.json so the app and docs quote a compared number.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
preds = pd.read_csv(ROOT / "models" / "retrospective_eval_predictions.csv", parse_dates=["month"])
preds = preds.sort_values(["market", "commodity", "month"])
preds["prev_spike"] = preds.groupby(["market", "commodity"])["target_spike"].shift(1)
sub = preds.dropna(subset=["prev_spike"])

result = {
    "n_rows_total": int(len(preds)),
    "n_rows_with_prev_month": int(len(sub)),
    "model_auc_calibrated_on_subset": round(float(roc_auc_score(sub["target_spike"], sub["calibrated_score"])), 4),
    "prev_month_spike_auc": round(float(roc_auc_score(sub["target_spike"], sub["prev_spike"])), 4),
    "note": "Majority baseline carries no ranking (constant score). Subset = rows with a prior month for the same series.",
}
(ROOT / "baseline_auc.json").write_text(json.dumps(result, indent=2))

metrics_path = ROOT / "models" / "metrics.json"
metrics = json.loads(metrics_path.read_text())
metrics["retrospective_eval"]["prev_month_spike_auc"] = result["prev_month_spike_auc"]
metrics["retrospective_eval"]["prev_month_spike_auc_n"] = result["n_rows_with_prev_month"]
metrics["retrospective_eval"]["model_auc_calibrated_on_prev_month_subset"] = result["model_auc_calibrated_on_subset"]
metrics_path.write_text(json.dumps(metrics, indent=2))
print(json.dumps(result, indent=2))
