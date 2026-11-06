"""Train and evaluate the price-spike early-warning classifier, v4 CANDIDATE.

FROZEN PROTOCOL (fixed before running, no adjustments after viewing results):
- model: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03,
  l2_regularization=1.0, max_leaf_nodes=15, random_state=42); spike
  definition: monthly rise > 2%; decision threshold 0.5
- chronological windows: train < 2024-10 | calibrate 2024-10..2025-09 |
  evaluate >= 2025-10
- isotonic calibration fit ONLY on the calibration window, on a model
  trained ONLY on the training window -> every date used for fitting or
  calibrating is strictly before every evaluated or served date
- evaluated ONCE on the evaluation window

HONESTY: the full dataset was exposed during development (framing, model,
features, hyperparameters were chosen after exploratory scoring). These
numbers are a RETROSPECTIVE evaluation of a frozen pipeline. There is no
independent final test on this dataset and none is claimed. Not beating
a baseline on some metric is reported as-is.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import (
    SPIKE_THRESHOLD,
    build_features,
    load_series,
    model_frame,
    split_chronological,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "wfp_food_prices_egy.csv"
MODELS = ROOT / "models"


def scores(y, proba, threshold=0.5):
    y = np.asarray(y, dtype=int)
    pred = (proba >= threshold).astype(int)
    return {
        "auc": round(roc_auc_score(y, proba), 3),
        "accuracy_raw_threshold_0p5": round(accuracy_score(y, pred), 3),
        "f1_spike_class_raw_threshold_0p5": round(f1_score(y, pred), 3),
        "brier_raw": round(brier_score_loss(y, proba), 4),
    }


def baselines(frame, y, train_pos_rate):
    y = np.asarray(y, dtype=int)
    majority = int(train_pos_rate >= 0.5)
    prev = (frame["r1"] > SPIKE_THRESHOLD).astype(int)
    return {
        "majority_accuracy": round(accuracy_score(y, [majority] * len(y)), 3),
        "prev_month_spike_accuracy": round(accuracy_score(y, prev), 3),
    }


def window_meta(name, frame):
    return {
        "window": name,
        "first_month": str(frame["month"].min().date()),
        "last_month": str(frame["month"].max().date()),
        "n_rows": int(len(frame)),
    }


def main():
    df = load_series(DATA)
    feat = build_features(df)
    train_f, calib_f, eval_f = split_chronological(feat)
    tr, xcols = model_frame(train_f)
    ca, _ = model_frame(calib_f)
    ev, _ = model_frame(eval_f)

    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.03, l2_regularization=1.0,
        max_leaf_nodes=15, random_state=42
    )
    clf.fit(tr[xcols], tr["target_spike"])

    # calibrate ONLY on the calibration window
    proba_ca = clf.predict_proba(ca[xcols])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(proba_ca, ca["target_spike"].astype(int))

    # evaluate ONCE on the evaluation window
    proba_ev = clf.predict_proba(ev[xcols])[:, 1]
    cal_ev = calibrator.predict(proba_ev)
    y_ev = ev["target_spike"].astype(int)
    eval_metrics = scores(y_ev, proba_ev)
    eval_metrics["brier_calibrated"] = round(brier_score_loss(y_ev, cal_ev), 4)
    eval_metrics["accuracy_calibrated_threshold_0p5"] = round(accuracy_score(y_ev, (cal_ev >= 0.5).astype(int)), 3)
    eval_metrics["f1_spike_class_calibrated_threshold_0p5"] = round(f1_score(y_ev, (cal_ev >= 0.5).astype(int)), 3)
    eval_metrics.update(baselines(ev, y_ev, tr["target_spike"].mean()))
    eval_metrics["n_rows"] = int(len(ev))
    eval_metrics["positive_rate"] = round(float(y_ev.mean()), 3)

    metrics = {
        "task": f"binary: monthly price rises > {SPIKE_THRESHOLD:.0%} next month",
        "model": "HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, l2_regularization=1.0, max_leaf_nodes=15, frozen) - v4 candidate, selected on rolling origins 2016-2022 only (development-exposed data, not a holdout)",
        "protocol": "chronological train -> calibrate -> evaluate, frozen before run, evaluated once",
        "train_window": window_meta("train (< 2024-10)", tr),
        "calibration_window": window_meta("calibration (2024-10 .. 2025-09)", ca),
        "retrospective_eval_window": window_meta("eval (>= 2025-10)", ev),
        "retrospective_eval": eval_metrics,
        "baseline_random_auc": 0.5,
        "independent_final_test": False,
        "dataset_development_exposure": (
            "entire dataset was exposed during development (framing, model "
            "class, features, hyperparameters); numbers are a retrospective "
            "evaluation of a frozen pipeline, not an independent test"
        ),
        "asof_serving_rule": (
            "as-of scores are served only for months after the calibration "
            "window (>= 2025-10); every fit/calibration date strictly "
            "precedes every served date"
        ),
    }

    MODELS.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "calibrator": calibrator,
            "feature_columns": xcols,
            "spike_threshold": SPIKE_THRESHOLD,
            "train_last_month": str(tr["month"].max().date()),
            "calibration_last_month": str(ca["month"].max().date()),
            "earliest_asof_month": "2025-10-01",
        },
        MODELS / "model.joblib",
    )
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    out = ev[["market", "commodity", "month", "price", "target_spike"]].copy()
    out["raw_score"] = proba_ev
    out["calibrated_score"] = cal_ev
    out.to_csv(MODELS / "retrospective_eval_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
