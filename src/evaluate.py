"""Regenerate evaluation figures from the saved model: ROC/PR curves per
window and a calibration (reliability) plot. Deterministic re-derivation
of the same calendar split used by train.py."""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, load_series, model_frame, split_chronological

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "wfp_food_prices_egy.csv"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"


def main():
    bundle = joblib.load(MODELS / "model.joblib")
    clf, cal, xcols = bundle["model"], bundle["calibrator"], bundle["feature_columns"]
    metrics = json.loads((MODELS / "metrics.json").read_text())

    df = load_series(DATA)
    feat = build_features(df)
    _, calib_f, eval_f = split_chronological(feat)
    windows = {}
    for name, f in (("calibration (fit window)", calib_f),
                    ("retrospective eval", eval_f)):
        d, _ = model_frame(f)
        windows[name] = (d["target_spike"].astype(int).to_numpy(),
                         clf.predict_proba(d[xcols])[:, 1])

    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (y, p) in windows.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = metrics["retrospective_eval"]["auc"] if name.startswith("retrospective") else None
        axes[0].plot(fpr, tpr, label=f"{name}" + (f" (AUC {auc:.2f})" if auc else ""))
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{name} (base {y.mean():.2f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="random")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_pr_curves.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, (y, p) in windows.items():
        frac, mean_pred = calibration_curve(y, cal.predict(p), n_bins=6, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.set(xlabel="mean calibrated score", ylabel="observed spike frequency",
           )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_curve.png", dpi=140)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
