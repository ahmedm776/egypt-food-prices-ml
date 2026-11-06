"""ONE bounded calibration-repair comparison for egypt-food-prices-ml (utility repair, not accuracy work).

Question: can a strictly-monotone (positive-slope Platt) calibration arm protect the SHIPPED
v4 model's ranking utility under regime shift, where isotonic's flat segments can tie-destroy
AUC (documented PROTOCOL_FREEZE note 7; measured on v3 origins: 2016 AUC 0.720->0.563 calibrated)?

FROZEN BEFORE FITS (calib_freeze.json): origins 2016/2018/2020/2022 (selection origins ONLY -
shipped 2024-10 origin excluded from selection), v4 model config identical to shipped
(max_iter=200, lr=0.03, l2=1.0, max_leaf_nodes=15, seed 42), windows train<Y | calib Y..Y+12mo |
eval Y+12mo..Y+24mo, metrics AUC+Brier per arm, adoption rule fixed. All origins
development-exposed - robustness checks, never holdouts. No threshold/feature/target changes.
Platt arm: LogisticRegression on 1D raw scores; coefficient MUST be > 0 per origin (asserted
from the fit, not assumed) else the arm is recorded as failing monotonicity; actual AUC always
measured from predictions (finite-precision/sigmoid ties would show up in the measured AUC).
Brier improvement is NOT claimed before measurement.
"""
import json, sys, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG / "src"))
from features import SPIKE_THRESHOLD, build_features, load_series, model_frame

ORIGINS = ["2016-10", "2018-10", "2020-10", "2022-10"]
V4 = dict(max_iter=200, learning_rate=0.03, l2_regularization=1.0, max_leaf_nodes=15, random_state=42)

def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def windows(feat, origin):
    y0 = pd.Timestamp(origin); y1 = y0 + pd.DateOffset(years=1); y2 = y0 + pd.DateOffset(years=2)
    return feat[feat["month"] < y0], feat[(feat["month"] >= y0) & (feat["month"] < y1)], feat[(feat["month"] >= y1) & (feat["month"] < y2)]

stage = sys.argv[1] if len(sys.argv) > 1 else "freeze"

if stage == "freeze":
    freeze = {
        "frozen_at": "2026-11-05T21:30 EET", "stage": "pre-fit",
        "origins": ORIGINS, "model_config": {k: v for k, v in V4.items()},
        "windows": "train < Y | calibrate Y..Y+12mo | evaluate Y+12mo..Y+24mo",
        "arms": ["raw", "isotonic", "platt(positive-slope asserted)"],
        "metrics": ["roc_auc", "brier_score_loss"],
        "adoption_rule": "adopt platt ONLY IF on EVERY origin auc_platt >= auc_raw - 0.005 AND mean(brier_platt) <= mean(brier_isotonic) + 0.005; otherwise keep isotonic, document, stop. Fixed before fits; no tweaking after results.",
        "platt_monotonicity": "logistic coefficient asserted > 0 per origin; measured AUC used (ties from float/sigmoid would be visible in it)",
        "exposure": "all 4 origins development-exposed; robustness checks, NOT independent validation; shipped 2024-10 origin excluded from selection",
        "data_sha256": sha_file(PKG / "data" / "wfp_food_prices_egy.csv"),
        "code_sha256": sha_file(Path(__file__).resolve()),
        "caps": {"wall_min": 20, "fit_compute_min": 5},
    }
    json.dump(freeze, open(HERE / "calib_freeze.json", "w"), indent=1)
    print("frozen:", json.dumps({k: freeze[k] for k in ("origins", "adoption_rule", "data_sha256")}, indent=1))

elif stage == "run":
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    t0 = time.time(); fit_sec_total = 0.0
    assert sha_file(PKG / "data" / "wfp_food_prices_egy.csv") == json.load(open(HERE / "calib_freeze.json"))["data_sha256"]
    df = load_series(PKG / "data" / "wfp_food_prices_egy.csv")
    feat = build_features(df)
    results = []
    for origin in ORIGINS:
        train_f, calib_f, eval_f = windows(feat, origin)
        tr, xcols = model_frame(train_f); ca, _ = model_frame(calib_f); ev, _ = model_frame(eval_f)
        assert not (set(tr["month"]) & set(ca["month"])) and not (set(ca["month"]) & set(ev["month"])), "window overlap"
        tf = time.time()
        clf = HistGradientBoostingClassifier(**V4).fit(tr[xcols], tr["target_spike"].astype(int))
        fit_sec_total += time.time() - tf
        raw_ca = clf.predict_proba(ca[xcols])[:, 1]
        y_ca = ca["target_spike"].astype(int).to_numpy()
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_ca, y_ca)
        platt = LogisticRegression(random_state=42).fit(raw_ca.reshape(-1, 1), y_ca)
        slope = float(platt.coef_[0][0])
        raw = clf.predict_proba(ev[xcols])[:, 1]
        y = ev["target_spike"].astype(int).to_numpy()
        cal_iso = iso.predict(raw)
        cal_pl = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        rec = {
            "origin": origin, "n_train": int(len(tr)), "n_calib": int(len(ca)), "n_eval": int(len(ev)),
            "eval_base_rate": round(float(y.mean()), 3),
            "auc_raw": round(float(roc_auc_score(y, raw)), 4),
            "auc_isotonic": round(float(roc_auc_score(y, cal_iso)), 4),
            "auc_platt": round(float(roc_auc_score(y, cal_pl)), 4),
            "brier_raw": round(float(brier_score_loss(y, raw)), 4),
            "brier_isotonic": round(float(brier_score_loss(y, cal_iso)), 4),
            "brier_platt": round(float(brier_score_loss(y, cal_pl)), 4),
            "platt_slope": round(slope, 4),
            "platt_slope_positive": bool(slope > 0),
        }
        results.append(rec)
        print(json.dumps(rec), flush=True)
    out = {
        "protocol": json.load(open(HERE / "calib_freeze.json")),
        "results": results,
        "mean_brier_isotonic": round(float(np.mean([r["brier_isotonic"] for r in results])), 4),
        "mean_brier_platt": round(float(np.mean([r["brier_platt"] for r in results])), 4),
        "wall_sec": round(time.time() - t0, 1), "histgb_fit_sec_total": round(fit_sec_total, 1),
    }
    ok_rank = all(r["auc_platt"] >= r["auc_raw"] - 0.005 and r["platt_slope_positive"] for r in results)
    ok_brier = out["mean_brier_platt"] <= out["mean_brier_isotonic"] + 0.005
    out["adopt_platt"] = bool(ok_rank and ok_brier)
    out["decision_detail"] = {"rank_preserved_all_origins": ok_rank, "brier_within_margin": ok_brier}
    json.dump(out, open(HERE / "calib_results.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("mean_brier_isotonic", "mean_brier_platt", "adopt_platt", "decision_detail", "wall_sec", "histgb_fit_sec_total")}, indent=1), flush=True)
