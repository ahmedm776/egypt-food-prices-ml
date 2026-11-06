"""Baseline comparison under the rolling-origin protocol (development robustness).

Selection origins: 2016/2018/2020/2022-10 ONLY. The 2024-10 (shipped) origin is
EXCLUDED from selection and reported separately. All data was exposed during
development - every number here is development robustness, not a holdout claim.
Variants: HGBC shipped | HGBC regularized | Logistic L2 (all features) |
Logistic L2 derived-only (no market/commodity one-hots). Calibration: isotonic
on the calib window (same as shipped). No score chasing: report as-is.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

PKG = Path('/downloads/food/drive-pkg/egypt-food-prices-ml')
sys.path.insert(0, str(PKG / 'src'))
from features import SPIKE_THRESHOLD, build_features, load_series, model_frame, DERIVED_COLS

SELECTION = ['2016-10', '2018-10', '2020-10', '2022-10']
HELD_FROM_SELECTION = '2024-10'

def windows(feat, origin):
    y0 = pd.Timestamp(origin); y1 = y0 + pd.DateOffset(years=1); y2 = y0 + pd.DateOffset(years=2)
    ev = feat[feat['month'] >= y1] if origin == HELD_FROM_SELECTION else feat[(feat['month'] >= y1) & (feat['month'] < y2)]
    return feat[feat['month'] < y0], feat[(feat['month'] >= y0) & (feat['month'] < y1)], ev

def make_variant(name):
    if name == 'hgbc_shipped':
        return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.03, random_state=42), None
    if name == 'hgbc_regularized':
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, l2_regularization=1.0, max_leaf_nodes=15, random_state=42), None
    if name == 'logistic_all':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42)), None
    if name == 'logistic_derived_only':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42)), 'derived'
    raise ValueError(name)

def run(feat, origin, name):
    train_f, calib_f, eval_f = windows(feat, origin)
    tr, xcols_all = model_frame(train_f); ca, _ = model_frame(calib_f); ev, _ = model_frame(eval_f)
    clf, mode = make_variant(name)
    xcols = DERIVED_COLS if mode == 'derived' else xcols_all
    if len(ev) == 0 or ev['target_spike'].nunique() < 2: return None
    clf.fit(tr[xcols], tr['target_spike'].astype(int))
    raw_ca = clf.predict_proba(ca[xcols])[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip').fit(raw_ca, ca['target_spike'].astype(int))
    raw = clf.predict_proba(ev[xcols])[:, 1]; cal = iso.predict(raw)
    y = ev['target_spike'].astype(int).to_numpy()
    train_pos = float(tr['target_spike'].mean()); majority = int(train_pos >= 0.5)
    prev = (ev['r1'] > SPIKE_THRESHOLD).astype(int)
    acc_raw = accuracy_score(y, (raw >= 0.5).astype(int))
    return {'origin': origin, 'variant': name,
            'auc_uncalibrated': round(roc_auc_score(y, raw), 3),
            'auc_calibrated': round(roc_auc_score(y, cal), 3),
            'accuracy_raw05': round(acc_raw, 3),
            'f1_raw05': round(f1_score(y, (raw >= 0.5).astype(int)), 3),
            'brier_raw': round(brier_score_loss(y, raw), 4),
            'brier_calibrated': round(brier_score_loss(y, cal), 4),
            'majority_accuracy': round(accuracy_score(y, [majority]*len(y)), 3),
            'prev_month_spike_accuracy': round(accuracy_score(y, prev), 3),
            'beats_majority_raw05': bool(acc_raw > accuracy_score(y, [majority]*len(y))),
            'beats_prev_raw05': bool(acc_raw > accuracy_score(y, prev))}

def main():
    df = load_series(PKG / 'data' / 'data' / 'wfp_food_prices_egy.csv') if False else load_series(PKG / 'data' / 'wfp_food_prices_egy.csv')
    feat = build_features(df)
    variants = ['hgbc_shipped', 'hgbc_regularized', 'logistic_all', 'logistic_derived_only']
    rows = []
    for o in SELECTION + [HELD_FROM_SELECTION]:
        for v in variants:
            r = run(feat, o, v)
            if r: rows.append(r)
    sel = [r for r in rows if r['origin'] in SELECTION]
    summary = {}
    for v in variants:
        rs = [r for r in sel if r['variant'] == v]
        summary[v] = {'mean_auc_selection_origins': round(float(np.mean([r['auc_uncalibrated'] for r in rs])), 3),
                      'min_auc_selection_origins': min(r['auc_uncalibrated'] for r in rs),
                      'origins_beating_majority': sum(r['beats_majority_raw05'] for r in rs),
                      'origins_beating_prev': sum(r['beats_prev_raw05'] for r in rs),
                      'mean_brier_calibrated': round(float(np.mean([r['brier_calibrated'] for r in rs])), 4)}
    report = {'experiment': 'baseline/variant comparison, rolling origins (development robustness)',
              'date': '2026-11-04',
              'scope': 'All origins development-exposed; 2024-10 EXCLUDED from selection and reported separately. No untouched-holdout claim. Negative findings reported as-is.',
              'selection_summary': summary,
              'per_origin': rows,
              'held_from_selection': [r for r in rows if r['origin'] == HELD_FROM_SELECTION]}
    Path('/downloads/food/strengthen/baseline_compare.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=1))

if __name__ == '__main__':
    main()
