"""Rolling-origin robustness check for the frozen v3 spike classifier.

PROVENANCE / SCOPE (honest labeling):
- Same frozen pipeline as src/train.py (HGBC max_iter=600 lr=0.03 seed42,
  isotonic calibration on a 12-month window, threshold 0.5, spike >2% MoM).
- Rolling origins: train < Y | calibrate Y..Y+12mo | evaluate Y+12mo..Y+24mo
  for Y in 2016-10, 2018-10, 2020-10, 2022-10, 2024-10 (last = shipped).
- Origins before 2024-10 were NOT the windows the pipeline was tuned on,
  but the whole dataset was exposed during development; this is a
  robustness check on development-exposed data, NOT an untouched holdout.
- Negative findings (origins where the model does not beat baselines) are
  reported as-is. Does not modify any shipped artifact.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

PKG = Path('/downloads/food/drive-pkg/egypt-food-prices-ml')
sys.path.insert(0, str(PKG / 'src'))
from features import SPIKE_THRESHOLD, build_features, load_series, model_frame

ORIGINS = ['2016-10', '2018-10', '2020-10', '2022-10', '2024-10']

def windows(feat, origin):
    y0 = pd.Timestamp(origin)
    y1 = y0 + pd.DateOffset(years=1)
    y2 = y0 + pd.DateOffset(years=2)
    return feat[feat['month'] < y0], feat[(feat['month'] >= y0) & (feat['month'] < y1)], feat[feat['month'] >= y1] if origin == '2024-10' else feat[(feat['month'] >= y1) & (feat['month'] < y2)]

def run_origin(feat, origin):
    train_f, calib_f, eval_f = windows(feat, origin)
    tr, xcols = model_frame(train_f); ca, _ = model_frame(calib_f); ev, _ = model_frame(eval_f)
    if len(ev) == 0 or ev['target_spike'].nunique() < 2:
        return {'origin': origin, 'skipped': 'eval window empty or single-class'}
    clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.03, random_state=42)
    clf.fit(tr[xcols], tr['target_spike'].astype(int))
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(clf.predict_proba(ca[xcols])[:, 1], ca['target_spike'].astype(int))
    raw = clf.predict_proba(ev[xcols])[:, 1]
    cal = iso.predict(raw)
    y = ev['target_spike'].astype(int).to_numpy()
    train_pos = float(tr['target_spike'].mean())
    majority = int(train_pos >= 0.5)
    prev = (ev['r1'] > SPIKE_THRESHOLD).astype(int)
    pred_raw = (raw >= 0.5).astype(int)
    out = {'origin': origin,
           'train_end': str(pd.Timestamp(origin).date()),
           'eval_months': [str(ev['month'].min().date()), str(ev['month'].max().date())],
           'n_train': int(len(tr)), 'n_calib': int(len(ca)), 'n_eval': int(len(ev)),
           'eval_base_rate': round(float(y.mean()), 3),
           'auc_uncalibrated': round(roc_auc_score(y, raw), 3),
           'auc_calibrated': round(roc_auc_score(y, cal), 3),
           'accuracy_raw_threshold_0p5': round(accuracy_score(y, pred_raw), 3),
           'f1_spike_class_raw_threshold_0p5': round(f1_score(y, pred_raw), 3),
           'accuracy_calibrated_threshold_0p5': round(accuracy_score(y, (cal >= 0.5).astype(int)), 3),
           'brier_calibrated': round(brier_score_loss(y, cal), 4),
           'majority_accuracy': round(accuracy_score(y, [majority]*len(y)), 3),
           'prev_month_spike_accuracy': round(accuracy_score(y, prev), 3),
           'beats_majority_raw05': bool(accuracy_score(y, pred_raw) > accuracy_score(y, [majority]*len(y)))}
    return out

def main():
    df = load_series(PKG / 'data' / 'wfp_food_prices_egy.csv')
    feat = build_features(df)
    results = [run_origin(feat, o) for o in ORIGINS]
    report = {
        'experiment': 'rolling-origin robustness check, frozen v3 pipeline',
        'date': '2026-11-04',
        'scope': ('Origins before 2024-10 were not the tuned windows, but the full dataset was '
                  'exposed during development; this is a robustness check on development-exposed '
                  'data, not an untouched holdout. Negative findings reported as-is.'),
        'protocol': 'train < Y | isotonic calibrate Y..Y+12mo | evaluate Y+12mo..Y+24mo (shipped 2024-10 origin evaluates >= 2025-10 to data end)',
        'results': results,
        'auc_range': [min(r['auc_calibrated'] for r in results if 'auc_calibrated' in r),
                      max(r['auc_calibrated'] for r in results if 'auc_calibrated' in r)],
        'origins_beating_majority_raw05': sum(1 for r in results if r.get('beats_majority_raw05')),
        'n_origins': len(results),
    }
    out = Path('/downloads/food/strengthen/rolling_origin_eval.json')
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['results'], indent=1))
    print('AUC range:', report['auc_range'], '| beats majority raw05:', report['origins_beating_majority_raw05'], '/', report['n_origins'])

if __name__ == '__main__':
    main()
