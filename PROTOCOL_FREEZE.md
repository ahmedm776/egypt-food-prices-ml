# Protocol freeze log - egypt-food-prices-ml (2026-11-04)

Frozen facts about the shipped v3 pipeline (verified against canonical package code):

1. FEATURE CREATION (src/features.py): all features are calendar-exact lags
   k in {1,2,3,4,5,6,12} months via _lag_map, computed per (market, commodity)
   group on month-sorted series. r1/r2/r3, vol3/vol6, yoy, mom6, month_sin/cos,
   months_since_start all derive from lags >= 1 month only. A row for month m
   uses prices through m-1 only - no same-row lookahead.
2. LAG/GROUP PURGING: _lag_map looks up exact calendar months inside one
   series; missing months yield NaN (never a cross-series or misaligned
   value), and model_frame drops rows without price_lag_1. Series never share
   lag values.
3. TARGET: target_spike[m] = (price[m]/price[m-1] - 1 > 2%). NaN when
   price_lag_1 missing; dropped from fit/eval.
4. WINDOWS: train < 2024-10 (n=3159) | isotonic calibration 2024-10..2025-09
   (n=230, fit on raw scores of the train-only model) | evaluate >= 2025-10
   (n=215; retrospective development evaluation, the window was reused across versions - not an untouched final test). Every fit/calibration date strictly precedes every served date.
5. THRESHOLDING (diagnosed 2026-11-04): shipped accuracy/F1 threshold RAW
   scores at 0.5 (acc .767, F1 .390); thresholding CALIBRATED scores gives
   acc .786, F1 .361 on the same predictions. AUC .734 and Brier_cal .1445 are
   identical either way. metrics.json uses the raw-threshold numbers.
6. EXPOSURE: entire dataset was exposed during development (framing, model
   class, features, hyperparameters). All rolling-origin results (any year)
   are development robustness checks, never untouched-holdout claims.
   Variant selection must EXCLUDE the 2024-10 origin.
7. CALIBRATION NOTE: isotonic regression is non-decreasing; flat segments
   create ties that can lower measured AUC on windows whose score
   distribution differs from the calibration window. Observed drops
   (e.g. 2016 origin .720 -> .563) are consistent with tie effects;
   regime-shift causality is NOT proven by this alone.
