# Robustness evaluation (rolling origins) - honest limits

All numbers below are DEVELOPMENT ROBUSTNESS checks on data that was exposed
during development. The 2024-10 origin (the shipped window) was EXCLUDED from
variant selection. Nothing here is an untouched-holdout claim. Negative
findings are reported as-is. Raw artifacts: rolling_origin_eval.json,
baseline_compare.json, PROTOCOL_FREEZE.md (same folder).

## Protocol

Rolling origins: train < Y | isotonic calibration Y..Y+12mo | evaluate
Y+12..24mo, Y in {2016, 2018, 2020, 2022, 2024}-10. Same frozen feature
pipeline and thresholding convention (accuracy/F1 threshold the RAW score at
0.5; calibrated-threshold metrics are named separately in metrics.json).

## v3 (shipped) across origins - discrimination is time-unstable

| origin  | n_eval | AUC uncal | AUC cal | acc raw | majority | prev-month |
|---------|--------|-----------|---------|---------|----------|------------|
| 2016-10 | 228    | .720      | .563    | .715    | .825     | .833       |
| 2018-10 | 228    | .629      | .630    | .763    | .860     | .833       |
| 2020-10 | 396    | .511      | .519    | .518    | .525     | .556       |
| 2022-10 | 168    | .687      | .663    | .661    | .649     | .661       |
| 2024-10 | 215    | .734      | .724    | .767    | .800     | .805       |

- AUC swings .51-.73 by era; the 2020 origin is near-random.
- The shipped window is among the better origins - consistent with the
  disclosed development exposure, now quantified.
- Isotonic ties can lower measured AUC on shifted windows (2016: .720 ->
  .563); this is a tie effect, regime-shift causality is NOT claimed.

## v4 candidate (regularized: 200 iters, l2=1.0, 15 leaves) vs v3

Retrospective-window ranking baseline (v4, models/retrospective_eval_predictions.csv,
baseline_auc.py): paired on the 192/215 series-months with a previous month
for the same series, a prev-month-spike rule scores AUC 0.625 vs the model's
0.750 (calibrated, same subset) - the ranking signal is not trivially
temporal. The headline 0.759 AUC is on all 215 series-months and is not
compared directly against the 192-row baseline.
The majority baseline carries no ranking (constant score).

Selection used origins 2016-2022 ONLY. Per-origin (AUC uncal, v3 -> v4):
2016: .720 -> .695 (worse) | 2018: .629 -> .700 | 2020: .511 -> .529 |
2022: .687 -> .734 | 2024 (excluded from selection): .734 -> .759.
Mean AUC on selection origins .637 -> .664; min AUC .511 -> .529;
mean calibrated Brier .2391 -> .2218. v4 leads on 3 of 4 selection origins
and trails on 2016.

## What neither version beats

- No HGBC variant outperformed the prev-month-spike heuristic on accuracy
  at any origin tested (v3 and v4 alike); the best result is a tie (both
  variants, 2022 origin, .661 = .661). logistic_all beats it on 2 of 4
  selection origins but ranks worse on AUC and was not adopted.
- Majority beats the model on 3 of 4 selection origins for both HGBC variants.
- On the shipped window, no candidate beats prev-month (.805) or majority
  (.800) on accuracy; v4 raw acc is .795.

## Calibration note

On the shipped window, calibration improves Brier (.1412 raw -> .1380
calibrated) but calibrated-threshold accuracy/F1 (.800/.189) differs from
raw-threshold (.795/.353); both are named explicitly in metrics.json.
