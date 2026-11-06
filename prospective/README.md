# Late-issued October 2026 ledger (added 2026-11-05) - NOT prospective evidence

Everything else in this project is a retrospective, development-exposed evaluation.
This ledger was issued on 2026-11-05, after October 2026 had ended and after the model was
developed, so it is a LATE-ISSUED BACKFILL: hashes make later edits detectable but do not
show that October outcomes were unseen during development, and the data CSV simply ended in
Sep 2026. A genuinely prospective ledger needs a target month that has not started when the
ledger is made (the make_ledger.py script labels `evidence_class` automatically), with the
required previous-month input observation present. As of 2026-11-05 November has already begun,
so the earliest eligible target is December 2026, which needs November 2026 observations. Nothing here changes the model, task or
one-month horizon. `ledger_2026-10_late-issued.json` freezes the shipped, unchanged model's
calibrated spike-risk scores for that month, with SHA-256 hashes of the model file,
the data file, and the monthly rows the predictions used.

- Scoring uses the stored predictions and the stored as-of previous-month-spiked values; nothing is refit and the input cutoff (Sep 2026) is recorded. Predictions: 17 series (all "Cairo (national average)"). 30 series are skipped because
  they have no Sep 2026 observation, so the model cannot score them. n = 17 is small: one
  month of results will have low statistical power and is not a validation by itself.
- Scoring, once WFP publishes Oct 2026 data (no polling is set up):
  `python prospective/score_ledger.py prospective/ledger_2026-10_late-issued.json NEW_WFP.csv`
  It refuses to run if the ledger or model hash changed, flags revisions of earlier rows,
  and reports n, spike count, AUC (when both classes exist), Brier, accuracy at 0.5 vs
  the majority and previous-month-spiked baselines. Metrics were declared in the ledger
  before the outcome.
- Limits: the freeze time is self-attested (UTC string inside the file). It only becomes
  externally verifiable if the ledger hash is published or timestamped elsewhere; nothing
  was published. The WFP file may be revised later.
- Reproduce the ledger: `python prospective/make_ledger.py` (rewrites the file; the original
  ledger hash is 5cb98b4151355bb71efd1bd32d2670412bf24579b0f46bd08a9ffe95157b9003 and the
  timestamp changes on regeneration, so keep the frozen file).
- Tests: `tests/test_prospective.py` (5: hash integrity, target-month rule, no-outcome case,
  scoring mechanics on a fabricated row in a temporary copy only).
