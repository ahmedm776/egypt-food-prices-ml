# LinkedIn project description (ready to paste)

**Egypt Food Price Spike Early Warning - End-to-End ML**

Built an end-to-end machine learning system that estimates the risk of a
food price spike in Egypt: for any of 47 commodity price series (WFP/HDX
data, 2010-2026), it predicts the probability that the average monthly
price rises more than 2% in the coming month.

What I did:

- Framed the problem honestly: initial experiments showed plain next-month
  price regression cannot beat a persistence baseline on this data, so I
  reframed it as spike-risk classification, where the model demonstrably
  adds value.
- Built a leakage-safe feature pipeline: calendar-exact price lags (1-12
  months), month-over-month ratios, rolling volatility, YoY momentum and
  seasonality - all strictly from past observations, with automated tests
  proving no future information leaks into features.
- Trained a gradient-boosted tree classifier (scikit-learn
  HistGradientBoostingClassifier) with a strict chronological protocol:
  trained on data through Sep 2024, isotonic-calibrated on Oct 2024 -
  Sep 2025, scored retrospectively on Oct 2025 - Sep 2026 (development-exposed data, no independent final test).
  Retrospective result: ROC AUC 0.734 vs 0.50 random, calibration improves
  Brier from 0.151 to 0.145. The whole dataset saw development exposure,
  so I claim no independent final test - and I report that accuracy does
  NOT beat trivial baselines, presenting the model as a calibrated
  risk-ranking tool, not an accuracy claim.
- Served the model through a FastAPI web app (web UI + JSON API) that
  attaches the model's training/calibration date cutoffs to every score,
  refuses as-of requests that would be in-sample (before Oct 2025), shows
  evaluation metrics and baselines next to each score, and includes a
  history-check mode that scores a past month and shows what actually
  happened.
- Wrote 30 automated tests covering the leakage guarantee, chronological
  window boundaries, artifact fit/calibration dates preceding served
  dates, prediction-feature alignment against real prices, in-sample
  refusal, and the API contract; fully reproducible with a pinned
  requirements.txt.

Tech: Python, scikit-learn, pandas, FastAPI, pytest, matplotlib.
Data: WFP Food Prices for Egypt via HDX (CC BY-IGO).
