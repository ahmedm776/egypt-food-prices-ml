# Live source re-verification (2026-11-06)

## What was done
The shipped series were re-checked against WFP's own live export on 2026-11-06,
independent of the original download used during development.

Source (public, no login):
- Explorer UI: https://dataviz.vam.wfp.org/data-explorer/economic/prices
- Export endpoint the UI calls: POST https://api.vam.wfp.org/economicExplorer/Prices/PricesExport
- Request body used (the exact body the site's own Export button sends):
  {"adm0Code":[40765],"marketID":[10254],
   "commodityID":[52,108,114,172,173,180,385,434,728,758,868,875,1301,1302,1303,1350,1351],
   "priceTypeID":15,"startDate":"2006-09-12","endDate":"2031-10-12","timeAggregation":1}
  = Egypt / Cairo (national average) / the 17 ledger commodities / Retail / monthly.

Export received: 864 rows. 768 "Aggregated" (actual) rows, monthly 2021-01 to
2026-09, data source CAPMAS, plus 96 "Forecast" rows (WFP's own model, into 2027).
Export file sha256: f89fc0e5b3e855bb38a5937fc6c46401a0d13f8f55d11ee963dc4f2409393e4c
(export kept out of this package; it is re-fetchable from the source above).

## Result
All 768 overlapping (commodity, month) actual values in the export match
data/wfp_food_prices_egy.csv EXACTLY (zero mismatches) for the 17 ledger
commodities, Cairo (national average), Retail, 2021-01..2026-09.

## What this does and does not show
- This is provenance evidence: the package series agree with WFP's current
  official export. It is NOT independent predictive validation, because these
  are the same historical values the model was developed on (all
  development-exposed, per PROTOCOL_FREEZE.md).
- The WFP export contains NO October 2026 actual rows as of 2026-11-06. This
  shows only that this source has none; it does not prove October actuals do
  not exist anywhere, and it does not prove other WFP surfaces (e.g. HAPI)
  cannot differ.
- CAPMAS published its October 2026 CPI release on 2026-11-05 (headline CPI
  296.1 points, +0.6% m/m) per press coverage:
  https://en.amwalalghad.com/urgent-egypts-annual-inflation-eases-to-12-in-october-2026-capmas/
  (secondary source; the official CAPMAS bulletin page could not be retrieved
  directly during this check). When or whether WFP ingests October market
  prices is UNVERIFIED.
- The 96 forecast rows in the export are WFP's own model output. They are
  preserved only as part of the export record and will NEVER be used to score
  the ledger or any model metric. Scoring uses actual ("Aggregated") rows only.

## Relation to the 2026-10 ledger
prospective/ledger_2026-10_late-issued.json was frozen 2026-11-05 for target
month 2026-10, after September data was already out: it is LATE-ISSUED, not
strictly prospective. If October actuals are later obtained from a source whose
equivalence to this package's series is verified (as done here for
2021-01..2026-09), scoring it is an unseen-outcome evaluation (outcomes not
observed at issuance), with n=17 (small, low power). It is not an untouched
prospective test and the issuance date is not adjusted after the fact.

## Test suite
44/44 tests pass on this package state (pytest tests/, 2026-11-06).
