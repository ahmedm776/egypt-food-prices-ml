# Data source and license

- Dataset: Food Prices for Egypt, World Food Programme (WFP), distributed
  via the Humanitarian Data Exchange (HDX):
  https://data.humdata.org/dataset/wfp-food-prices-for-egypt
- License: Creative Commons Attribution for Intergovernmental Organisations
  (CC BY-IGO): https://data.humdata.org/about/license
- File: wfp_food_prices_egy.csv, 3,702 price records, Aug 2010 - Sep 2026.
- The CSV's first row after the header is an HDX metadata row; loaders in
  this project skip it (`skiprows=[1]`).
- Attribution: "Food Prices for Egypt, World Food Programme (WFP), via HDX."

## Before running

Download the CSV resource from the dataset page above, save it as
`data/wfp_food_prices_egy.csv`, and retain its metadata row. The app and full
test suite need that local file. The published evaluation used the frozen
3,702-record snapshot through September 2026; newer upstream data may change
fixture-specific results. No CSV is redistributed in this repository.
