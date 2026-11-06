"""Feature engineering for the Egypt food price spike early-warning model.

Task: binary classification - will a commodity's average monthly price rise
more than SPIKE_THRESHOLD (2%) in the next month?

Every feature for a target month t is built only from observations strictly
before t. Lags are calendar-exact: price_lag_k is the price exactly k months
before t and NaN when that month was not observed (the Cairo series has
gaps), so no information crosses the time boundary.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SPIKE_THRESHOLD = 0.02
LAG_KS = (1, 2, 3, 4, 5, 6, 9, 12)
DERIVED_COLS = [
    "r1", "r2", "r3", "vol3", "vol6", "yoy", "mom6",
    "month_sin", "month_cos", "months_since_start", "price_lag_1",
]
MARKETS = ["National Average", "Cairo (national average)"]


def load_series(csv_path):
    """One row per (market, commodity, month) with mean monthly price."""
    if not Path(csv_path).is_file():
        raise FileNotFoundError("WFP source CSV is not bundled. See data/DATA_SOURCE.md; place wfp_food_prices_egy.csv in data/ before running app or tests.")
    df = pd.read_csv(csv_path, skiprows=[1])
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].values.astype("datetime64[M]")
    df = df[df["market"].isin(MARKETS)]
    return (
        df.groupby(["market", "commodity", "month"], as_index=False)["price"]
        .mean()
        .sort_values(["market", "commodity", "month"])
        .reset_index(drop=True)
    )


def _lag_map(g, k):
    prices = dict(zip(g["month"], g["price"]))
    return g["month"].map(lambda m: prices.get(m - pd.offsets.MonthBegin(k), np.nan))


def add_series_features(g):
    """Add calendar-exact lags and derived features to one sorted series."""
    g = g.sort_values("month").copy()
    for k in LAG_KS:
        g[f"price_lag_{k}"] = _lag_map(g, k)
    g["r1"] = g["price_lag_1"] / g["price_lag_2"] - 1
    g["r2"] = g["price_lag_2"] / g["price_lag_3"] - 1
    g["r3"] = g["price_lag_3"] / g["price_lag_4"] - 1
    lag3 = g[[f"price_lag_{k}" for k in (1, 2, 3)]]
    lag6 = g[[f"price_lag_{k}" for k in (1, 2, 3, 4, 5, 6)]]
    g["vol3"] = lag3.std(axis=1) / lag3.mean(axis=1)
    g["vol6"] = lag6.std(axis=1) / lag6.mean(axis=1)
    g["yoy"] = g["price_lag_1"] / g["price_lag_12"] - 1
    g["mom6"] = g["price_lag_1"] / g["price_lag_6"] - 1
    g["month_sin"] = np.sin(2 * np.pi * g["month"].dt.month / 12)
    g["month_cos"] = np.cos(2 * np.pi * g["month"].dt.month / 12)
    g["months_since_start"] = (g["month"] - g["month"].min()).dt.days / 30.44
    return g


def build_features(df):
    """Full feature frame with one-hot market/commodity and target column."""
    parts = [add_series_features(g) for _, g in df.groupby(["market", "commodity"])]
    out = pd.concat(parts, ignore_index=True)
    out["target_spike"] = (out["price"] / out["price_lag_1"] - 1 > SPIKE_THRESHOLD).astype("float")
    out.loc[out["price_lag_1"].isna(), "target_spike"] = np.nan
    out = pd.get_dummies(out, columns=["market", "commodity"], prefix=["mkt", "com"], dtype=float)
    mkt_cols = [c for c in out.columns if c.startswith("mkt_")]
    com_cols = [c for c in out.columns if c.startswith("com_")]
    out["market"] = out[mkt_cols].idxmax(axis=1).str[4:]
    out["commodity"] = out[com_cols].idxmax(axis=1).str[4:]
    return out


CALIB_START = pd.Timestamp("2024-10-01")
EVAL_START = pd.Timestamp("2025-10-01")
# Earliest month the shipped pipeline can score as-of: the month after the
# calibration window ends. Earlier months would be scored in-sample
# (training or calibration data), which the app refuses.
EARLIEST_ASOF_MONTH = EVAL_START


def split_chronological(df_feat):
    """Chronological train -> calibration -> evaluation windows.

    train: months < 2024-10 (series A lives entirely here; series B 2021-2024)
    calib: 2024-10 .. 2025-09 (isotonic calibrator fit here, on the model
           trained only on `train`)
    eval:  months >= 2025-10 (retrospective evaluation of the frozen
           pipeline)

    HONESTY NOTE: the whole dataset was exposed during development (task
    framing, model class, features and hyperparameters were chosen after
    exploratory scoring of various windows). This is therefore a
    RETROSPECTIVE evaluation of a frozen pipeline - there is no
    independent final test on this dataset, and none is claimed.

    No purge gap is needed for a 1-month target: a row's label uses only
    prices at its own month and the previous one, both strictly before
    the window that starts after it.
    """
    train = df_feat[df_feat["month"] < CALIB_START]
    calib = df_feat[(df_feat["month"] >= CALIB_START) & (df_feat["month"] < EVAL_START)]
    evalw = df_feat[df_feat["month"] >= EVAL_START]
    return train, calib, evalw


def model_frame(df_feat):
    """Rows usable for fit/eval (need last month's price) + feature matrix."""
    dummies = [c for c in df_feat.columns if c.startswith(("mkt_", "com_"))]
    xcols = DERIVED_COLS + dummies
    usable = df_feat[df_feat["price_lag_1"].notna() & df_feat["target_spike"].notna()]
    return usable, xcols
