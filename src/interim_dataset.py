"""Build a stand-in modelling table so the baselines and LightGBM can be coded now.

This is NOT the final dataset. Members 1-3 own the real target, cleaning and
features; replace data/interim/model_table.csv with their output when it lands.
It follows the same contract (see docs/baselines.md) and the agreed rules:

- three classes from SSTA: warming (> +0.5 C), neutral, cooling (< -0.5 C);
- SSTA = SST minus a site x calendar-month climatology fit on training rows only;
- the target is the class at the same site HORIZON_DAYS later, taken from the
  first reading within TARGET_TOLERANCE_DAYS after that date (else missing);
- features look backward only, inside one site.

Run from the repository root, after src/site_assignment.py:
    python -m src.interim_dataset
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.split import split_record

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "el_nino_features.csv"
ASSIGN_PATH = ROOT / "data" / "processed" / "site_assignments.csv"
OUT_PATH = ROOT / "data" / "interim" / "model_table.csv"

THRESHOLD_C = 0.5
HORIZON_DAYS = 90
TARGET_TOLERANCE_DAYS = 7
LAG_DAYS = (1, 7, 30)
WINDOWS = ("7D", "30D", "90D")
WEATHER = ["zon_winds", "mer_winds", "air_temp", "humidity"]


def risk_class(ssta, threshold=THRESHOLD_C):
    """Map anomalies to warming / neutral / cooling; missing stays missing."""
    out = pd.Series(pd.NA, index=ssta.index, dtype="string")
    out[ssta > threshold] = "warming"
    out[ssta < -threshold] = "cooling"
    out[ssta.between(-threshold, threshold)] = "neutral"
    return out


def load_sited_rows():
    raw = pd.read_csv(RAW_PATH)
    raw.insert(0, "obs_id", range(len(raw)))
    sites = pd.read_csv(ASSIGN_PATH, usecols=["obs_id", "site_id", "nominal_lat", "nominal_lon", "use_for_climatology"])
    df = raw.merge(sites, on="obs_id", how="left", validate="one_to_one")
    assert len(df) == len(raw)
    parts = split_record(df)
    df = pd.concat([parts.train, parts.validation, parts.test], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"].astype(str).str.zfill(6), format="%y%m%d")
    return df[df["site_id"].notna()].sort_values(["site_id", "date"], ignore_index=True)


def add_anomaly(df):
    fit = df[(df["split"] == "train") & df["use_for_climatology"]]
    clim = fit.groupby(["site_id", "month"])["ss_temp"].mean().rename("sst_climatology")
    df = df.join(clim, on=["site_id", "month"])
    df["ssta"] = df["ss_temp"] - df["sst_climatology"]
    df["current_class"] = risk_class(df["ssta"])
    return df


def add_target(df, horizon_days=HORIZON_DAYS, tolerance_days=TARGET_TOLERANCE_DAYS):
    """Class at the first reading in [date + horizon, date + horizon + tolerance]."""
    df = df.copy()
    df["target_date"] = df["date"] + pd.Timedelta(days=horizon_days)
    future = df.loc[df["current_class"].notna(), ["site_id", "date", "current_class"]]
    future = future.rename(columns={"date": "observed_target_date", "current_class": "target_class"})
    matched = pd.merge_asof(
        df[["site_id", "target_date"]].reset_index().sort_values("target_date"),
        future.sort_values("observed_target_date"),
        left_on="target_date", right_on="observed_target_date", by="site_id",
        direction="forward", tolerance=pd.Timedelta(days=tolerance_days),
    ).set_index("index").sort_index()
    df["observed_target_date"] = matched["observed_target_date"]
    df["target_class"] = matched["target_class"].astype("string")
    return df


def add_history(df):
    """Backward-looking lags, rolling means and trends, per site."""
    df = df.copy()
    base = df[["site_id", "date", "ssta"] + WEATHER]
    for lag in LAG_DAYS:
        shifted = base.assign(date=base["date"] + pd.Timedelta(days=lag))
        shifted = shifted.rename(columns={c: f"{c}_lag{lag}" for c in ["ssta"] + WEATHER})
        cols = ["site_id", "date", f"ssta_lag{lag}"] + ([f"{c}_lag{lag}" for c in WEATHER] if lag == 1 else [])
        df = df.merge(shifted[cols], on=["site_id", "date"], how="left")
    by_site = df.set_index("date").groupby("site_id", sort=False)
    for window in WINDOWS:
        for col in ["ssta", "zon_winds"]:
            rolled = by_site[col].rolling(window, min_periods=1).mean()
            df[f"{col}_mean_{window.lower()}"] = rolled.to_numpy()
    df["ssta_trend_30d"] = df["ssta"] - df["ssta_lag30"]
    df["ssta_trend_7d"] = df["ssta"] - df["ssta_lag7"]
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def build():
    df = add_history(add_target(add_anomaly(load_sited_rows())))
    drop = ["obs_id", "year", "month", "day", "latitude", "longitude", "use_for_climatology", "ss_temp", "sst_climatology"]
    return df.drop(columns=drop).sort_values(["date", "site_id"], ignore_index=True)


def main():
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {OUT_PATH}: {len(df):,} rows, {df['target_class'].notna().sum():,} with a target")
    print(pd.crosstab(df["split"], df["target_class"].fillna("missing"), margins=True))


if __name__ == "__main__":
    main()
