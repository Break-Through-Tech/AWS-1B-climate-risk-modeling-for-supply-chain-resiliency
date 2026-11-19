import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "el_nino_missing_handled.csv"
ORIGINAL_PATH = ROOT / "data" / "el_nino_features.csv"
ASSIGN_PATH = ROOT / "data" / "processed" / "site_assignments.csv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.split import TEST_END, TRAIN_END, VALIDATION_END, split_record
from src.noaa_climatology import build_buoy_baseline


def create_target(ssta):
    """Convert an SSTA into El Niño, neutral, or La Niña."""
    if pd.isna(ssta):
        return pd.NA
    if ssta >= 0.5:
        return "el_nino"
    if ssta > -0.5:
        return "neutral"
    return "la_nina"


def load_observations(raw_path=RAW_PATH, assignment_path=ASSIGN_PATH):
    """Join filled observations to sites after checking raw-row alignment."""
    raw = pd.read_csv(raw_path)
    if raw_path == RAW_PATH:
        original = pd.read_csv(ORIGINAL_PATH)
        unchanged = ["year", "month", "day", "date", "latitude", "longitude", "ss_temp"]
        pd.testing.assert_frame_equal(
            raw[unchanged], original[unchanged], check_exact=True
        )
    raw.insert(0, "obs_id", range(len(raw)))

    assignments = pd.read_csv(assignment_path)
    df = raw.merge(
        assignments[["obs_id", "site_id", "use_for_climatology"]],
        on="obs_id",
        how="left",
        validate="one_to_one",
    )
    if len(df) != len(raw):
        raise ValueError(
            f"Site assignment merge changed the row count: {len(raw):,} -> {len(df):,}"
        )
    print(f"Rows without a site assignment: {df['site_id'].isna().sum():,}")
    return df


def add_splits(df):
    """Assign chronological train, validation, and test partitions."""
    split_result = split_record(df)
    return pd.concat(
        [split_result.train, split_result.validation, split_result.test],
        ignore_index=True,
    )


def add_daily_anomalies(df):
    """Calculate the original daily site/month climatology and classes."""
    # Fit the site/month climatology on training rows only, then apply it to
    # all rows. Rows excluded from climatology are still retained in df.
    train = df[
        (df["split"] == "train")
        & (df["use_for_climatology"] == True)  # noqa: E712
    ]
    climatology = (
        train.groupby(["site_id", "month"])["ss_temp"]
        .mean()
        .rename("clim_sst")
    )
    df = df.join(climatology, on=["site_id", "month"])

    df["ssta"] = df["ss_temp"] - df["clim_sst"]
    df["current_class"] = df["ssta"].apply(create_target)
    return df


def split_cutoff(df):
    """The latest target date allowed in each row's partition."""
    return df["split"].map(
        {
            "train": pd.Timestamp(TRAIN_END),
            "validation": pd.Timestamp(VALIDATION_END),
            "test": pd.Timestamp(TEST_END),
        }
    )


def add_future_daily_classes(df):
    """Keep the existing first-reading-at-least-three-months-later labels."""
    df = df.copy()
    # Create a calendar date three months ahead. The observations are not
    # guaranteed to occur every day, so use the first available reading on or
    # after that target date within the same physical site.
    df["date"] = pd.to_datetime(df["date"], format="%y%m%d")
    df = df.sort_values(["site_id", "date"], kind="mergesort").reset_index(drop=True)
    df["_row_id"] = range(len(df))
    df["target_date"] = df["date"] + pd.DateOffset(months=3)

    future = df[["site_id", "date", "current_class"]].rename(
        columns={"date": "label_date", "current_class": "future_class"}
    )
    eligible = df["site_id"].notna()
    target_lookup = pd.merge_asof(
        df.loc[eligible, ["_row_id", "site_id", "target_date"]]
        .sort_values(["target_date", "site_id"]),
        future[future["site_id"].notna()]
        .sort_values(["label_date", "site_id"]),
        left_on="target_date",
        right_on="label_date",
        by="site_id",
        direction="forward",
    )
    df["target_class"] = pd.NA
    df.loc[target_lookup["_row_id"], "target_class"] = target_lookup[
        "future_class"
    ].to_numpy()
    df["label_date"] = pd.NaT
    df.loc[target_lookup["_row_id"], "label_date"] = target_lookup[
        "label_date"
    ].to_numpy()

    # A label is usable only if its source observation was available by the
    # end of the row's partition. Keep the date for auditing the saved data.
    cutoff = split_cutoff(df)
    late_label = df["label_date"].gt(cutoff)
    df.loc[late_label, "target_class"] = pd.NA
    df = df.drop(columns=["_row_id", "target_date"])

    df["target_class"] = df["target_class"].astype("string")
    print(f"Rows with labels beyond their split cutoff: {late_label.sum():,}")
    return df


def aggregate_monthly(df):
    """Average available sensor readings within each site and calendar month."""
    source = df.copy()
    source["date"] = pd.to_datetime(source["date"], format="%y%m%d")
    source["month_start"] = source["date"].dt.to_period("M").dt.to_timestamp()
    sensors = ["ss_temp", "zon_winds", "mer_winds", "humidity", "air_temp"]
    monthly = source.groupby(
        ["site_id", "month_start", "split"], as_index=False
    ).agg(
        **{column: (column, "mean") for column in sensors},
        readings=("ss_temp", "count"),
    )
    monthly = monthly.rename(columns={"ss_temp": "monthly_sst"})
    monthly["month"] = monthly["month_start"].dt.month
    month_angle = 2 * np.pi * (monthly["month"] - 1) / 12
    monthly["month_sin"] = np.sin(month_angle)
    monthly["month_cos"] = np.cos(month_angle)
    monthly["year"] = monthly["month_start"].dt.year
    return monthly


def add_monthly_anomalies(monthly, observations, sst_baseline):
    """Use NOAA SST normals and training-only wind/humidity normals."""
    sensors = ["zon_winds", "mer_winds", "humidity"]
    baselines = {
        "zon_winds": "clim_zon_winds",
        "mer_winds": "clim_mer_winds",
        "humidity": "clim_humidity",
    }
    source = observations.loc[
        observations["split"].eq("train")
        & observations["use_for_climatology"].eq(True)  # noqa: E712
    ].copy()
    source["date"] = pd.to_datetime(source["date"], format="%y%m%d")
    source["month_start"] = source["date"].dt.to_period("M").dt.to_timestamp()
    training_months = source.groupby(
        ["site_id", "month_start", "month"], as_index=False
    )[sensors].mean()
    climatology = (
        training_months.groupby(["site_id", "month"], as_index=False)[sensors]
        .mean()
        .rename(columns=baselines)
    )
    result = monthly.merge(
        climatology, on=["site_id", "month"], how="left", validate="many_to_one"
    )
    result = result.merge(
        sst_baseline, on=["site_id", "month"], how="left", validate="many_to_one"
    )
    if result["clim_sst"].isna().any():
        missing = result.loc[result["clim_sst"].isna(), ["site_id", "month"]]
        raise ValueError(f"Missing NOAA SST baseline for {missing.drop_duplicates().to_dict('records')}")
    result["ssta"] = result["monthly_sst"] - result["clim_sst"]
    for sensor in ("zon_winds", "mer_winds", "humidity"):
        result[f"{sensor}_anomaly"] = result[sensor] - result[baselines[sensor]]
    result["current_class"] = result["ssta"].apply(create_target).astype("string")
    return result


def add_future_monthly_targets(monthly):
    """Match each site to its SSTA exactly three calendar months later."""
    source = monthly.copy()
    source["label_month"] = source["month_start"] + pd.DateOffset(months=3)
    future = source[["site_id", "month_start", "ssta"]].rename(
        columns={"month_start": "label_month", "ssta": "target_ssta"}
    )
    result = source.merge(
        future, on=["site_id", "label_month"], how="left", validate="many_to_one"
    )
    result.loc[
        result["label_month"].gt(split_cutoff(result)), "target_ssta"
    ] = float("nan")
    result["target_class"] = result["target_ssta"].apply(create_target).astype("string")
    return result


def main():
    observations = add_splits(load_observations())
    daily = add_future_daily_classes(add_daily_anomalies(observations))
    classification = daily.dropna(subset=["target_class"]).copy()
    classification.to_csv(ROOT / "data" / "processed" / "model_data_classification.csv", index=False)
    print(f"Daily classification rows: {len(classification):,}")

    monthly = add_future_monthly_targets(
        add_monthly_anomalies(
            aggregate_monthly(observations), observations, build_buoy_baseline()
        )
    )
    monthly_model = monthly.dropna(subset=["target_ssta"]).copy()
    monthly_model.to_csv(ROOT / "data" / "processed" / "model_data_monthly.csv", index=False)
    print(f"Monthly regression rows: {len(monthly_model):,}")
    return classification, monthly_model


if __name__ == "__main__":
    main()
