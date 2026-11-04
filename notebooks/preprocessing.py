import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "el_nino_features.csv"
ASSIGN_PATH = ROOT / "data" / "processed" / "site_assignments.csv"
THRESHOLD = 0.5

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.split import split_record


def create_target(ssta):
    """Convert a sea-surface-temperature anomaly into a risk class."""
    if pd.isna(ssta):
        return pd.NA
    if ssta <= -THRESHOLD:
        return "cooling"
    if ssta >= THRESHOLD:
        return "warming"
    return "normal"


def main():
    # The site-assignment script uses the raw row position as obs_id.
    raw = pd.read_csv(RAW_PATH)
    raw.insert(0, "obs_id", range(len(raw)))

    assignments = pd.read_csv(ASSIGN_PATH)
    df = raw.merge(
        assignments[["obs_id", "site_id", "use_for_climatology"]],
        on="obs_id",
        how="left",
        validate="one_to_one",
    )

    # Build the chronological partitions from the raw observations.
    split_result = split_record(df)
    df = pd.concat(
        [split_result.train, split_result.validation, split_result.test],
        ignore_index=True,
    )

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

    # Create a calendar date three months ahead. The observations are not
    # guaranteed to occur every day, so use the first available reading on or
    # after that target date within the same physical site.
    df["date"] = pd.to_datetime(df["date"], format="%y%m%d")
    df = df.sort_values(["site_id", "date"], kind="mergesort").reset_index(drop=True)
    df["_row_id"] = range(len(df))
    df["target_date"] = df["date"] + pd.DateOffset(months=3)

    future = df[["site_id", "date", "current_class"]].rename(
        columns={"current_class": "future_class"}
    )
    eligible = df["site_id"].notna()
    target_lookup = pd.merge_asof(
        df.loc[eligible, ["_row_id", "site_id", "target_date"]]
        .sort_values(["target_date", "site_id"]),
        future[future["site_id"].notna()]
        .sort_values(["date", "site_id"]),
        left_on="target_date",
        right_on="date",
        by="site_id",
        direction="forward",
    )
    df["target_class"] = pd.NA
    df.loc[target_lookup["_row_id"], "target_class"] = target_lookup[
        "future_class"
    ].to_numpy()
    df = df.drop(columns=["_row_id", "target_date"])

    # Keep missing labels missing in the complete dataframe, but exclude them
    # from any model training or scoring dataset.
    df["target_class"] = df["target_class"].astype("string")
    model_df = df.dropna(subset=["target_class"]).copy()

    missing_count = len(df) - len(model_df)
    print(f"Rows excluded from training/scoring due to missing target: {missing_count:,}")
    print(model_df["target_class"].value_counts())
    print(model_df["target_class"].value_counts(normalize=True).mul(100))
    return model_df


if __name__ == "__main__":
    main()
