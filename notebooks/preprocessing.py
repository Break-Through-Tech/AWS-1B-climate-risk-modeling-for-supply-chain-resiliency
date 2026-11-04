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

    # Create a target three observations ahead within each physical site.
    df["date"] = pd.to_datetime(df["date"], format="%y%m%d")
    df = df.sort_values(["site_id", "date"], kind="mergesort")
    df["target_class"] = df.groupby("site_id")["current_class"].shift(-3)

    print(df["target_class"].value_counts(dropna=False))
    print(df["target_class"].value_counts(normalize=True, dropna=False).mul(100))
    return df


if __name__ == "__main__":
    main()
