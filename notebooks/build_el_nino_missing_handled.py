"""Export the tested missing-data rules without running the audit notebook.

Cleaning summary
----------------
Use measured history from the same temporary geographic area (5 degrees latitude
by 10 degrees longitude). Each area/day is represented by its measured median
(middle value), so "last area day" is not an individual buoy's last reading.

- Zonal wind and humidity: use the last area day only if it is one day old.
- Air temperature: use the last area day at age one; at ages two/three use the
  median of seven days of measured history ending on that last measured day.
- Meridional wind: the same rule, with 30 days of measured history instead.
- Leave gaps missing when measured history is absent or older than these limits.
  Keep missing SST unchanged because it may define training labels. Never use
  same-day/future measurements or previous estimates as evidence for a fill.

These working limits came from recovery tests; they do not prove better forecast
accuracy. Remaining input NaNs can be handled by models such as LightGBM rather
than forcing estimates for long, unsupported absences.

Result for the original 178,080-row dataset: all 11 columns, row order, and
measured values are preserved. Missing cells fall from 151,330 to 97,431 after
53,899 replacements (35.6% of the original missing cells). Each replacement is
recorded separately; the output is deliberately not NaN-free.

Run from the repository root:
    python notebooks/build_el_nino_missing_handled.py --check-notebook

Outputs: the original 11 columns, a separate cell-level fill log, and a JSON
validation report. Existing files are never overwritten. Use a different
--output path for another run; log/report names follow that output by default.

Areas are temporary 5-degree latitude x 10-degree longitude bins, not buoy IDs.
Only original measured values from earlier dates support estimates. Unsupported
gaps and missing SST stay missing. This is missing-data preprocessing, not a
claim that the CSV is NaN-free or ready for every machine-learning estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "el_nino_features.csv"
OUTPUT = ROOT / "data" / "el_nino_missing_handled.csv"
COLUMNS = ["year", "month", "day", "date", "latitude", "longitude",
           "zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]
SENSORS = ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]
LOG_COLUMNS = ["source_row", "source_csv_line", "column", "was_missing",
               "replacement_value", "observation_date", "method", "rule",
               "supporting_measurement_date", "age_days", "history_days",
               "history_window_start", "history_window_end", "supporting_days",
               "lat_band", "lon_east_band", "partly_observed_day"]
NOTEBOOK_FILLS = {"zon_winds": 12762, "mer_winds": 13635, "humidity": 16282,
                  "air_temp": 11220, "ss_temp": 0}


@dataclass(frozen=True)
class FillRule:
    method: str
    history_days: int
    max_age_days: int


# Freeze the selected rules; do not reselect methods on each export.
RULES = {
    "air_temp": FillRule("last_then_median", 7, 3),
    "mer_winds": FillRule("last_then_median", 30, 3),
    "zon_winds": FillRule("last", 30, 1),
    "humidity": FillRule("last", 30, 1),
}


def source_dates(raw: pd.DataFrame) -> pd.Series:
    """Validate this dataset's schema and turn two-digit years into dates."""
    if list(raw.columns) != COLUMNS or raw.empty:
        raise ValueError("Expected a nonempty dataset with the original 11 columns in order")
    if not raw.index.equals(pd.RangeIndex(len(raw))):
        raise ValueError("Expected source rows numbered consecutively from zero")
    if not all(pd.api.types.is_numeric_dtype(raw[c]) for c in COLUMNS):
        raise ValueError("All original fields must be numeric or missing")
    if np.isinf(raw.to_numpy(dtype=float)).any():
        raise ValueError("Infinite values are not supported")
    if raw[COLUMNS[:6]].isna().any().any():
        raise ValueError("Dates and coordinates must be present to support time-aware fills")
    if not raw.year.between(80, 98).all():
        raise ValueError("This source uses years 80-98, meaning 1980-1998")
    dates = pd.to_datetime({"year": raw.year + 1900, "month": raw.month,
                            "day": raw.day}, errors="raise")
    expected_date = raw.year * 10000 + raw.month * 100 + raw.day
    if not raw.date.eq(expected_date).all():
        raise ValueError("The source date field disagrees with year/month/day")
    if not raw.latitude.between(-90, 90).all() or not raw.longitude.between(-180, 180).all():
        raise ValueError("Coordinates fall outside the original longitude/latitude convention")
    return dates


def area_estimates(series: pd.Series, rule: FillRule) -> pd.DataFrame:
    """Estimate each day from earlier measured days only, including partial days."""
    dates = pd.Series(series.index, index=series.index)
    measured = series.notna()
    # Shift by a calendar day: even another measurement today cannot be a donor.
    donor = dates.where(measured).ffill().shift(1)
    last = series.ffill().shift(1)
    middle = (series.rolling(rule.history_days, min_periods=1).median()
              .where(measured).ffill().shift(1))
    support = (series.rolling(rule.history_days, min_periods=1).count()
               .where(measured).ffill().shift(1))
    age = (dates - donor).dt.days
    use_median = age.gt(1) if rule.method == "last_then_median" else pd.Series(False, index=series.index)
    if rule.method not in {"last", "last_then_median"}:
        raise ValueError(f"Unsupported method: {rule.method}")
    guess = last.where(~use_median, middle).where(age.between(1, rule.max_age_days))
    window_days = pd.Series(1, index=series.index).where(~use_median, rule.history_days)
    return pd.DataFrame({
        "estimate": guess, "supporting_measurement_date": donor, "age_days": age,
        "method": np.where(use_median, "recent_area_median", "last_area_day"),
        "history_days": window_days,
        "history_window_start": donor - pd.to_timedelta(window_days - 1, unit="D"),
        "history_window_end": donor,
        "supporting_days": pd.Series(1, index=series.index).where(~use_median, support),
    })


def build_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a schema-preserving copy and a separate record of every changed cell."""
    dates = source_dates(raw)
    working = raw.copy()
    working["observation_date"] = dates
    working["lat_band"] = np.floor(raw.latitude / 5).astype(int) * 5
    working["lon_east_band"] = np.floor(raw.longitude.mod(360) / 10).astype(int) * 10
    keys = ["lat_band", "lon_east_band"]
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    result, logs = raw.copy(), []

    for area, part in working.groupby(keys):
        # This table is built once from original measurements, never filled values.
        measured_daily = part.groupby("observation_date")[list(RULES)].median().reindex(calendar)
        for column, rule in RULES.items():
            daily = measured_daily[column]
            estimates = area_estimates(daily, rule).reindex(pd.DatetimeIndex(part.observation_date)).copy()
            estimates.index = part.index
            use = part[column].isna() & estimates.estimate.notna()
            result.loc[part.index[use], column] = estimates.loc[use, "estimate"]
            if not use.any():
                continue
            log = estimates.loc[use].rename(columns={"estimate": "replacement_value"})
            log.insert(0, "source_row", log.index)
            log["source_csv_line"] = log.source_row + 2  # Header is line one.
            log["column"], log["was_missing"] = column, True
            log["observation_date"] = part.loc[use, "observation_date"]
            log["rule"] = rule.method
            log["lat_band"], log["lon_east_band"] = area
            log["partly_observed_day"] = part.loc[use, "observation_date"].map(daily.notna()).to_numpy()
            logs.append(log)

    audit = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=LOG_COLUMNS)
    for column in ["source_row", "source_csv_line", "age_days", "history_days", "supporting_days"]:
        audit[column] = audit[column].astype("int64")
    audit = audit.reindex(columns=LOG_COLUMNS).sort_values(["source_row", "column"]).reset_index(drop=True)
    return result, audit


def validate_dataset(raw: pd.DataFrame, result: pd.DataFrame, audit: pd.DataFrame,
                     check_notebook: bool = False) -> dict:
    """Fail on changed measurements, unexplained edits, or time/age violations."""
    dates = source_dates(raw)
    pd.testing.assert_index_equal(result.columns, raw.columns)
    pd.testing.assert_index_equal(result.index, raw.index)
    pd.testing.assert_frame_equal(result[COLUMNS[:6]], raw[COLUMNS[:6]], check_exact=True)
    for column in COLUMNS:
        observed = raw[column].notna()
        pd.testing.assert_series_equal(result.loc[observed, column], raw.loc[observed, column], check_exact=True)
    pd.testing.assert_series_equal(result.ss_temp, raw.ss_temp, check_exact=True)
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise AssertionError("Infinite values in output")
    if audit.duplicated(["source_row", "column"]).any():
        raise AssertionError("More than one audit entry for a changed cell")

    fills = {}
    for column in SENSORS:
        changed = raw[column].isna() & result[column].notna()
        entries = audit.loc[audit.column.eq(column)]
        if set(entries.source_row) != set(raw.index[changed]):
            raise AssertionError(f"Audit entries do not match replacements in {column}")
        if len(entries):
            rows = entries.source_row.to_numpy(dtype=int)
            np.testing.assert_array_equal(entries.replacement_value.to_numpy(), result.loc[rows, column].to_numpy())
            if not entries.observation_date.reset_index(drop=True).equals(dates.loc[rows].reset_index(drop=True)):
                raise AssertionError("Logged observation dates do not match source rows")
            rule = RULES[column]
            if not ((entries.supporting_measurement_date < entries.observation_date).all()
                    and entries.age_days.between(1, rule.max_age_days).all()
                    and (entries.history_window_end < entries.observation_date).all()):
                raise AssertionError("A fill uses a same-day/future donor or stale history")
            expected_age = (entries.observation_date - entries.supporting_measurement_date).dt.days
            np.testing.assert_array_equal(entries.age_days.to_numpy(), expected_age.to_numpy())
        fills[column] = int(changed.sum())

    if check_notebook and (len(raw) != 178080 or fills != NOTEBOOK_FILLS):
        raise AssertionError(f"Export differs from the notebook reference: rows={len(raw)}, fills={fills}")
    missing_before = {c: int(raw[c].isna().sum()) for c in COLUMNS}
    missing_after = {c: int(result[c].isna().sum()) for c in COLUMNS}
    remaining = sum(missing_after.values())
    return {
        "rows": len(result), "columns": list(result.columns), "rules": {c: asdict(r) for c, r in RULES.items()},
        "filled_cells": sum(fills.values()), "fills_by_column": fills,
        "missing_before_by_column": missing_before, "missing_after_by_column": missing_after,
        "remaining_missing_cells": remaining, "nan_free": remaining == 0,
        "directly_ready_for_nan_intolerant_model": remaining == 0,
        "checks": {"original_schema_and_row_order": True, "measured_values_unchanged": True,
                   "sst_unchanged": True, "all_replacements_logged": True,
                   "past_only_donor_dates_and_age_limits": True,
                   "no_infinite_values": True, "notebook_reference_checked": check_notebook},
        "training_note": "Remaining missing predictors require an explicit training-time policy. "
                         "Split chronologically before fitting any learned imputer. Do not impute SST labels.",
        "scope_note": "Only the tested missing-data treatment was applied; this does not certify "
                      "all possible data-quality issues or final model accuracy.",
        "area_note": "Temporary 5-degree latitude x 10-degree longitude bins are not buoy IDs.",
        "source_row_note": "Audit source_row is zero-based; source_csv_line includes the CSV header.",
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_dataset(source: Path, output: Path, log_path: Path, report_path: Path,
                   check_notebook: bool = False) -> dict:
    source, output, log_path, report_path = [p.resolve() for p in (source, output, log_path, report_path)]
    if len({source, output, log_path, report_path}) != 4:
        raise ValueError("Source, CSV, fill log, and report must be different files")
    for path in [output, log_path, report_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; select new output paths")
    source_hash = sha256(source)
    raw = pd.read_csv(source)
    result, audit = build_dataset(raw)
    report = validate_dataset(raw, result, audit, check_notebook)
    for path in [output, log_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, mode="x")
    audit.to_csv(log_path, index=False, mode="x", date_format="%Y-%m-%d")

    # Re-read the actual deliverables, not just their in-memory versions.
    saved = pd.read_csv(output, float_precision="round_trip")
    saved_audit = pd.read_csv(log_path, float_precision="round_trip", parse_dates=[
        "observation_date", "supporting_measurement_date", "history_window_start", "history_window_end"])
    pd.testing.assert_frame_equal(saved, result, check_exact=True)
    validate_dataset(raw, saved, saved_audit, check_notebook)
    if sha256(source) != source_hash:
        raise AssertionError("Source file changed during export")
    report["checks"].update(saved_csv_and_log_rechecked=True, source_file_unchanged=True)
    report.update(created_utc=datetime.now(timezone.utc).isoformat(),
                  source=str(source), output=str(output), fill_log=str(log_path),
                  source_sha256=source_hash, output_sha256=sha256(output), fill_log_sha256=sha256(log_path))
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--check-notebook", action="store_true", help="Require the recorded 178,080 rows and 53,899 fills")
    args = parser.parse_args()
    log = args.log or args.output.with_name(args.output.stem + "_fill_log.csv")
    report_path = args.report or args.output.with_name(args.output.stem + "_validation.json")
    report = export_dataset(args.input, args.output, log, report_path, args.check_notebook)
    print(f"Saved {report['rows']:,} rows; filled {report['filled_cells']:,} cells.")
    print(f"Remaining missing cells: {report['remaining_missing_cells']:,}; NaN-free: {report['nan_free']}.")
    print(pd.DataFrame({"filled": report["fills_by_column"], "missing_after": report["missing_after_by_column"]}).fillna(0).astype(int))
    print(f"Fill log: {log}\nValidation report: {report_path}")


if __name__ == "__main__":
    main()
