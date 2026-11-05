"""Run: python -B -m unittest discover -s tests -v"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from build_el_nino_missing_handled import (  # noqa: E402
    COLUMNS, NOTEBOOK_FILLS, RULES, SOURCE, build_dataset, export_dataset,
    validate_dataset,
)


def fixture(dates, values):
    dates = pd.DatetimeIndex(dates)
    raw = pd.DataFrame({"year": dates.year - 1900, "month": dates.month,
                        "day": dates.day, "date": (dates.year - 1900) * 10000 + dates.month * 100 + dates.day,
                        "latitude": -0.02, "longitude": -109.46})
    for column in RULES:
        raw[column] = np.asarray(values, dtype=float)
    raw["ss_temp"] = 25.0
    return raw.reindex(columns=COLUMNS)


class MissingDataTests(unittest.TestCase):
    def test_elapsed_limits_and_measured_only_history(self):
        raw = fixture(pd.date_range("1980-01-01", periods=13), [10, 20, 30] + [np.nan] * 10)
        result, log = build_dataset(raw)
        self.assertEqual(result.loc[3, "air_temp"], 30)
        self.assertEqual(result.loc[4, "air_temp"], 20)
        self.assertEqual(result.loc[5, "air_temp"], 20)
        self.assertTrue(result.loc[6:, "air_temp"].isna().all())
        self.assertEqual(result.loc[3, "humidity"], 30)
        self.assertTrue(result.loc[4:, "humidity"].isna().all())
        validate_dataset(raw, result, log)

    def test_partial_day_excludes_other_same_day_readings(self):
        raw = fixture(["1980-01-01", "1980-01-01", "1980-01-02", "1980-01-02"],
                      [10, 14, 100, np.nan])
        result, log = build_dataset(raw)
        self.assertEqual(result.loc[3, "air_temp"], 12)
        self.assertEqual(result.loc[2, "air_temp"], 100)
        entry = log.loc[log.column.eq("air_temp")].iloc[0]
        self.assertEqual(entry.source_csv_line, 5)
        self.assertTrue(entry.partly_observed_day)
        self.assertEqual(entry.supporting_measurement_date, pd.Timestamp("1980-01-01"))
        self.assertEqual(entry.method, "last_area_day")

    def test_initial_missing_and_calendar_gaps(self):
        raw = fixture(["1980-01-01", "1980-01-03", "1980-01-05"], [np.nan, 50, np.nan])
        result, log = build_dataset(raw)
        self.assertTrue(pd.isna(result.loc[0, "air_temp"]))
        self.assertTrue(pd.isna(result.loc[2, "zon_winds"]))
        self.assertEqual(result.loc[2, "air_temp"], 50)
        self.assertEqual(log.loc[log.column.eq("air_temp"), "age_days"].iloc[0], 2)

    def test_current_and_future_values_cannot_change_past_fills(self):
        raw = fixture(["1980-01-01", "1980-01-02", "1980-01-02", "1980-01-03"],
                      [10, 100, np.nan, 200])
        changed = raw.copy()
        for column in RULES:
            changed.loc[[1, 3], column] = 999999
        before, _ = build_dataset(raw)
        after, _ = build_dataset(changed)
        for column in RULES:
            self.assertEqual(before.loc[2, column], after.loc[2, column])

    def test_sst_preserved_and_bad_donor_detected(self):
        raw = fixture(pd.date_range("1980-01-01", periods=2), [10, np.nan])
        raw.loc[1, "ss_temp"] = np.nan
        result, log = build_dataset(raw)
        self.assertTrue(pd.isna(result.loc[1, "ss_temp"]))
        log.loc[0, "supporting_measurement_date"] = log.loc[0, "observation_date"]
        with self.assertRaises(AssertionError):
            validate_dataset(raw, result, log)

    def test_saved_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, output, log, report = [folder / name for name in ["raw.csv", "out.csv", "log.csv", "report.json"]]
            fixture(pd.date_range("1980-01-01", periods=2), [10, np.nan]).to_csv(source, index=False)
            checks = export_dataset(source, output, log, report)
            self.assertTrue(checks["checks"]["saved_csv_and_log_rechecked"])
            self.assertTrue(checks["nan_free"])
            with self.assertRaises(FileExistsError):
                export_dataset(source, output, log, report)
            with self.assertRaises(ValueError):
                export_dataset(source, source, log, report)

    def test_empty_log_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "raw.csv"
            fixture(pd.date_range("1980-01-01", periods=2), [10, 20]).to_csv(source, index=False)
            report = export_dataset(source, folder / "out.csv", folder / "log.csv", folder / "report.json")
            self.assertEqual(report["filled_cells"], 0)

    def test_all_values_match_notebook_computation(self):
        """Independent transfer check without executing the statistical experiments."""
        notebook = json.loads((ROOT / "notebooks" / "missingness_audit.ipynb").read_text(encoding="utf-8"))
        history_cell = next(c for c in notebook["cells"] if c.get("id") == "refine-history")
        namespace = {"np": np, "pd": pd}
        exec("".join(history_cell["source"]), namespace)
        raw = pd.read_csv(SOURCE)
        result, log = build_dataset(raw)
        expected = raw.copy()
        work = raw.copy()
        work["lat"] = np.floor(work.latitude / 5).astype(int) * 5
        work["lon"] = np.floor(work.longitude.mod(360) / 10).astype(int) * 10
        work["day_full"] = pd.to_datetime({"year": work.year + 1900, "month": work.month, "day": work.day})
        calendar = pd.date_range(work.day_full.min(), work.day_full.max(), freq="D")
        notebook_methods = {"air_temp": "switch_7", "mer_winds": "switch_30", "zon_winds": "last", "humidity": "last"}
        for _, part in work.groupby(["lat", "lon"]):
            for column, rule in RULES.items():
                series = part.groupby("day_full")[column].median().reindex(calendar)
                past = namespace["strict_prior_history"](series, rule.history_days).reindex(pd.DatetimeIndex(part.day_full))
                past.index = part.index
                guess = namespace["choose_refined_value"](past["last"], past.middle, past.age, notebook_methods[column])
                use = part[column].isna() & past.age.between(1, rule.max_age_days) & guess.notna()
                expected.loc[part.index[use], column] = guess.loc[use]
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        checked = validate_dataset(raw, result, log, check_notebook=True)
        self.assertEqual(checked["fills_by_column"], NOTEBOOK_FILLS)
        self.assertEqual(checked["remaining_missing_cells"], 97431)


if __name__ == "__main__":
    unittest.main()
