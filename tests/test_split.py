import unittest
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.split import DEFAULT_CUTOFFS, split_record


class SplitRecordTest(unittest.TestCase):
    def test_reversed_cutoff_ranges_raise(self):
        for partition in ("train", "validation", "test"):
            start = f"{partition}_start"
            end = f"{partition}_end"
            cutoffs = replace(
                DEFAULT_CUTOFFS,
                **{start: getattr(DEFAULT_CUTOFFS, end) + timedelta(days=1)},
            )
            with self.subTest(partition=partition):
                with self.assertRaisesRegex(ValueError, f"{start}.*{end}"):
                    split_record(
                        pd.DataFrame(columns=["year", "month", "day", "latitude", "longitude"]),
                        cutoffs,
                    )

    def test_overlapping_or_out_of_order_cutoffs_raise(self):
        cases = [
            ({"validation_start": date(1994, 12, 31)}, "train_end.*validation_start"),
            ({"validation_start": date(1994, 12, 30)}, "train_end.*validation_start"),
            ({"test_start": date(1996, 12, 31)}, "validation_end.*test_start"),
            ({"test_start": date(1996, 12, 30)}, "validation_end.*test_start"),
            (
                {"validation_start": date(1980, 1, 1), "validation_end": date(1980, 2, 1)},
                "train_end.*validation_start",
            ),
            (
                {"test_start": date(1993, 1, 1), "test_end": date(1993, 12, 31)},
                "validation_end.*test_start",
            ),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    split_record(
                        pd.DataFrame(columns=["year", "month", "day", "latitude", "longitude"]),
                        replace(DEFAULT_CUTOFFS, **changes),
                    )

    def test_cutoff_endpoints_must_be_calendar_dates(self):
        for partition in ("train", "validation", "test"):
            for endpoint in ("start", "end"):
                field = f"{partition}_{endpoint}"
                for value in (None, "1994-12-31", datetime(1994, 12, 31, 12)):
                    with self.subTest(field=field, value=value):
                        with self.assertRaisesRegex(ValueError, field):
                            split_record(
                                pd.DataFrame(
                                    columns=["year", "month", "day", "latitude", "longitude"]
                                ),
                                replace(DEFAULT_CUTOFFS, **{field: value}),
                            )

    def test_custom_single_day_ranges_and_gaps(self):
        cutoffs = replace(
            DEFAULT_CUTOFFS,
            train_start=date(1994, 6, 15),
            train_end=date(1994, 6, 15),
            validation_start=date(1994, 6, 17),
            validation_end=date(1994, 6, 17),
            test_start=date(1994, 6, 19),
            test_end=date(1994, 6, 19),
        )
        frame = pd.DataFrame(
            {
                "year": [94, 94, 94],
                "month": [6, 6, 6],
                "day": [19, 15, 17],
                "latitude": [0.0, 0.0, 0.0],
                "longitude": [-110.0, -110.0, -110.0],
            }
        )
        for config in (cutoffs, asdict(cutoffs)):
            with self.subTest(config_type=type(config).__name__):
                result = split_record(frame, config)
                for name, expected_day in (("train", 15), ("validation", 17), ("test", 19)):
                    partition = getattr(result, name)
                    self.assertEqual(partition["day"].tolist(), [expected_day])
                    self.assertEqual(partition["split"].tolist(), [name])
                self.assertEqual(result.folds, [])

        with self.assertRaisesRegex(ValueError, "1994-06-16.*outside"):
            split_record(frame.iloc[[0]].assign(day=16), cutoffs)

    def test_year_1994_is_training_and_year_79_raises(self):
        row = {
            "year": 1994,
            "month": 6,
            "day": 15,
            "latitude": "0.00",
            "longitude": "-110.00",
            "ss_temp": "26.24",
        }
        result = split_record(pd.DataFrame([row]))
        self.assertEqual(len(result.train), 1)
        self.assertEqual(result.train.iloc[0]["split"], "train")
        self.assertEqual(len(result.validation), 0)
        self.assertEqual(len(result.test), 0)

        bad = dict(row)
        bad["year"] = 79
        with self.assertRaises(ValueError) as caught:
            split_record(pd.DataFrame([bad]))
        message = str(caught.exception)
        self.assertIn("row 0", message)
        self.assertIn("79", message)

    def test_boundary_buoys_stay_together_and_reversed_input_matches(self):
        rows = [
            {
                "year": 94,
                "month": 12,
                "day": 31,
                "latitude": "1.00",
                "longitude": "-120.00",
                "ss_temp": "25.0",
            },
            {
                "year": 94,
                "month": 12,
                "day": 31,
                "latitude": "0.00",
                "longitude": "-110.00",
                "ss_temp": "26.0",
            },
            {
                "year": 95,
                "month": 1,
                "day": 1,
                "latitude": "0.50",
                "longitude": "-115.00",
                "ss_temp": "27.0",
            },
        ]
        frame = pd.DataFrame(rows)
        forward = split_record(frame)
        backward = split_record(frame.iloc[::-1].reset_index(drop=True))

        self.assertEqual(len(forward.train), 2)
        self.assertEqual(set(forward.train["split"]), {"train"})
        self.assertEqual(list(forward.validation["split"]), ["validation"])
        self.assertEqual(len(forward.test), 0)
        pd.testing.assert_frame_equal(forward.train, backward.train)
        pd.testing.assert_frame_equal(forward.validation, backward.validation)
        pd.testing.assert_frame_equal(forward.test, backward.test)

    def test_raw_file_partitions_and_fold_endpoints(self):
        raw_path = Path(__file__).resolve().parents[1] / "data" / "el_nino_features.csv"
        frame = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
        self.assertEqual(
            len(frame),
            178080,
            "raw file does not have 178080 rows; expected counts are not rescaled",
        )
        result = split_record(frame)
        expected_columns = list(frame.columns) + ["split"]
        self.assertEqual(list(result.train.columns), expected_columns)
        self.assertEqual(list(result.validation.columns), expected_columns)
        self.assertEqual(list(result.test.columns), expected_columns)
        self.assertEqual(set(result.train["split"]), {"train"})
        self.assertEqual(set(result.validation["split"]), {"validation"})
        self.assertEqual(set(result.test["split"]), {"test"})
        self.assertEqual(len(result.train), 101994)
        self.assertEqual(len(result.validation), 43772)
        self.assertEqual(len(result.test), 32314)

        train_days = _days(result.train)
        validation_days = _days(result.validation)
        test_days = _days(result.test)
        self.assertEqual(len(train_days), 5101)
        self.assertEqual(len(validation_days), 731)
        self.assertEqual(len(test_days), 539)
        self.assertEqual(max(train_days), date(1994, 12, 31))
        self.assertEqual(min(validation_days), date(1995, 1, 1))
        self.assertEqual(min(test_days), date(1997, 1, 1))
        self.assertEqual(train_days & validation_days, set())
        self.assertEqual(train_days & test_days, set())
        self.assertEqual(validation_days & test_days, set())

        train_blanks = _blank_count(result.train["ss_temp"])
        validation_blanks = _blank_count(result.validation["ss_temp"])
        test_blanks = _blank_count(result.test["ss_temp"])
        self.assertEqual(train_blanks, 13413)
        self.assertEqual(validation_blanks, 1841)
        self.assertEqual(test_blanks, 1753)
        self.assertEqual(train_blanks + validation_blanks + test_blanks, 17007)

        expected_folds = (
            (date(1980, 3, 7), date(1982, 11, 16), date(1982, 11, 17), date(1985, 9, 9)),
            (date(1980, 3, 7), date(1985, 9, 9), date(1985, 9, 10), date(1988, 1, 7)),
            (date(1980, 3, 7), date(1988, 1, 7), date(1988, 1, 8), date(1990, 5, 6)),
            (date(1980, 3, 7), date(1990, 5, 6), date(1990, 5, 7), date(1992, 9, 2)),
            (date(1980, 3, 7), date(1992, 9, 2), date(1992, 9, 3), date(1994, 12, 31)),
        )
        self.assertEqual(len(result.folds), 5)
        for fold, expected in zip(result.folds, expected_folds, strict=True):
            self.assertEqual(
                (fold.train_start, fold.train_end, fold.test_start, fold.test_end),
                expected,
            )


def _calendar_day(year, month, day):
    year = int(str(year).strip())
    month = int(str(month).strip())
    day = int(str(day).strip())
    if 80 <= year <= 98:
        year += 1900
    return date(year, month, day)


def _days(frame):
    return {
        _calendar_day(year, month, day)
        for year, month, day in zip(frame["year"], frame["month"], frame["day"], strict=True)
    }


def _blank_count(series):
    count = 0
    for value in series.tolist():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            count += 1
            continue
        if str(value).strip() == "":
            count += 1
    return count


if __name__ == "__main__":
    unittest.main()
