"""Chronological train, validation, and test split of the buoy record."""

from __future__ import annotations

import math
import numbers
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

TRAIN_START = date(1980, 3, 7)
TRAIN_END = date(1994, 12, 31)
VALIDATION_START = date(1995, 1, 1)
VALIDATION_END = date(1996, 12, 31)
TEST_START = date(1997, 1, 1)
TEST_END = date(1998, 6, 23)
N_SPLITS = 5

_INT_TEXT = re.compile(r"[+-]?\d+")


@dataclass(frozen=True)
class Cutoffs:
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date
    n_splits: int


DEFAULT_CUTOFFS = Cutoffs(
    train_start=TRAIN_START,
    train_end=TRAIN_END,
    validation_start=VALIDATION_START,
    validation_end=VALIDATION_END,
    test_start=TEST_START,
    test_end=TEST_END,
    n_splits=N_SPLITS,
)


@dataclass(frozen=True)
class FoldRange:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    folds: list[FoldRange]


def split_record(frame, cutoffs=None):
    """Assign each row to train, validation, or test by its calendar date."""
    resolved = _cutoffs(cutoffs)
    dates = [
        _row_date(index, year, month, day)
        for index, year, month, day in zip(
            frame.index, frame["year"], frame["month"], frame["day"], strict=True
        )
    ]
    labels = [
        _partition(day, resolved, index) for index, day in zip(frame.index, dates, strict=True)
    ]
    labeled = frame.copy()
    labeled["split"] = labels
    labeled["_sort_date"] = dates
    labeled["_sort_lat"] = [_text(value) for value in frame["latitude"]]
    labeled["_sort_lon"] = [_text(value) for value in frame["longitude"]]
    labeled = labeled.sort_values(
        ["_sort_date", "_sort_lat", "_sort_lon"], kind="mergesort"
    ).drop(columns=["_sort_date", "_sort_lat", "_sort_lon"])
    train_dates = sorted(
        {day for day, label in zip(dates, labels, strict=True) if label == "train"}
    )
    return SplitResult(
        train=_take(labeled, "train"),
        validation=_take(labeled, "validation"),
        test=_take(labeled, "test"),
        folds=_fold_ranges(train_dates, resolved.n_splits),
    )


def _fold_ranges(dates, n_splits):
    """Fold the sorted distinct training dates.

    sklearn TimeSeriesSplit is applied to that date list. The spec's index
    arithmetic is the contract when the library endpoints differ.
    """
    arithmetic = _arithmetic_folds(dates, n_splits)
    sklearn_folds = _sklearn_folds(dates, n_splits)
    if sklearn_folds == arithmetic:
        return sklearn_folds
    return arithmetic


def _arithmetic_folds(dates, n_splits):
    n_dates = len(dates)
    n_folds = n_splits + 1
    if n_dates < n_folds:
        return []
    fold_size = n_dates // n_folds
    remainder = n_dates % n_folds
    if fold_size == 0:
        return []
    folds = []
    for k in range(n_splits):
        test_start_index = remainder + (k + 1) * fold_size
        test_end_index = test_start_index + fold_size - 1
        folds.append(
            FoldRange(
                train_start=dates[0],
                train_end=dates[test_start_index - 1],
                test_start=dates[test_start_index],
                test_end=dates[test_end_index],
            )
        )
    return folds


def _sklearn_folds(dates, n_splits):
    if len(dates) <= n_splits:
        return []
    try:
        pairs = list(TimeSeriesSplit(n_splits=n_splits, gap=0).split(dates))
    except ValueError:
        return []
    return [
        FoldRange(
            train_start=dates[int(train_idx[0])],
            train_end=dates[int(train_idx[-1])],
            test_start=dates[int(test_idx[0])],
            test_end=dates[int(test_idx[-1])],
        )
        for train_idx, test_idx in pairs
    ]


def _take(labeled, name):
    return labeled.loc[labeled["split"] == name].reset_index(drop=True)


def _text(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
        if math.isnan(number):
            return ""
    return str(value)


def _cutoffs(cutoffs):
    if cutoffs is None:
        return DEFAULT_CUTOFFS
    if isinstance(cutoffs, dict):
        return Cutoffs(**cutoffs)
    return cutoffs


def _integer_cell(value, index, field):
    if isinstance(value, bool) or value is None:
        raise ValueError(f"row {index}: {field} {value!r} is not an integer")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"row {index}: {field} {value!r} is not an integer")
        return int(number)
    text = str(value).strip()
    if _INT_TEXT.fullmatch(text) is None:
        raise ValueError(f"row {index}: {field} {value!r} is not an integer")
    return int(text)


def _calendar_year(year, index):
    if 80 <= year <= 98:
        return 1900 + year
    if 1980 <= year <= 1998:
        return year
    raise ValueError(f"row {index}: year {year} is outside 80-98 and 1980-1998")


def _row_date(index, year, month, day):
    year_int = _integer_cell(year, index, "year")
    month_int = _integer_cell(month, index, "month")
    day_int = _integer_cell(day, index, "day")
    calendar_year = _calendar_year(year_int, index)
    try:
        return date(calendar_year, month_int, day_int)
    except ValueError as exc:
        raise ValueError(
            f"row {index}: year {year_int}, month {month_int}, day {day_int} is not a real date"
        ) from exc


def _partition(day, cutoffs, index):
    if cutoffs.train_start <= day <= cutoffs.train_end:
        return "train"
    if cutoffs.validation_start <= day <= cutoffs.validation_end:
        return "validation"
    if cutoffs.test_start <= day <= cutoffs.test_end:
        return "test"
    raise ValueError(
        f"row {index}: date {day.isoformat()} is outside the train, validation, and test ranges"
    )


def main():
    """Read the raw buoy table and write the three partition files."""
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data" / "el_nino_features.csv"
    output_dir = root / "data" / "processed"
    frame = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    result = split_record(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.train.to_csv(output_dir / "train.csv", index=False)
    result.validation.to_csv(output_dir / "validation.csv", index=False)
    result.test.to_csv(output_dir / "test.csv", index=False)


if __name__ == "__main__":
    main()
