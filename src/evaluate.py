"""Validate saved forecasts and publish reproducible forecast evaluations.

This module never fits a model. Real-data provenance records are declarations
backed by upstream review, not an automatic proof of scientific correctness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import C_GRID, CLASSES, RESERVED, score
from src.split import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {
    "train": (TRAIN_START, TRAIN_END),
    "validation": (VALIDATION_START, VALIDATION_END),
    "test": (TEST_START, TEST_END),
}
MODEL_NAMES = {"majority", "persistence", "logistic", "lightgbm"}
PROBA = [f"proba_{c}" for c in CLASSES]
KEYS = ["site_id", "date"]
REFERENCE_COLUMNS = set(
    KEYS
    + [
        "split",
        "target_date",
        "observed_target_date",
        "target_class",
        "current_class",
        "nominal_lat",
        "nominal_lon",
    ]
)
PREDICTION_COLUMNS = set(
    KEYS + ["model", "split", "target_date", "y_true", "y_pred"] + PROBA
)
ALIGNED_COLUMNS = [
    "model",
    "site_id",
    "date",
    "split",
    "target_date",
    "observed_target_date",
    "y_true",
    "y_pred",
    *PROBA,
]
TARGET = {
    "id": "local-ssta-3class-90d-tol7-v1",
    "classes": CLASSES,
    "threshold_c": 0.5,
    "threshold_policy": "strict_outer_inclusive_neutral",
    "horizon_unit": "days",
    "horizon_value": 90,
    "match_policy": "first_nonmissing_class_on_or_after",
    "tolerance_days": 7,
    "climatology_fit_splits": ["train"],
}
FOLDS = [
    ("1980-03-07", "1988-12-31", "1989-01-01", "1990-12-31"),
    ("1980-03-07", "1990-12-31", "1991-01-01", "1992-12-31"),
    ("1980-03-07", "1992-12-31", "1993-01-01", "1994-12-31"),
]
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
ISO = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


class EvaluationContractError(ValueError):
    """An expected input/dependency failure with a stable machine-readable code."""

    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def as_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}


def _fail(code, message, **details):
    raise EvaluationContractError(code, message, details)


def _fields(obj, required, where):
    if not isinstance(obj, dict):
        _fail("E_SCHEMA", f"{where} must be an object")
    missing, extra = set(required) - obj.keys(), obj.keys() - set(required)
    if missing or extra:
        _fail(
            "E_SCHEMA",
            f"{where} has incorrect fields",
            missing=sorted(missing),
            extra=sorted(extra),
        )


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        _fail("E_SCHEMA", f"{where} must be a nonempty string")


def _version(value, where):
    if type(value) is not int or value != 1:
        _fail("E_SCHEMA", f"{where}.schema_version must be integer 1")


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value, where):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _fail("E_SCHEMA", f"{where} must be a lowercase SHA-256 digest")


def _commit(value, synthetic, where):
    if synthetic and value == "synthetic_fixture":
        return
    if not isinstance(value, str) or not HEX40.fullmatch(value):
        _fail("E_SCHEMA", f"{where} must identify the actual source commit")


def _utc(value, where):
    try:
        if not isinstance(value, str):
            raise ValueError()
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
            raise ValueError()
        return result
    except (ValueError, AttributeError):
        _fail("E_SCHEMA", f"{where} must be a UTC timestamp with timezone")


def sha256_file(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail("E_DEPENDENCY", f"Cannot read required file: {path}", error=str(exc))
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    """Reject NaN and Infinity, which Python accepts but JSON does not."""
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(path):
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        _fail("E_DEPENDENCY", f"Cannot read required JSON: {path}", error=str(exc))
    except (ValueError, UnicodeError) as exc:
        _fail("E_SCHEMA", f"Invalid JSON: {path}", error=str(exc))


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_ref(ref, parent, where, artifact=False):
    _fields(
        ref,
        {"path", "sha256", "format", "num_iterations"}
        if artifact
        else {"path", "sha256"},
        where,
    )
    _text(ref["path"], f"{where}.path")
    _sha(ref["sha256"], f"{where}.sha256")
    path = (Path(parent) / ref["path"]).resolve()
    actual = sha256_file(path)
    if actual != ref["sha256"]:
        _fail("E_HASH", f"Hash mismatch: {path}", expected=ref["sha256"], actual=actual)
    return {**ref, "path": str(path)}


def _csv(path):
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            header = next(csv.reader(stream), [])
        if len(header) != len(set(header)):
            _fail("E_SCHEMA", f"Duplicate CSV column names: {path}")
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except OSError as exc:
        _fail("E_DEPENDENCY", f"Cannot read CSV: {path}", error=str(exc))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        _fail("E_SCHEMA", f"Invalid CSV: {path}", error=str(exc))


def _columns(frame, required, where, exact=False):
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        _fail("E_SCHEMA", f"{where} must be a DataFrame with distinct columns")
    missing = set(required) - set(frame.columns)
    extra = set(frame.columns) - set(required) if exact else set()
    if missing or extra:
        _fail(
            "E_SCHEMA",
            f"{where} has incorrect columns",
            missing=sorted(missing),
            extra=sorted(extra),
        )


def _missing(series):
    return series.isna() | series.eq("")


def _examples(frame, mask=None):
    chosen = frame if mask is None else frame.loc[mask]
    return {
        "count": len(chosen),
        "examples": chosen[KEYS].head(20).astype(str).to_dict("records"),
    }


def _dates(series, where, nullable=False):
    missing = _missing(series)
    valid_string = series.map(
        lambda value: isinstance(value, str) and bool(ISO.fullmatch(value))
    )
    bad = ~valid_string & ~missing if nullable else ~valid_string
    if bad.any():
        _fail(
            "E_SCHEMA",
            f"{where} requires exact ISO YYYY-MM-DD dates",
            count=int(bad.sum()),
        )
    try:
        return pd.to_datetime(series.mask(missing), format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError, OverflowError) as exc:
        _fail("E_SCHEMA", f"{where} contains an invalid calendar date", error=str(exc))


def _keys(frame, where):
    _columns(frame, KEYS, where)
    bad = frame.site_id.map(lambda v: not isinstance(v, str) or not v or v.strip() != v)
    if bad.any():
        _fail(
            "E_SCHEMA",
            f"{where} contains missing/whitespace site IDs",
            **_examples(frame, bad),
        )
    _dates(frame.date, f"{where}.date")
    duplicate = frame.duplicated(KEYS, keep=False)
    if duplicate.any():
        _fail(
            "E_DUPLICATE_KEY",
            f"{where} contains duplicate site/date keys",
            **_examples(frame, duplicate),
        )


def _keyset(frame):
    return set(frame[KEYS].itertuples(index=False, name=None))


def population_hash(frame):
    keys = frame[KEYS].drop_duplicates().sort_values(KEYS, kind="stable")
    return hashlib.sha256(
        keys.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _labels(series, where, nullable=False):
    bad = ~series.isin(CLASSES)
    if nullable:
        bad &= ~_missing(series)
    if bad.any():
        _fail(
            "E_LABEL",
            f"{where} contains unknown or missing classes",
            count=int(bad.sum()),
            values=sorted(set(series[bad].astype(str)))[:20],
        )


def validate_target_contract(contract, synthetic=None):
    _fields(contract, {*TARGET, "status", "builder_commit"}, "target_contract")
    for key, expected in TARGET.items():
        if contract[key] != expected or (
            key in {"horizon_value", "tolerance_days"}
            and type(contract[key]) is not int
        ):
            _fail(
                "E_TARGET_CONTRACT",
                f"Unsupported target contract field: {key}",
                expected=expected,
                actual=contract[key],
            )
    if not _finite_number(contract["threshold_c"]):
        _fail("E_TARGET_CONTRACT", "threshold_c must be numeric")
    if not isinstance(contract["status"], str) or contract["status"] not in {
        "interim",
        "team_final",
    }:
        _fail("E_TARGET_CONTRACT", "Invalid target status")
    # The pure interface has no dataset-kind argument; the run-level validator
    # supplies it and enforces the synthetic-only source-identity exception.
    _commit(
        contract["builder_commit"],
        synthetic is not False,
        "target_contract.builder_commit",
    )


def _prepare_reference(reference, contract, split):
    validate_target_contract(contract)
    if not isinstance(split, str) or split not in {"validation", "test"}:
        _fail("E_SPLIT", "Evaluation split must be validation or test")
    ref = reference.copy(deep=True).reset_index(drop=True)
    _columns(ref, REFERENCE_COLUMNS, "reference")
    _keys(ref, "reference")
    dates = _dates(ref.date, "reference.date")
    expected_split = pd.Series("", index=ref.index)
    for name, (start, end) in SPLITS.items():
        expected_split.loc[dates.between(pd.Timestamp(start), pd.Timestamp(end))] = name
    wrong = ref.split.ne(expected_split) | expected_split.eq("")
    if wrong.any():
        _fail(
            "E_SPLIT",
            "Reference split disagrees with issue date",
            **_examples(ref, wrong),
        )
    for column, limit in [("nominal_lat", 90), ("nominal_lon", 180)]:
        try:
            values = pd.to_numeric(ref[column], errors="raise")
        except (ValueError, TypeError):
            _fail("E_SCHEMA", f"Invalid {column}")
        if (
            not np.isfinite(values.to_numpy(dtype=float)).all()
            or values.abs().gt(limit).any()
        ):
            _fail("E_SCHEMA", f"{column} must be finite and within coordinate bounds")
        ref[column] = values
        if ref.groupby("site_id")[column].nunique().gt(1).any():
            _fail("E_SCHEMA", f"Canonical {column} varies within a site")
    active = ["train", "validation"] + (["test"] if split == "test" else [])
    part = ref[ref.split.isin(active)].copy()
    issue = _dates(part.date, "reference.date")
    requested = _dates(part.target_date, "reference.target_date")
    observed = _dates(
        part.observed_target_date, "reference.observed_target_date", nullable=True
    )
    if requested.ne(issue + pd.Timedelta(days=90)).any():
        _fail("E_TARGET_CONTRACT", "Requested target date is not issue date + 90 days")
    _labels(part.target_class, "reference.target_class", nullable=True)
    _labels(part.current_class, "reference.current_class", nullable=True)
    missing_label = _missing(part.target_class)
    paired = missing_label.eq(observed.isna())
    if not paired.all():
        _fail(
            "E_TARGET_PAIR",
            "Target class and observed date must be missing together",
            **_examples(part, ~paired),
        )
    outside = observed.notna() & (
        (observed < requested) | (observed > requested + pd.Timedelta(days=7))
    )
    if outside.any():
        _fail(
            "E_TARGET_WINDOW",
            "Observed target falls outside the allowed 90–97-day window",
            **_examples(part, outside),
        )
    ends = pd.to_datetime(
        part.split.map({name: str(bounds[1]) for name, bounds in SPLITS.items()})
    )
    conditions = {
        "requested_target_after_split_end": requested.gt(ends),
        "missing_target": missing_label,
        "observed_target_after_split_end": observed.gt(ends),
    }
    reason = pd.Series("", index=part.index)
    for label, mask in conditions.items():
        reason.loc[reason.eq("") & mask] = label
    part["_eligible"] = reason.eq("")
    eligible = part[part._eligible]
    donors = part.set_index(KEYS).current_class
    donor_index = pd.MultiIndex.from_arrays(
        [eligible.site_id, eligible.observed_target_date], names=KEYS
    )
    donor_classes = donors.reindex(donor_index).reset_index(drop=True)
    mismatch = (
        donor_classes.ne(eligible.target_class.reset_index(drop=True))
        | donor_classes.isna()
    )
    if mismatch.any():
        _fail(
            "E_TARGET_PAIR",
            "Eligible target does not match a same-site observed donor class",
            **_examples(eligible.reset_index(drop=True), mismatch),
        )
    exclusions = part.loc[~part._eligible, ["split", *KEYS]].copy()
    exclusions["reason"] = reason.loc[~part._eligible]
    counts = {}
    for name in active:
        mask = part.split.eq(name)
        n = int(mask.sum())
        en = int((mask & part._eligible).sum())
        counts[name] = {
            "reference_n": n,
            "eligible_n": en,
            "excluded_n": n - en,
            "primary_exclusions": {
                label: int((mask & reason.eq(label)).sum()) for label in conditions
            },
            "condition_counts_non_additive": {
                label: int((mask & cond).sum()) for label, cond in conditions.items()
            },
        }
    return ref, part, {"splits": counts, "exclusions": exclusions.to_dict("records")}


def validate_and_align(reference, predictions_by_model, contract, split):
    """Validate active reference rows and align complete prediction populations.

    Prediction DataFrame attrs may carry ``class_order`` from a verified receipt.
    The default is the canonical class order for direct callers.
    """
    ref, active, audit = _prepare_reference(reference, contract, split)
    rows = active[active.split.eq(split) & active._eligible].drop(columns="_eligible")
    if rows.empty:
        _fail("E_EMPTY_POPULATION", f"No eligible {split} forecasts")
    if not isinstance(predictions_by_model, dict) or not predictions_by_model:
        _fail("E_SCHEMA", "predictions_by_model must be a nonempty model mapping")
    expected_keys = _keyset(rows)
    selected_reference_keys = _keyset(ref[ref.split.eq(split)])
    fit_splits = ["train"] if split == "validation" else ["train", "validation"]
    fitting = active[active.split.isin(fit_splits) & active._eligible]
    majority = None if fitting.empty else fitting.target_class.mode().iloc[0]
    result, per_model = [], {}
    for name, source in sorted(predictions_by_model.items()):
        if name not in MODEL_NAMES:
            _fail("E_SCHEMA", f"Unsupported model: {name}")
        allowed_columns = PREDICTION_COLUMNS | (
            {"observed_target_date"} if "observed_target_date" in source else set()
        )
        _columns(source, allowed_columns, f"predictions[{name}]", exact=True)
        frame = source.loc[source.model.eq(name)].copy().reset_index(drop=True)
        order = source.attrs.get("class_order", CLASSES)
        if (
            not isinstance(order, list)
            or not all(isinstance(c, str) for c in order)
            or sorted(order) != sorted(CLASSES)
        ):
            _fail("E_MODEL_SCHEMA", f"Invalid class order for {name}")
        # Only keys/split labels are inspected for ignored outcomes.
        candidate_keys = list(zip(frame.site_id, frame.date))
        selected_keys_mask = pd.Series(
            [k in selected_reference_keys for k in candidate_keys], index=frame.index
        )
        wrong_split = selected_keys_mask & frame.split.ne(split)
        if wrong_split.any():
            _fail(
                "E_TRUTH_MISMATCH",
                f"{name}: selected reference keys have incorrect split labels",
                **_examples(frame, wrong_split),
            )
        ignored = frame.loc[~frame.split.eq(split)].groupby("split").size().to_dict()
        pred = frame.loc[frame.split.eq(split)].copy()
        _keys(pred, f"predictions[{name}] selected split")
        actual_keys = _keyset(pred)
        if actual_keys != expected_keys:
            _fail(
                "E_PREDICTION_COVERAGE",
                f"{name}: prediction keys differ from eligible reference",
                missing_count=len(expected_keys - actual_keys),
                extra_count=len(actual_keys - expected_keys),
                missing_examples=[
                    dict(zip(KEYS, k)) for k in sorted(expected_keys - actual_keys)[:20]
                ],
                extra_examples=[
                    dict(zip(KEYS, k)) for k in sorted(actual_keys - expected_keys)[:20]
                ],
            )
        joined = rows.merge(
            pred, on=KEYS, validate="one_to_one", suffixes=("_ref", "_pred")
        )
        for field in ["split", "target_date"]:
            if not joined[f"{field}_ref"].eq(joined[f"{field}_pred"]).all():
                _fail("E_TRUTH_MISMATCH", f"{name}: {field} differs from reference")
        if not joined.target_class.eq(joined.y_true).all():
            _fail("E_TRUTH_MISMATCH", f"{name}: y_true differs from reference")
        if "observed_target_date" in pred:
            if not joined.observed_target_date_ref.eq(
                joined.observed_target_date_pred
            ).all():
                _fail(
                    "E_TRUTH_MISMATCH",
                    f"{name}: observed target date differs from reference",
                )
            observed_col = "observed_target_date_ref"
        else:
            observed_col = "observed_target_date"
        _labels(joined.y_pred, f"predictions[{name}].y_pred")
        try:
            probabilities = (
                joined[PROBA].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
            )
        except (ValueError, TypeError):
            _fail("E_PROBABILITY", f"{name}: probabilities must be numeric")
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6)
        ):
            _fail(
                "E_PROBABILITY",
                f"{name}: probabilities must be finite, in [0,1], and sum to one",
            )
        if name in {"majority", "persistence"}:
            onehot = np.array(
                [[float(label == c) for c in CLASSES] for label in joined.y_pred]
            )
            if not np.array_equal(probabilities, onehot):
                _fail("E_PROBABILITY", f"{name}: expected exact one-hot indicators")
            if majority is None:
                _fail(
                    "E_FIT_POPULATION",
                    "Deterministic baseline needs a nonempty fitting population",
                )
            expected = (
                pd.Series(majority, index=joined.index)
                if name == "majority"
                else joined.current_class.mask(_missing(joined.current_class), majority)
            )
            if not joined.y_pred.eq(expected).all():
                _fail(
                    "E_LABEL",
                    f"{name}: predictions violate the recorded majority/persistence rule",
                )
        else:
            ordered = probabilities[:, [CLASSES.index(c) for c in order]]
            labels = np.asarray(order)[ordered.argmax(axis=1)]
            if not np.array_equal(labels, joined.y_pred.to_numpy()):
                _fail(
                    "E_PROBABILITY",
                    f"{name}: y_pred disagrees with probabilities/class-order tie rule",
                )
        aligned = joined[KEYS + ["y_true", "y_pred"]].copy()
        aligned["model"] = name
        aligned["split"] = split
        aligned["target_date"] = joined.target_date_ref
        aligned["observed_target_date"] = joined[observed_col]
        aligned[PROBA] = probabilities
        result.append(aligned[ALIGNED_COLUMNS])
        per_model[name] = {
            "matched_prediction_n": len(aligned),
            "prediction_coverage": 1.0,
            "ignored_rows_by_split": {str(k): int(v) for k, v in ignored.items()},
        }
    aligned = (
        pd.concat(result, ignore_index=True)
        .sort_values(["model", "date", "site_id"], kind="stable")
        .reset_index(drop=True)
    )
    absent = [c for c in CLASSES if c not in set(rows.target_class)]
    audit.update(
        {
            "schema_version": 1,
            "status": "passed",
            "split": split,
            "models": per_model,
            "population_sha256": population_hash(rows),
            "warnings": [],
            "errors": [],
            "absent_true_classes": absent,
            "checks": [
                "reference_schema",
                "unique_forecast_keys",
                "split_dates",
                "target_horizon",
                "target_pair_window",
                "actual_target_boundary",
                "same_site_donor",
                "exact_prediction_coverage",
                "truth_alignment",
                "label_probability_mapping",
            ],
        }
    )
    if absent:
        audit["warnings"].append(
            f"No true samples for: {', '.join(absent)}; fixed three-class Macro F1 retained."
        )
    return aligned, audit


def score_predictions(aligned_predictions):
    _columns(aligned_predictions, ALIGNED_COLUMNS, "aligned_predictions")
    if aligned_predictions.empty:
        _fail("E_EMPTY_POPULATION", "Cannot score an empty population")
    _labels(aligned_predictions.y_true, "aligned.y_true")
    _labels(aligned_predictions.y_pred, "aligned.y_pred")
    metrics = {}
    for (model, split), rows in aligned_predictions.groupby(
        ["model", "split"], sort=True
    ):
        metrics.setdefault(model, {})[split] = score(rows.y_true, rows.y_pred)
    return metrics


def _test_gate(split, use_test):
    if not isinstance(split, str) or split not in {"validation", "test"}:
        _fail("E_SCHEMA", "evaluation_split must be validation or test")
    if (split == "test") != bool(use_test):
        _fail("E_TEST_GATE", "A test manifest and --use-test must be supplied together")


def _validate_config(config, use_test):
    _fields(
        config,
        {
            "schema_version",
            "run_id",
            "dataset_kind",
            "evaluation_split",
            "reference",
            "target_contract",
            "models",
            "upstream_review",
            "final_selection",
        },
        "manifest",
    )
    _version(config["schema_version"], "manifest")
    if not isinstance(config["run_id"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]{0,63}", config["run_id"]
    ):
        _fail("E_SCHEMA", "Invalid run_id")
    if not isinstance(config["dataset_kind"], str) or config["dataset_kind"] not in {
        "synthetic",
        "interim",
        "final",
    }:
        _fail("E_SCHEMA", "Unknown dataset_kind")
    _test_gate(config["evaluation_split"], use_test)
    validate_target_contract(
        config["target_contract"], config["dataset_kind"] == "synthetic"
    )
    if (
        config["dataset_kind"] == "final"
        and config["target_contract"]["status"] != "team_final"
    ):
        _fail(
            "E_UPSTREAM_INVALID",
            "Final data requires the actual team-final target agreement",
        )
    if (
        config["evaluation_split"] == "validation"
        and config["final_selection"] is not None
    ):
        _fail(
            "E_TEST_GATE",
            "Validation manifest must not contain a final-test selection record",
        )
    if config["evaluation_split"] == "test" and config["final_selection"] is None:
        _fail("E_TEST_GATE", "Test scoring requires a frozen final-selection record")
    models = config["models"]
    if not isinstance(models, list) or not models:
        _fail("E_SCHEMA", "models must be a nonempty array")
    names = []
    for entry in models:
        _fields(entry, {"name", "predictions", "training_receipt"}, "model entry")
        if not isinstance(entry["name"], str) or entry["name"] not in MODEL_NAMES:
            _fail("E_SCHEMA", "Unsupported model name")
        names.append(entry["name"])
    if len(names) != len(set(names)):
        _fail("E_SCHEMA", "Duplicate model entries")
    if config["dataset_kind"] != "synthetic" and not {
        "majority",
        "persistence",
        "logistic",
    } <= set(names):
        _fail("E_DEPENDENCY", "Project-data comparisons require all three baselines")


def _review(config, parent):
    ref = config["upstream_review"]
    if ref is None:
        if config["dataset_kind"] != "synthetic":
            _fail(
                "E_DEPENDENCY",
                "Project-data runs require upstream review and team agreement evidence",
            )
        return None, False
    verified = _file_ref(ref, parent, "upstream_review")
    review = read_json(verified["path"])
    _fields(
        review,
        {
            "schema_version",
            "team_agreement",
            "checks",
            "source_commits",
            "known_limitations",
        },
        "upstream_review",
    )
    _version(review["schema_version"], "upstream_review")
    agreement = review["team_agreement"]
    _fields(
        agreement,
        {"status", "target_definition", "cleaned_input", "approvals"},
        "team_agreement",
    )
    if agreement["status"] != "confirmed" or agreement["target_definition"] != TARGET:
        _fail(
            "E_UPSTREAM_INVALID",
            "Actual team agreement must confirm the complete selected target definition",
        )
    cleaned = _file_ref(
        agreement["cleaned_input"],
        Path(verified["path"]).parent,
        "team_agreement.cleaned_input",
    )
    required_roles = {"target", "cleaning", "features", "baselines", "lightgbm"}
    roles = set()
    if not isinstance(agreement["approvals"], list) or not agreement["approvals"]:
        _fail("E_UPSTREAM_INVALID", "Team approval evidence is missing")
    for approval in agreement["approvals"]:
        _fields(
            approval,
            {"member", "roles", "confirmed_at_utc", "evidence"},
            "team approval",
        )
        _text(approval["member"], "approval.member")
        _text(approval["evidence"], "approval.evidence")
        _utc(approval["confirmed_at_utc"], "approval.confirmed_at_utc")
        if (
            not isinstance(approval["roles"], list)
            or not approval["roles"]
            or not all(isinstance(role, str) for role in approval["roles"])
            or not set(approval["roles"]) <= required_roles
        ):
            _fail("E_SCHEMA", "Invalid team approval roles")
        roles.update(approval["roles"])
    if roles != required_roles:
        _fail(
            "E_UPSTREAM_INVALID",
            "Agreement does not cover all required owners",
            missing_roles=sorted(required_roles - roles),
        )
    _fields(review["checks"], {f"U{i:02d}" for i in range(1, 7)}, "upstream checks")
    for name, check in review["checks"].items():
        _fields(check, {"status", "evidence"}, name)
        if check["status"] != "passed":
            _fail("E_UPSTREAM_INVALID", f"Required upstream check {name} did not pass")
        _text(check["evidence"], f"{name}.evidence")
    if not isinstance(review["source_commits"], list) or not review["source_commits"]:
        _fail("E_SCHEMA", "Upstream review needs actual source revisions")
    for commit in review["source_commits"]:
        _commit(commit, False, "upstream source commit")
    if not isinstance(review["known_limitations"], list) or not all(
        isinstance(x, str) for x in review["known_limitations"]
    ):
        _fail("E_SCHEMA", "known_limitations must be a string array")
    return {**verified, "cleaned_input": cleaned, "review": review}, config[
        "dataset_kind"
    ] != "synthetic"


def _cv_evidence(settings, recipe, model):
    cv = settings.get("model_selection") if isinstance(settings, dict) else None
    if not isinstance(cv, dict):
        _fail("E_UPSTREAM_INVALID", f"{model}: training-period CV evidence is missing")
    _fields(
        cv,
        {"protocol_id", "folds", "candidates", "selected_candidate"},
        "model_selection",
    )
    if (
        cv["protocol_id"] != "expanding_calendar_3fold_v1"
        or not isinstance(cv["folds"], list)
        or len(cv["folds"]) != 3
    ):
        _fail(
            "E_UPSTREAM_INVALID",
            "CV must use the three prescribed expanding calendar folds",
        )
    for fold, dates in zip(cv["folds"], FOLDS):
        _fields(
            fold,
            {
                "fit_start",
                "fit_end",
                "score_start",
                "score_end",
                "fit_keys_sha256",
                "scoring_keys_sha256",
                "fit_n",
                "scoring_n",
                "scoring_reference_n",
                "scoring_class_support",
            },
            "CV fold",
        )
        if (
            tuple(fold[k] for k in ("fit_start", "fit_end", "score_start", "score_end"))
            != dates
        ):
            _fail("E_UPSTREAM_INVALID", "CV fold dates differ from the frozen protocol")
        for k in ("fit_keys_sha256", "scoring_keys_sha256"):
            _sha(fold[k], f"CV.{k}")
        for k in ("fit_n", "scoring_n", "scoring_reference_n"):
            if type(fold[k]) is not int or fold[k] <= 0:
                _fail(
                    "E_UPSTREAM_INVALID",
                    "CV population counts must be positive integers",
                )
        support = fold["scoring_class_support"]
        _fields(support, CLASSES, "CV support")
        if (
            not all(type(v) is int and v > 0 for v in support.values())
            or sum(support.values()) != fold["scoring_n"]
            or fold["scoring_n"] > fold["scoring_reference_n"]
        ):
            _fail("E_UPSTREAM_INVALID", "CV class support/coverage does not reconcile")
    candidates = cv["candidates"]
    if not isinstance(candidates, list) or not candidates:
        _fail("E_UPSTREAM_INVALID", "CV candidate grid is missing")
    ids, checked = set(), []
    for candidate in candidates:
        _fields(
            candidate,
            {"id", "parameters", "fold_macro_f1", "mean_macro_f1"},
            "CV candidate",
        )
        _text(candidate["id"], "candidate.id")
        if candidate["id"] in ids or not isinstance(candidate["parameters"], dict):
            _fail(
                "E_SCHEMA",
                "CV candidate IDs must be unique and parameters must be objects",
            )
        ids.add(candidate["id"])
        values = candidate["fold_macro_f1"]
        if (
            not isinstance(values, list)
            or len(values) != 3
            or not all(_finite_number(x) and 0 <= x <= 1 for x in values)
        ):
            _fail("E_UPSTREAM_INVALID", "CV requires three finite fold Macro F1 values")
        average = sum(values) / 3
        if (
            not _finite_number(candidate["mean_macro_f1"])
            or abs(candidate["mean_macro_f1"] - average) > 1e-12
        ):
            _fail(
                "E_UPSTREAM_INVALID",
                "CV aggregate must be the unweighted mean of fold scores",
            )
        checked.append(candidate)
    if model == "logistic":
        cs = [c["parameters"].get("C") for c in checked]
        if any(not _finite_number(c) for c in cs) or sorted(cs) != sorted(C_GRID):
            _fail(
                "E_UPSTREAM_INVALID",
                "Logistic CV must record the prescribed seven-value C grid",
            )

        def tie_key(candidate):
            return candidate["parameters"]["C"]
    else:
        for candidate in checked:
            value = candidate["parameters"].get("num_iterations")
            if type(value) is not int or value <= 0:
                _fail(
                    "E_UPSTREAM_INVALID",
                    "LightGBM candidates require fixed positive num_iterations",
                )

        def tie_key(candidate):
            return (
                candidate["parameters"]["num_iterations"],
                json.dumps(
                    candidate["parameters"], sort_keys=True, separators=(",", ":")
                ),
            )

    maximum = max(c["mean_macro_f1"] for c in checked)
    winner = min(
        [c for c in checked if maximum - c["mean_macro_f1"] <= 1e-12], key=tie_key
    )
    if (
        cv["selected_candidate"] != winner["id"]
        or winner["parameters"] != recipe["parameters"]
    ):
        _fail(
            "E_UPSTREAM_INVALID",
            "Selected recipe disagrees with CV ranking/tie-break or parameter evidence",
        )


def _receipt(entry, config, parent, reference_ref, active):
    name = entry["name"]
    verified = _file_ref(entry["training_receipt"], parent, f"{name}.training_receipt")
    receipt_path = Path(verified["path"])
    receipt = read_json(receipt_path)
    _fields(
        receipt,
        {
            "schema_version",
            "model",
            "prediction_split",
            "reference_sha256",
            "target_contract_id",
            "training_code_commit",
            "fit_splits",
            "fit_rows",
            "preprocessing_fit_splits",
            "features",
            "class_order",
            "probability_kind",
            "majority_class",
            "settings",
            "selection_settings",
            "selection_source",
            "model_artifact",
            "runtime_versions",
        },
        f"{name} receipt",
    )
    _version(receipt["schema_version"], "receipt")
    synthetic = config["dataset_kind"] == "synthetic"
    for key, expected in {
        "model": name,
        "prediction_split": config["evaluation_split"],
        "reference_sha256": reference_ref["sha256"],
        "target_contract_id": TARGET["id"],
        "preprocessing_fit_splits": ["train"],
    }.items():
        if receipt[key] != expected:
            _fail(
                "E_UPSTREAM_INVALID", f"{name}: receipt {key} does not match this run"
            )
    _commit(receipt["training_code_commit"], synthetic, "training_code_commit")
    fit_splits = (
        ["train"]
        if config["evaluation_split"] == "validation"
        else ["train", "validation"]
    )
    if receipt["fit_splits"] != fit_splits:
        _fail("E_FIT_POPULATION", f"{name}: incorrect fit_splits")
    fit_ref = _file_ref(receipt["fit_rows"], receipt_path.parent, f"{name}.fit_rows")
    fit_keys = _csv(fit_ref["path"])
    _columns(fit_keys, KEYS, "fit_rows", exact=True)
    _keys(fit_keys, "fit_rows")
    expected = active[active.split.isin(fit_splits) & active._eligible]
    if expected.empty or _keyset(fit_keys) != _keyset(expected):
        _fail(
            "E_FIT_POPULATION",
            f"{name}: actual fit keys differ from boundary-safe fitting population",
            expected_n=len(expected),
            actual_n=len(fit_keys),
            extra_examples=[
                dict(zip(KEYS, k))
                for k in sorted(_keyset(fit_keys) - _keyset(expected))[:20]
            ],
        )
    features = receipt["features"]
    learned = name in {"logistic", "lightgbm"}
    if (
        not isinstance(features, list)
        or any(not isinstance(f, str) or not f for f in features)
        or len(features) != len(set(features))
        or bool(features) != learned
        or set(features) & RESERVED
    ):
        _fail("E_MODEL_SCHEMA", f"{name}: invalid ordered feature list")
    for feature in features:
        if feature not in active:
            _fail("E_MODEL_SCHEMA", f"{name}: feature is missing: {feature}")
        try:
            values = pd.to_numeric(
                active[feature].mask(_missing(active[feature])), errors="raise"
            ).to_numpy(dtype=float)
        except (TypeError, ValueError):
            _fail("E_MODEL_SCHEMA", f"{name}: feature must be numeric: {feature}")
        if np.isinf(values).any():
            _fail("E_MODEL_SCHEMA", f"{name}: infinite feature values: {feature}")
    order = receipt["class_order"]
    if (
        not isinstance(order, list)
        or not all(isinstance(c, str) for c in order)
        or sorted(order) != sorted(CLASSES)
    ):
        _fail("E_MODEL_SCHEMA", f"{name}: invalid class order")
    kind = "predict_proba" if learned else "one_hot"
    if receipt["probability_kind"] != kind:
        _fail("E_MODEL_SCHEMA", f"{name}: incorrect probability kind")
    majority = expected.target_class.mode().iloc[0]
    if receipt["majority_class"] != (None if learned else majority):
        _fail(
            "E_UPSTREAM_INVALID", f"{name}: incorrect recorded fitting-majority class"
        )
    versions = receipt["runtime_versions"]
    if (
        not isinstance(versions, dict)
        or not versions
        or "python" not in versions
        or not all(isinstance(v, str) and v for v in versions.values())
    ):
        _fail(
            "E_SCHEMA", f"{name}: actual runtime versions including python are required"
        )
    settings_ref = _file_ref(
        receipt["settings"], receipt_path.parent, f"{name}.settings"
    )
    settings = read_json(settings_ref["path"])
    if not isinstance(settings, dict):
        _fail("E_SCHEMA", f"{name}: settings must be a JSON object")
    selection_ref = _file_ref(
        receipt["selection_settings"], receipt_path.parent, f"{name}.selection_settings"
    )
    recipe = read_json(selection_ref["path"])
    _fields(
        recipe,
        {
            "schema_version",
            "model",
            "target_contract_id",
            "features",
            "class_order",
            "recipe",
            "parameters",
        },
        "selection_settings",
    )
    _version(recipe["schema_version"], "selection_settings")
    for field in ("model", "target_contract_id", "features", "class_order"):
        if recipe[field] != receipt[field]:
            _fail(
                "E_UPSTREAM_INVALID",
                f"{name}: frozen recipe {field} disagrees with receipt",
            )
    _text(recipe["recipe"], "selection_settings.recipe")
    if not isinstance(recipe["parameters"], dict):
        _fail("E_SCHEMA", "selection_settings.parameters must be an object")
    if learned:
        if receipt["selection_source"] == "train_cv":
            _cv_evidence(settings, recipe, name)
        elif not (synthetic and receipt["selection_source"] == "none"):
            _fail(
                "E_UPSTREAM_INVALID",
                f"{name}: project models require the prescribed training-only CV evidence",
            )
    elif receipt["selection_source"] != "none":
        _fail(
            "E_UPSTREAM_INVALID",
            f"{name}: deterministic baseline selection_source must be none",
        )
    artifact = receipt["model_artifact"]
    if learned:
        if artifact is None:
            _fail(
                "E_DEPENDENCY",
                f"{name}: actual split-specific model artifact is missing",
            )
        artifact = _file_ref(
            artifact, receipt_path.parent, f"{name}.model_artifact", artifact=True
        )
        if name == "logistic":
            if (
                artifact["format"] != "sklearn_joblib"
                or artifact["num_iterations"] is not None
            ):
                _fail(
                    "E_MODEL_SCHEMA",
                    "Logistic artifact must be sklearn_joblib with null num_iterations",
                )
        elif (
            artifact["format"] != "lightgbm_text"
            or type(artifact["num_iterations"]) is not int
            or artifact["num_iterations"] <= 0
        ):
            _fail(
                "E_MODEL_SCHEMA",
                "LightGBM needs a native text model with a positive iteration count",
            )
        if (
            name == "lightgbm"
            and "num_iterations" in recipe["parameters"]
            and recipe["parameters"]["num_iterations"] != artifact["num_iterations"]
        ):
            _fail(
                "E_MODEL_SCHEMA",
                "LightGBM artifact iteration count differs from its frozen recipe",
            )
    elif artifact is not None:
        _fail(
            "E_MODEL_SCHEMA",
            "Deterministic baselines must not declare a fitted model artifact",
        )
    return {
        "training_receipt": verified,
        "receipt": receipt,
        "receipt_path": str(receipt_path),
        "fit_rows": fit_ref,
        "settings": settings_ref,
        "selection_settings": selection_ref,
        "model_artifact": artifact,
    }


def _final_selection(config, parent, inputs):
    if config["evaluation_split"] != "test":
        return None, None
    verified = _file_ref(config["final_selection"], parent, "final_selection")
    freeze = read_json(verified["path"])
    _fields(
        freeze,
        {
            "schema_version",
            "frozen_at_utc",
            "target_contract_id",
            "reference_sha256",
            "validation_run_manifest",
            "rule_id",
            "selected_model",
            "models",
            "decision_note",
        },
        "final_selection",
    )
    _version(freeze["schema_version"], "final_selection")
    frozen_at = _utc(freeze["frozen_at_utc"], "final_selection.frozen_at_utc")
    _text(freeze["decision_note"], "decision_note")
    if (
        freeze["target_contract_id"] != TARGET["id"]
        or freeze["reference_sha256"] != inputs["reference"]["sha256"]
    ):
        _fail(
            "E_TEST_GATE", "Frozen selection target/reference differs from the test run"
        )
    base = Path(verified["path"]).parent
    validation_ref = _file_ref(
        freeze["validation_run_manifest"], base, "validation_run_manifest"
    )
    if Path(validation_ref["path"]).name != "manifest.json":
        _fail(
            "E_TEST_GATE",
            "validation_run_manifest must name the completed run's manifest.json",
        )
    validation = load_verified_run(Path(validation_ref["path"]).parent)
    if frozen_at < _utc(validation["created_at_utc"], "validation.created_at_utc"):
        _fail(
            "E_TEST_GATE",
            "Selection freeze predates the recorded validation evaluation",
        )
    if (
        validation["inputs"]["reference"]["sha256"] != inputs["reference"]["sha256"]
        or validation["target_contract"] != config["target_contract"]
    ):
        _fail(
            "E_TEST_GATE", "Validation and test must use the same target and reference"
        )
    if set(validation["inputs"]["models"]) != set(inputs["models"]):
        _fail("E_TEST_GATE", "Validation and test model sets differ")
    selected = read_json(Path(validation_ref["path"]).parent / "selection.json")
    if selected.get("status") != "frozen_ready" or any(
        freeze[k] != selected[k] for k in ("rule_id", "selected_model")
    ):
        _fail(
            "E_TEST_GATE",
            "Frozen selection does not match a ready validation selection",
        )
    if not isinstance(freeze["models"], list):
        _fail("E_SCHEMA", "final_selection.models must be an array")
    names = []
    frozen_models = {}
    for entry in freeze["models"]:
        _fields(entry, {"name", "selection_settings"}, "frozen model")
        name = entry["name"]
        if not isinstance(name, str) or name not in inputs["models"] or name in names:
            _fail(
                "E_TEST_GATE", "Frozen model set contains an unknown or duplicate model"
            )
        names.append(name)
        file = _file_ref(entry["selection_settings"], base, "frozen selection_settings")
        expected = inputs["models"][name]["selection_settings"]["sha256"]
        if (
            file["sha256"] != expected
            or file["sha256"]
            != validation["inputs"]["models"][name]["selection_settings"]["sha256"]
        ):
            _fail(
                "E_TEST_GATE", f"{name}: predictive settings changed after validation"
            )
        frozen_models[name] = file
    if set(names) != set(inputs["models"]):
        _fail("E_TEST_GATE", "Frozen selection omits evaluated models")
    return {
        **verified,
        "validation_run_manifest": validation_ref,
        "selection_settings": frozen_models,
    }, selected


def _runtime():
    versions = {"python": platform.python_version()}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "joblib",
        "lightgbm",
        "shap",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _source():
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    files = {}
    for relative in (
        "src/evaluate.py",
        "src/explain.py",
        "src/_evaluation_reports.py",
        "src/baselines.py",
        "src/split.py",
        "requirements-evaluation.txt",
    ):
        path = ROOT / relative
        if path.is_file():
            files[relative] = sha256_file(path)
    status = git("status", "--porcelain")
    return {
        "head": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "files": files,
    }


def _verify_recorded_files(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            path = value["path"]
            if not isinstance(path, str) or not Path(path).is_absolute():
                _fail("E_SCHEMA", "Recorded input path must be absolute")
            _sha(value["sha256"], "recorded input sha256")
            if sha256_file(path) != value["sha256"]:
                _fail("E_HASH", f"Recorded input changed: {path}")
        # receipt/review objects preserve source-relative paths; their resolved
        # copies are siblings in the manifest and are verified separately.
        for key, child in value.items():
            if key not in {"receipt", "review"}:
                _verify_recorded_files(child)
    elif isinstance(value, list):
        for child in value:
            _verify_recorded_files(child)


def load_verified_run(run_dir, use_test=False):
    """Load a complete evaluation only after checking its input/output hashes."""
    run = Path(run_dir).resolve()
    manifest = read_json(run / "manifest.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or manifest.get("schema_version") != 1
    ):
        _fail("E_SCHEMA", "Evaluation run has no valid completion marker")
    _test_gate(manifest.get("evaluation_split"), use_test)
    required = {
        "inputs",
        "outputs",
        "input_manifest",
        "input_manifest_path",
        "input_manifest_sha256",
        "target_contract",
        "dataset_kind",
        "run_id",
        "created_at_utc",
    }
    if not required <= manifest.keys():
        _fail(
            "E_SCHEMA",
            "Completed evaluation manifest is missing required provenance fields",
        )
    if (
        sha256_file(manifest["input_manifest_path"])
        != manifest["input_manifest_sha256"]
    ):
        _fail("E_HASH", "Original evaluation input manifest changed")
    if read_json(manifest["input_manifest_path"]) != manifest["input_manifest"]:
        _fail("E_HASH", "Recorded input manifest copy differs from its verified file")
    config = manifest["input_manifest"]
    _validate_config(config, use_test)
    parent = Path(manifest["input_manifest_path"]).parent
    for key in ("run_id", "dataset_kind", "evaluation_split", "target_contract"):
        if manifest[key] != config[key]:
            _fail("E_HASH", f"Completed manifest {key} differs from its original input")
    inputs = manifest["inputs"]
    if not isinstance(inputs, dict) or not isinstance(inputs.get("models"), dict):
        _fail("E_SCHEMA", "Completed manifest has invalid resolved inputs")
    expected_reference = _file_ref(config["reference"], parent, "reference")
    if inputs.get("reference") != expected_reference:
        _fail("E_HASH", "Resolved reference differs from the input manifest")
    if set(inputs["models"]) != {entry["name"] for entry in config["models"]}:
        _fail("E_HASH", "Resolved model set differs from the input manifest")
    for entry in config["models"]:
        name = entry["name"]
        model = inputs["models"][name]
        receipt_ref = _file_ref(entry["training_receipt"], parent, "training_receipt")
        receipt = read_json(receipt_ref["path"])
        if (
            model.get("training_receipt") != receipt_ref
            or model.get("receipt_path") != receipt_ref["path"]
            or model.get("receipt") != receipt
        ):
            _fail("E_HASH", f"{name}: copied receipt differs from its verified source")
        if model.get("predictions") != _file_ref(
            entry["predictions"], parent, "predictions"
        ):
            _fail("E_HASH", f"{name}: resolved prediction input differs")
        for key in ("fit_rows", "settings", "selection_settings", "model_artifact"):
            expected = (
                None
                if receipt[key] is None
                else _file_ref(
                    receipt[key],
                    Path(receipt_ref["path"]).parent,
                    key,
                    artifact=key == "model_artifact",
                )
            )
            if model.get(key) != expected:
                _fail(
                    "E_HASH",
                    f"{name}: resolved {key} differs from its verified receipt",
                )
    if config["upstream_review"] is None:
        if inputs.get("upstream_review") is not None:
            _fail("E_HASH", "Unexpected copied upstream review")
    else:
        review_ref = _file_ref(config["upstream_review"], parent, "upstream_review")
        review = read_json(review_ref["path"])
        copied = inputs.get("upstream_review", {})
        if (
            any(copied.get(k) != review_ref[k] for k in ("path", "sha256"))
            or copied.get("review") != review
        ):
            _fail("E_HASH", "Copied upstream review differs from its verified source")
        cleaned = _file_ref(
            review["team_agreement"]["cleaned_input"],
            Path(review_ref["path"]).parent,
            "cleaned_input",
        )
        if copied.get("cleaned_input") != cleaned:
            _fail(
                "E_HASH",
                "Resolved cleaned input differs from the actual team-agreement record",
            )
    if config["final_selection"] is None:
        if inputs.get("final_selection") is not None:
            _fail("E_HASH", "Unexpected copied final-selection record")
    else:
        final_ref = _file_ref(config["final_selection"], parent, "final_selection")
        copied = inputs.get("final_selection", {})
        if any(copied.get(k) != final_ref[k] for k in ("path", "sha256")):
            _fail(
                "E_HASH",
                "Resolved final-selection record differs from its original input",
            )
    _verify_recorded_files(manifest["inputs"])
    if (
        not isinstance(manifest["outputs"], dict)
        or not {
            "audit.json",
            "metrics.json",
            "aligned_predictions.csv",
            "selection.json",
        }
        <= manifest["outputs"].keys()
    ):
        _fail("E_SCHEMA", "Completed run is missing mandatory output hashes")
    for relative, digest in manifest["outputs"].items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run) or path == run / "manifest.json":
            _fail(
                "E_SCHEMA",
                "Output path escapes the run or recursively hashes its manifest",
            )
        _sha(digest, "output digest")
        if sha256_file(path) != digest:
            _fail("E_HASH", f"Evaluation output changed: {relative}")
    return manifest


def run_evaluation(manifest_path, output_root=None, use_test=False):
    """Validate all handoffs, then create one immutable evaluation bundle."""
    manifest_path = Path(manifest_path).resolve()
    initial_manifest_hash = sha256_file(manifest_path)
    config = read_json(manifest_path)
    # Establish only a safe run directory; subsequent contract failures leave an
    # audit there. Invalid run IDs never choose a filesystem destination.
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("run_id"), str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", config["run_id"])
    ):
        _fail("E_SCHEMA", "A valid run_id is required before creating output")
    output_root = (
        Path(output_root)
        if output_root is not None
        else ROOT / "reports" / "evaluation"
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run = output_root / config["run_id"]
    try:
        run.mkdir()
    except FileExistsError:
        _fail("E_OUTPUT_EXISTS", f"Output already exists: {run}")
    try:
        _validate_config(config, use_test)
        parent = manifest_path.parent
        reference_ref = _file_ref(config["reference"], parent, "reference")
        reference = _csv(reference_ref["path"])
        reference, active, _ = _prepare_reference(
            reference, config["target_contract"], config["evaluation_split"]
        )
        review, provenance_verified = _review(config, parent)
        inputs = {"reference": reference_ref, "models": {}, "upstream_review": review}
        predictions, cache, unselected = {}, {}, {}
        selected_names = {entry["name"] for entry in config["models"]}
        for entry in config["models"]:
            name = entry["name"]
            model = _receipt(entry, config, parent, reference_ref, active)
            pred_ref = _file_ref(entry["predictions"], parent, f"{name}.predictions")
            if pred_ref["path"] not in cache:
                cache[pred_ref["path"]] = _csv(pred_ref["path"])
            frame = cache[pred_ref["path"]].copy()
            _columns(frame, PREDICTION_COLUMNS, f"{name}.predictions")
            frame.attrs["class_order"] = model["receipt"]["class_order"]
            predictions[name] = frame
            model["predictions"] = pred_ref
            inputs["models"][name] = model
            unselected[pred_ref["path"]] = sorted(
                set(frame.model.astype(str)) - selected_names
            )
        freeze_ref, frozen_selection = _final_selection(config, parent, inputs)
        inputs["final_selection"] = freeze_ref
        aligned, audit = validate_and_align(
            reference,
            predictions,
            config["target_contract"],
            config["evaluation_split"],
        )
        audit["checks"].extend(
            [
                "file_hashes",
                "training_receipts",
                "fit_population_equality",
                "frozen_prediction_recipes",
            ]
        )
        audit["fit_key_checks"] = {
            name: {
                "n": len(
                    active[
                        active.split.isin(m["receipt"]["fit_splits"]) & active._eligible
                    ]
                ),
                "passed": True,
            }
            for name, m in inputs["models"].items()
        }
        audit["unselected_models_by_file"] = unselected
        audit["provenance_verified"] = provenance_verified
        audit["final_comparison_ready"] = bool(
            provenance_verified
            and config["dataset_kind"] == "final"
            and config["target_contract"]["status"] == "team_final"
            and selected_names == MODEL_NAMES
        )
        audit["warnings"].append(
            "Provenance records and upstream test evidence are reviewed declarations, not proof of every training operation."
        )
        if review:
            audit["warnings"].extend(review["review"]["known_limitations"])
            audit["checks"].extend(
                [
                    "recorded_team_agreement",
                    "upstream_U01_U06_evidence",
                    "training_cv_evidence",
                ]
            )
        metrics = score_predictions(aligned)
        _write_json(run / "metrics.json", metrics)
        aligned.to_csv(
            run / "aligned_predictions.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(audit["exclusions"], columns=["split", *KEYS, "reason"]).to_csv(
            run / "exclusions.csv", index=False, lineterminator="\n"
        )
        from src._evaluation_reports import write_reports

        reports = write_reports(
            aligned,
            reference,
            audit,
            config,
            metrics,
            run,
            frozen_selection=frozen_selection,
        )
        audit["warnings"].extend(reports.get("warnings", []))
        source = _source()
        if source["dirty"]:
            audit["warnings"].append(
                "Evaluation source is uncommitted; actual module hashes are recorded."
            )
        _write_json(run / "audit.json", audit)
        # Inputs can be regenerated by another process while plots are written.
        # Publish completion only if the bytes used by this run stayed frozen.
        _verify_recorded_files(inputs)
        if sha256_file(manifest_path) != initial_manifest_hash:
            _fail("E_HASH", "Evaluation input manifest changed during the run")
        outputs = {
            str(p.relative_to(run)): sha256_file(p)
            for p in sorted(run.rglob("*"))
            if p.is_file()
        }
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": config["run_id"],
            "dataset_kind": config["dataset_kind"],
            "evaluation_split": config["evaluation_split"],
            "target_contract": config["target_contract"],
            "input_manifest": config,
            "input_manifest_path": str(manifest_path),
            "input_manifest_sha256": initial_manifest_hash,
            "inputs": inputs,
            "population_sha256": audit["population_sha256"],
            "runtime_versions": _runtime(),
            "source": source,
            "command_arguments": {
                "manifest": str(manifest_path),
                "output_root": str(output_root),
                "use_test": bool(use_test),
            },
            "outputs": outputs,
        }
        _write_json(run / "manifest.json", manifest)
        return run
    except Exception as exc:
        # Remove only files from the freshly created run, never an existing run.
        for path in run.iterdir():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        error = (
            exc.as_dict()
            if isinstance(exc, EvaluationContractError)
            else {
                "code": "E_INTERNAL",
                "message": str(exc),
                "details": {"type": type(exc).__name__},
            }
        )
        _write_json(
            run / "audit.json",
            {
                "schema_version": 1,
                "status": "failed",
                "split": config.get("evaluation_split"),
                "errors": [error],
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "reports" / "evaluation"
    )
    parser.add_argument("--use-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        path = run_evaluation(args.manifest, args.output_root, args.use_test)
    except EvaluationContractError as exc:
        print(json.dumps(exc.as_dict(), allow_nan=False), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Completed evaluation: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
