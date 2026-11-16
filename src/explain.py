"""Verified, split-specific multiclass LightGBM SHAP reports.

Run ``python -m src.explain --evaluation-run reports/evaluation/RUN_ID``.
The model need only be included in the evaluation; it need not be its winner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import CLASSES, RESERVED
from src.evaluate import EvaluationContractError, load_verified_run, sha256_file

RTOL = 1e-5
ATOL = 1e-6
GLOBAL_SAMPLE_LIMIT = 1000
CASE_CLASSES = {
    "correct_warming": "warming",
    "missed_warming": "warming",
    "false_warming": "warming",
    "correct_cooling": "cooling",
}
LOCAL_COLUMNS = [
    "case_id",
    "status",
    "reason",
    "site_id",
    "date",
    "target_date",
    "observed_target_date",
    "y_true",
    "y_pred",
    "explained_class",
    "proba_cooling",
    "proba_neutral",
    "proba_warming",
    "base_value",
    "raw_score",
]


def _json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _dependencies():
    """Optional native/plotting packages must not affect metric-only imports."""
    try:
        lgb = importlib.import_module("lightgbm")
        mpl = importlib.import_module("matplotlib")
        mpl.use("Agg", force=True)
        shap = importlib.import_module("shap")
        plt = importlib.import_module("matplotlib.pyplot")
    except (ImportError, OSError) as exc:
        raise EvaluationContractError(
            "E_DEPENDENCY",
            "SHAP requires working LightGBM, SHAP and matplotlib installations.",
            {"cause": str(exc)},
        ) from exc
    return lgb, shap, plt


def _class_indices(class_order):
    if (
        not isinstance(class_order, list)
        or len(class_order) != 3
        or set(class_order) != set(CLASSES)
    ):
        raise EvaluationContractError(
            "E_MODEL_SCHEMA",
            "Model class_order must be a permutation of the three classes.",
        )
    return [class_order.index(label) for label in CLASSES]


def _normalise_shap(values, base_values, n_rows, n_features, class_order):
    """Accept only explicitly supported layouts, returning canonical class axes."""
    order = _class_indices(class_order)
    try:
        if isinstance(values, list):
            if len(values) != 3 or any(
                np.shape(v) != (n_rows, n_features) for v in values
            ):
                raise ValueError("list must contain three (rows, features) arrays")
            values = np.stack(values, axis=-1)
        if not isinstance(values, np.ndarray) or values.shape != (
            n_rows,
            n_features,
            3,
        ):
            raise ValueError("values must have shape (rows, features, 3)")
        values = np.asarray(values, dtype=float)
        base = np.asarray(base_values, dtype=float)
        if base.shape == (3,):
            base = np.broadcast_to(base, (n_rows, 3)).copy()
        elif base.shape != (n_rows, 3):
            raise ValueError("base values must have shape (3,) or (rows, 3)")
        if not np.isfinite(values).all() or not np.isfinite(base).all():
            raise ValueError("attributions and base values must be finite")
    except (TypeError, ValueError) as exc:
        raise EvaluationContractError("E_SHAP_SHAPE", str(exc)) from exc
    return values[:, :, order], base[:, order]


def _check_additivity(values, base_values, raw_scores):
    if raw_scores.shape != base_values.shape or values.shape[0] != raw_scores.shape[0]:
        raise EvaluationContractError(
            "E_SHAP_SHAPE", "Raw-score and attribution row/class dimensions differ."
        )
    reconstructed = base_values + values.sum(axis=1)
    if not np.isfinite(raw_scores).all() or not np.isfinite(reconstructed).all():
        raise EvaluationContractError(
            "E_SHAP_ADDITIVITY", "Raw-score reconstruction contains a nonfinite value."
        )
    error = float(np.max(np.abs(reconstructed - raw_scores)))
    if not np.allclose(reconstructed, raw_scores, rtol=RTOL, atol=ATOL):
        raise EvaluationContractError(
            "E_SHAP_ADDITIVITY",
            "Base values plus feature contributions do not reproduce all class raw scores.",
            {"max_absolute_error": error, "rtol": RTOL, "atol": ATOL},
        )
    return error


def _select_rows(rows):
    """Select by key hashes, never by errors for the global sample."""
    ranks = sorted(
        range(len(rows)),
        key=lambda i: (
            hashlib.sha256(
                f"0|{rows.iloc[i]['site_id']}|{rows.iloc[i]['date']}".encode("utf-8")
            ).hexdigest(),
            str(rows.iloc[i]["site_id"]),
            str(rows.iloc[i]["date"]),
        ),
    )
    global_indices = set(ranks[:GLOBAL_SAMPLE_LIMIT])
    conditions = {
        "correct_warming": (rows.y_true == "warming") & (rows.y_pred == "warming"),
        "missed_warming": (rows.y_true == "warming") & (rows.y_pred != "warming"),
        "false_warming": (rows.y_true != "warming") & (rows.y_pred == "warming"),
        "correct_cooling": (rows.y_true == "cooling") & (rows.y_pred == "cooling"),
    }
    cases = {
        name: next((i for i in ranks if bool(mask.iloc[i])), None)
        for name, mask in conditions.items()
    }
    selected = global_indices | {i for i in cases.values() if i is not None}
    indices = np.asarray([i for i in ranks if i in selected], dtype=int)
    return (
        indices,
        np.asarray([i in global_indices for i in indices], dtype=bool),
        cases,
    )


def _load_and_reproduce(manifest, run_dir, lgb):
    models = manifest.get("inputs", {}).get("models", {})
    if "lightgbm" not in models:
        raise EvaluationContractError(
            "E_DEPENDENCY", "The completed evaluation does not include LightGBM."
        )
    entry = models["lightgbm"]
    receipt = entry["receipt"]
    artifact = entry.get("model_artifact")
    if not artifact or artifact.get("format") != "lightgbm_text":
        raise EvaluationContractError(
            "E_MODEL_SCHEMA", "A native LightGBM text artifact is required."
        )
    iterations = artifact.get("num_iterations")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        raise EvaluationContractError(
            "E_MODEL_SCHEMA", "A positive selected iteration count is required."
        )
    features = receipt.get("features")
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(f, str) or not f for f in features)
        or len(set(features)) != len(features)
    ):
        raise EvaluationContractError(
            "E_MODEL_SCHEMA",
            "A nonempty ordered list of unique feature names is required.",
        )
    forbidden = set(features) & (RESERVED | {"model", "y_true", "y_pred"})
    if forbidden:
        raise EvaluationContractError(
            "E_MODEL_SCHEMA",
            "Identifiers, dates and labels cannot be model features.",
            {"features": sorted(forbidden)},
        )
    class_order = receipt.get("class_order")
    order = _class_indices(class_order)
    try:
        booster = lgb.Booster(model_file=artifact["path"])
    except Exception as exc:
        raise EvaluationContractError(
            "E_MODEL_SCHEMA",
            "Cannot load the recorded native LightGBM model.",
            {"cause": str(exc)},
        ) from exc
    objective = str(booster.dump_model(num_iteration=1).get("objective", "")).split(
        " ", 1
    )[0]
    if (
        objective != "multiclass"
        or booster.num_model_per_iteration() != 3
        or booster.current_iteration() != iterations
        or booster.num_feature() != len(features)
        or booster.feature_name() != features
    ):
        raise EvaluationContractError(
            "E_MODEL_SCHEMA",
            "Model objective, feature order or selected iteration count does not match its receipt.",
            {
                "objective": objective,
                "model_features": booster.feature_name(),
                "recorded_features": features,
                "model_iterations": booster.current_iteration(),
                "recorded_iterations": iterations,
            },
        )
    aligned = pd.read_csv(
        Path(run_dir) / "aligned_predictions.csv",
        keep_default_na=False,
        dtype={"site_id": str, "date": str},
    )
    rows = (
        aligned.loc[aligned.model == "lightgbm"]
        .sort_values(["date", "site_id"], kind="stable")
        .reset_index(drop=True)
    )
    if rows.empty or rows.duplicated(["site_id", "date"]).any():
        raise EvaluationContractError(
            "E_MODEL_SCHEMA",
            "LightGBM aligned predictions must have nonempty unique keys.",
        )
    if not rows["split"].eq(manifest["evaluation_split"]).all():
        raise EvaluationContractError(
            "E_MODEL_SCHEMA",
            "Aligned LightGBM rows do not match the recorded evaluation split.",
        )
    reference = pd.read_csv(
        manifest["inputs"]["reference"]["path"],
        keep_default_na=False,
        dtype={"site_id": str, "date": str},
    )
    missing = set(features) - set(reference.columns)
    if missing:
        raise EvaluationContractError(
            "E_MODEL_SCHEMA",
            "The reference lacks recorded model features.",
            {"features": sorted(missing)},
        )
    if reference.duplicated(["site_id", "date"]).any():
        raise EvaluationContractError(
            "E_MODEL_SCHEMA", "Reference keys are not unique."
        )
    joined = rows[["site_id", "date"]].merge(
        reference[["site_id", "date"] + features],
        on=["site_id", "date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise EvaluationContractError(
            "E_MODEL_SCHEMA", "An explanation key is absent from the reference."
        )
    try:
        X = (
            joined[features]
            .replace("", np.nan)
            .apply(pd.to_numeric, errors="raise")
            .astype(float)
        )
    except (ValueError, TypeError) as exc:
        raise EvaluationContractError(
            "E_MODEL_SCHEMA",
            "Features must be numeric or missing; no implicit encoding is supported.",
        ) from exc
    if np.isinf(X.to_numpy()).any():
        raise EvaluationContractError(
            "E_MODEL_SCHEMA", "Features cannot contain infinity."
        )
    try:
        native_proba = np.asarray(
            booster.predict(
                X,
                start_iteration=0,
                num_iteration=iterations,
                validate_features=True,
                num_threads=1,
            ),
            dtype=float,
        )
        native_raw = np.asarray(
            booster.predict(
                X,
                start_iteration=0,
                num_iteration=iterations,
                raw_score=True,
                validate_features=True,
                num_threads=1,
            ),
            dtype=float,
        )
    except Exception as exc:
        raise EvaluationContractError(
            "E_MODEL_REPRODUCTION",
            "The recorded model cannot predict the recorded feature matrix.",
            {"cause": str(exc)},
        ) from exc
    if (
        native_proba.shape != (len(rows), 3)
        or native_raw.shape != (len(rows), 3)
        or not np.isfinite(native_proba).all()
        or not np.isfinite(native_raw).all()
    ):
        raise EvaluationContractError(
            "E_MODEL_REPRODUCTION",
            "Model must produce finite three-class probability and raw-score arrays.",
        )
    expected = rows[[f"proba_{c}" for c in CLASSES]].to_numpy(dtype=float)
    canonical_proba = native_proba[:, order]
    predicted = np.asarray(class_order)[native_proba.argmax(axis=1)]
    if not np.allclose(
        canonical_proba, expected, rtol=RTOL, atol=ATOL
    ) or not np.array_equal(predicted, rows.y_pred.to_numpy()):
        raise EvaluationContractError(
            "E_MODEL_REPRODUCTION",
            "The saved model does not reproduce all saved probabilities and predicted classes.",
            {
                "max_probability_error": float(
                    np.max(np.abs(canonical_proba - expected))
                ),
                "label_mismatches": int(
                    np.count_nonzero(predicted != rows.y_pred.to_numpy())
                ),
            },
        )
    return booster, rows, X, native_raw[:, order], receipt, artifact


def _draw_global(shap, plt, output, X, values, global_mask, features, split):
    global_values = values[global_mask]
    global_X = X.iloc[np.flatnonzero(global_mask)]
    records = []
    for k, label in enumerate(CLASSES):
        means = np.abs(global_values[:, :, k]).mean(axis=0)
        ranked = sorted(
            range(len(features)), key=lambda j: (-float(means[j]), features[j])
        )
        for rank, j in enumerate(ranked, 1):
            records.append(
                {
                    "class": label,
                    "feature": features[j],
                    "mean_abs_shap": float(means[j]),
                    "n_global_rows": int(global_mask.sum()),
                    "rank": rank,
                }
            )
        top = ranked[:15]
        fig, ax = plt.subplots(figsize=(9, max(3, len(top) * 0.38 + 1.2)))
        ax.barh(
            [features[j] for j in top][::-1],
            [means[j] for j in top][::-1],
            color="#2b6f9c",
        )
        ax.set_xlabel("Mean absolute SHAP value (raw class score)")
        ax.set_title(
            f"LightGBM · {split} · {label}\nGlobal sample n={int(global_mask.sum())}"
        )
        fig.tight_layout()
        fig.savefig(output / "figures" / f"{label}_importance.png", dpi=160)
        plt.close(fig)
        # Use the same feature order as the magnitude plot and a local RNG for
        # reproducible beeswarm jitter without changing caller random state.
        try:
            shap.summary_plot(
                global_values[:, top, k],
                global_X.iloc[:, top],
                feature_names=[features[j] for j in top],
                plot_type="dot",
                max_display=len(top),
                sort=False,
                show=False,
                plot_size=(9, max(3, len(top) * 0.4 + 1.5)),
                rng=np.random.default_rng(0),
            )
            fig = plt.gcf()
            plt.title(
                f"LightGBM · {split} · {label} · global sample n={int(global_mask.sum())}"
            )
            plt.xlabel("SHAP value (raw class score)")
            fig.tight_layout()
            fig.savefig(
                output / "figures" / f"{label}_beeswarm.png",
                dpi=160,
                bbox_inches="tight",
            )
        finally:
            plt.close("all")
    return pd.DataFrame(
        records, columns=["class", "feature", "mean_abs_shap", "n_global_rows", "rank"]
    )


def _draw_local(
    shap, plt, output, rows, X, values, base, raw, indices, cases, features
):
    selected_index = {int(full): i for i, full in enumerate(indices)}
    records, contributions = [], []
    for case_id, label in CASE_CLASSES.items():
        full = cases[case_id]
        if full is None:
            records.append(
                {
                    "case_id": case_id,
                    "status": "unavailable",
                    "reason": "no_matching_forecast",
                }
            )
            continue
        i, k = selected_index[full], CLASSES.index(label)
        row = rows.iloc[full]
        record = {
            name: row[name]
            for name in [
                "site_id",
                "date",
                "target_date",
                "observed_target_date",
                "y_true",
                "y_pred",
            ]
            + [f"proba_{c}" for c in CLASSES]
        }
        record.update(
            case_id=case_id,
            status="available",
            reason=None,
            explained_class=label,
            base_value=float(base[i, k]),
            raw_score=float(raw[i, k]),
        )
        records.append(record)
        for j, feature in enumerate(features):
            contributions.append(
                {
                    "case_id": case_id,
                    "class": label,
                    "feature": feature,
                    "feature_value": float(X.iloc[i, j]),
                    "shap_value": float(values[i, j, k]),
                }
            )
        explanation = shap.Explanation(
            values=values[i, :, k],
            base_values=base[i, k],
            data=X.iloc[i].to_numpy(),
            feature_names=features,
        )
        shap.plots.waterfall(
            explanation, max_display=min(15, len(features)), show=False
        )
        fig = plt.gcf()
        title = (
            f"{case_id}: {row['site_id']} · issue {row['date']}\n"
            f"Requested {row['target_date']} · observed {row['observed_target_date']}\n"
            f"True {row['y_true']} · predicted {row['y_pred']} · explaining {label} raw score\n"
            + " · ".join(f"P({c})={float(row[f'proba_{c}']):.3f}" for c in CLASSES)
        )
        # SHAP adds raw-score labels above its axes. Keep the multiline case
        # heading above those labels, including for short two-feature plots.
        fig.suptitle(title, fontsize=10, y=1 + 0.45 / fig.get_figheight(), va="bottom")
        fig.savefig(
            output / "figures" / f"{case_id}_waterfall.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)
    return pd.DataFrame(records, columns=LOCAL_COLUMNS), pd.DataFrame(
        contributions,
        columns=["case_id", "class", "feature", "feature_value", "shap_value"],
    )


def _runtime_versions():
    versions = {"python": platform.python_version()}
    for package in (
        "numpy",
        "pandas",
        "lightgbm",
        "shap",
        "matplotlib",
        "scikit-learn",
    ):
        versions[package] = metadata.version(package)
    return versions


def _write_report(output, manifest, rows, global_n, cases, error):
    lines = [
        "# LightGBM SHAP report",
        "",
        f"Run: `{manifest['run_id']}`. Split: `{manifest['evaluation_split']}`. Dataset: `{manifest.get('dataset_kind', 'unspecified')}`.",
        "",
        "[Evaluation report](../report.md). LightGBM is explained because it was evaluated; it need not be the winning model.",
        "",
        f"The recorded model reproduced all {len(rows):,} saved forecasts. Global explanations use {global_n:,} deterministically selected forecasts out of {len(rows):,} eligible forecasts.",
        "",
        f"All three classes passed raw-score additivity checks (maximum absolute error {error:.12g}; rtol={RTOL:g}, atol={ATOL:g}).",
        "",
        "Attributions describe raw multiclass scores, not probability-point changes. Predictions are not asserted to be calibrated. Tree-path-dependent SHAP reflects this model and sample; correlated history features can share attribution. Attribution does not prove physical causation, necessity, or forecast accuracy.",
        "",
        "## Global explanations",
        "",
    ]
    for label in CLASSES:
        lines.extend(
            [
                f"### {label}",
                "",
                f"![{label} mean absolute raw-score contribution](figures/{label}_importance.png)",
                "",
                f"![{label} raw-score beeswarm](figures/{label}_beeswarm.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## Local illustrations",
            "",
            "Cases use predetermined categories and key-hash ordering. They are retrospective illustrations, not representative estimates of error rates.",
            "",
        ]
    )
    for case_id, full in cases.items():
        lines.extend([f"### {case_id}", ""])
        if full is None:
            lines.extend(["Unavailable: no matching forecast.", ""])
        else:
            row = rows.iloc[full]
            lines.extend(
                [
                    f"Site `{row['site_id']}`, issue `{row['date']}`; requested target `{row['target_date']}`, observed target `{row['observed_target_date']}`. True `{row['y_true']}`; predicted `{row['y_pred']}`.",
                    "",
                    f"![{case_id} raw-score contributions](figures/{case_id}_waterfall.png)",
                    "",
                ]
            )
    lines.extend(
        [
            "All feature contributions, base values, scores, sample keys and hashes are saved beside this report. The metric bundle was not modified.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_explanations(evaluation_run, use_test=False):
    """Verify and explain the included LightGBM model without training it."""
    run_dir = Path(evaluation_run).resolve()
    # Verify the completed parent before writing anything into its directory.
    manifest = load_verified_run(run_dir, use_test=use_test)
    output = run_dir / "shap"
    if output.exists():
        raise EvaluationContractError(
            "E_OUTPUT_EXISTS",
            "SHAP output already exists; preserve it and use a new evaluation run.",
            {"path": str(output)},
        )
    output.mkdir()
    try:
        if "lightgbm" not in manifest.get("inputs", {}).get("models", {}):
            raise EvaluationContractError(
                "E_DEPENDENCY",
                "The evaluation must include LightGBM to create SHAP reports.",
            )
        lgb, shap, plt = _dependencies()
        booster, rows, X, raw_scores, receipt, artifact = _load_and_reproduce(
            manifest, run_dir, lgb
        )
        indices, global_mask, cases = _select_rows(rows)
        sample_X = X.iloc[indices].reset_index(drop=True)
        explainer = shap.TreeExplainer(
            booster, model_output="raw", feature_perturbation="tree_path_dependent"
        )
        try:
            native_values = explainer.shap_values(sample_X, check_additivity=True)
        except Exception as exc:
            raise EvaluationContractError(
                "E_SHAP_ADDITIVITY",
                "SHAP could not compute checked raw-score contributions.",
                {"cause": str(exc)},
            ) from exc
        values, base = _normalise_shap(
            native_values,
            explainer.expected_value,
            len(indices),
            len(receipt["features"]),
            receipt["class_order"],
        )
        sample_raw = raw_scores[indices]
        error = _check_additivity(values, base, sample_raw)
        (output / "figures").mkdir()
        sample_records = []
        for i, full in enumerate(indices):
            row = rows.iloc[full]
            sample_records.append(
                {
                    "site_id": row.site_id,
                    "date": row.date,
                    "is_global_sample": bool(global_mask[i]),
                    "local_case_ids": json.dumps(
                        [case for case, pos in cases.items() if pos == full]
                    ),
                }
            )
        pd.DataFrame(sample_records).to_csv(output / "sample_keys.csv", index=False)
        global_frame = _draw_global(
            shap,
            plt,
            output,
            sample_X,
            values,
            global_mask,
            receipt["features"],
            manifest["evaluation_split"],
        )
        global_frame.to_csv(output / "global_importance.csv", index=False)
        local, contributions = _draw_local(
            shap,
            plt,
            output,
            rows,
            sample_X,
            values,
            base,
            sample_raw,
            indices,
            cases,
            receipt["features"],
        )
        local.to_csv(output / "local_cases.csv", index=False)
        contributions.to_csv(output / "local_contributions.csv", index=False)
        np.savez_compressed(
            output / "values.npz",
            values=values,
            base_values=base,
            raw_scores=sample_raw,
            feature_values=sample_X.to_numpy(),
            feature_names=np.asarray(receipt["features"], dtype=str),
            class_names=np.asarray(CLASSES, dtype=str),
        )
        _write_report(output, manifest, rows, int(global_mask.sum()), cases, error)
        # Artifact producers may regenerate files while explanations render.
        # A completed bundle must still refer to the verified original inputs.
        if load_verified_run(run_dir, use_test=use_test) != manifest:
            raise EvaluationContractError(
                "E_HASH", "Parent evaluation changed during explanation."
            )
        audit = {
            "schema_version": 1,
            "status": "passed",
            "evaluation_split": manifest["evaluation_split"],
            "reproduced_forecasts": len(rows),
            "explained_rows": len(indices),
            "global_sample_rows": int(global_mask.sum()),
            "max_absolute_additivity_error": error,
            "checks": [
                "parent_input_output_hashes",
                "model_feature_class_iteration_schema",
                "full_population_probability_and_label_reproduction",
                "all_class_raw_score_additivity",
            ],
            "errors": [],
        }
        _json(output / "audit.json", audit)
        entry = manifest["inputs"]["models"]["lightgbm"]
        result = {
            "schema_version": 1,
            "status": "complete",
            "run_id": manifest["run_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_split": manifest["evaluation_split"],
            "dataset_kind": manifest.get("dataset_kind"),
            "evaluation_manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "artifact_sha256": artifact["sha256"],
            "reference_sha256": manifest["inputs"]["reference"]["sha256"],
            "receipt_sha256": entry["training_receipt"]["sha256"],
            "source_sha256": sha256_file(Path(__file__)),
            "features": receipt["features"],
            "model_class_order": receipt["class_order"],
            "class_order": CLASSES,
            "num_iterations": artifact["num_iterations"],
            "runtime_versions": _runtime_versions(),
            "explainer": {
                "model_output": "raw",
                "feature_perturbation": "tree_path_dependent",
                "check_additivity": True,
                "rtol": RTOL,
                "atol": ATOL,
            },
            "sampling": {
                "seed_token": "0",
                "algorithm": "ascending SHA256 of UTF-8 '0|site_id|ISO_date', then site_id/date",
                "limit": GLOBAL_SAMPLE_LIMIT,
                "global_n": int(global_mask.sum()),
                "eligible_n": len(rows),
                "population_sha256": manifest.get("population_sha256"),
                "keys_file": "sample_keys.csv",
            },
            "shapes": {
                "values": list(values.shape),
                "base_values": list(base.shape),
                "raw_scores": list(sample_raw.shape),
                "feature_values": list(sample_X.shape),
            },
            "max_absolute_additivity_error": error,
            "outputs": {
                str(path.relative_to(output)): sha256_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        }
        _json(output / "manifest.json", result)
        return output / "manifest.json"
    except Exception as exc:
        # Failed partial plots/arrays must not look like a usable explanation.
        for path in output.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        code = exc.code if isinstance(exc, EvaluationContractError) else "E_EXECUTION"
        details = getattr(exc, "details", None)
        _json(
            output / "audit.json",
            {
                "schema_version": 1,
                "status": "failed",
                "evaluation_split": manifest.get("evaluation_split"),
                "errors": [{"code": code, "message": str(exc), "details": details}],
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluation-run", required=True, type=Path)
    parser.add_argument("--use-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_explanations(args.evaluation_run, use_test=args.use_test)
    except EvaluationContractError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Explanation failed: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
