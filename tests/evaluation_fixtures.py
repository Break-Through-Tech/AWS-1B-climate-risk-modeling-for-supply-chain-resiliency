"""Small, entirely synthetic artifacts for evaluation integration tests.

No project observations are read. The optional LightGBM model and logistic
pipeline are actually fitted and serialized; persistence deliberately wins
because the validation feature relationship is reversed in this fixture.
"""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.baselines import CLASSES, make_logistic


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def file_ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def target_contract():
    return {
        "id": "local-ssta-3class-90d-tol7-v1",
        "status": "interim",
        "classes": list(CLASSES),
        "threshold_c": 0.5,
        "threshold_policy": "strict_outer_inclusive_neutral",
        "horizon_unit": "days",
        "horizon_value": 90,
        "match_policy": "first_nonmissing_class_on_or_after",
        "tolerance_days": 7,
        "climatology_fit_splits": ["train"],
        "builder_commit": "synthetic_fixture",
    }


def build_fixture(
    directory,
    include_lightgbm=False,
    evaluation_split="validation",
    run_id="synthetic-validation",
    lightgbm_class_order=None,
):
    """Return paths to complete synthetic inputs (test freeze is supplied separately)."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    periods = [
        ("train", "1993-01-01", 9),
        ("validation", "1995-01-01", 9),
        ("test", "1997-01-01", 6),
    ]
    rows = []
    for split, start, count in periods:
        dates = pd.date_range(start, periods=count, freq="90D")
        for site, label in enumerate(CLASSES):
            for index, date in enumerate(dates):
                has_target = index + 1 < count
                rows.append(
                    {
                        "site_id": f"SYNTH_{site:03d}",
                        "date": date.strftime("%Y-%m-%d"),
                        "split": split,
                        "target_date": (date + pd.Timedelta(days=90)).strftime(
                            "%Y-%m-%d"
                        ),
                        "observed_target_date": dates[index + 1].strftime("%Y-%m-%d")
                        if has_target
                        else "",
                        "target_class": label if has_target else "",
                        "current_class": label,
                        "nominal_lat": float(site - 1),
                        "nominal_lon": [150.0, -150.0, -100.0][site],
                        "f1": float(site if split == "train" else 2 - site),
                        "f2": float(index % 3) if index % 4 else np.nan,
                    }
                )
    reference = pd.DataFrame(rows).sort_values(["date", "site_id"], ignore_index=True)
    reference_path = directory / "reference.csv"
    reference.to_csv(reference_path, index=False, lineterminator="\n")
    fit_splits = (
        ["train"] if evaluation_split == "validation" else ["train", "validation"]
    )
    fit = reference[
        reference["split"].isin(fit_splits) & reference.target_class.ne("")
    ].copy()
    scored = reference[
        reference.split.eq(evaluation_split) & reference.target_class.ne("")
    ].copy()
    fit_path = directory / "fit_rows.csv"
    fit[["site_id", "date"]].to_csv(fit_path, index=False, lineterminator="\n")
    features = ["f1", "f2"]
    majority = str(fit.target_class.mode().iloc[0])
    logistic = make_logistic(0.1).fit(fit[features], fit.target_class)
    logistic_path = directory / "logistic.joblib"
    joblib.dump(logistic, logistic_path)
    model_data = {
        "majority": (np.repeat(majority, len(scored)), None),
        "persistence": (scored.current_class.to_numpy(), None),
        "logistic": (
            logistic.predict(scored[features]),
            logistic.predict_proba(scored[features]),
        ),
    }
    booster = None
    model_path = None
    order = list(lightgbm_class_order or CLASSES)
    lgb_params = None
    if include_lightgbm:
        import lightgbm as lgb

        lgb_params = {
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.2,
            "num_leaves": 4,
            "max_depth": 3,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "num_threads": 1,
            "seed": 0,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        encoded = fit.target_class.map({c: i for i, c in enumerate(order)}).to_numpy()
        booster = lgb.train(
            lgb_params, lgb.Dataset(fit[features], label=encoded), num_boost_round=8
        )
        model_path = directory / "lightgbm.txt"
        booster.save_model(str(model_path), num_iteration=booster.current_iteration())
        probabilities = booster.predict(
            scored[features], num_iteration=booster.current_iteration()
        )
        predictions = np.array(order)[np.argmax(probabilities, axis=1)]
        canonical = probabilities[:, [order.index(c) for c in CLASSES]]
        model_data["lightgbm"] = (predictions, canonical)
    prediction_frames = []
    for name, (predictions, probabilities) in model_data.items():
        frame = (
            scored[["site_id", "date", "target_date", "split", "target_class"]]
            .rename(columns={"target_class": "y_true"})
            .copy()
        )
        frame.insert(0, "model", name)
        frame["y_pred"] = predictions
        for i, label in enumerate(CLASSES):
            frame[f"proba_{label}"] = (
                probabilities[:, i]
                if probabilities is not None
                else (frame.y_pred == label).astype(float)
            )
        prediction_frames.append(frame)
    predictions_path = directory / "predictions.csv"
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        predictions_path, index=False, lineterminator="\n"
    )
    versions = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "scikit-learn", "joblib"] + (
        ["lightgbm"] if include_lightgbm else []
    ):
        versions[package] = importlib.metadata.version(package)
    receipts = {}
    entries = []
    for name in model_data:
        deterministic = name in ("majority", "persistence")
        chosen_features = [] if deterministic else features
        class_order = order if name == "lightgbm" else list(CLASSES)
        parameters = (
            {
                "rule": name,
                "tie_break": "alphabetical",
                "missing_current": "fit_majority",
            }
            if deterministic
            else {"C": 0.1, "seed": 0}
            if name == "logistic"
            else dict(lgb_params, num_iterations=booster.current_iteration())
        )
        selection_path = write_json(
            directory / f"{name}_selection.json",
            {
                "schema_version": 1,
                "model": name,
                "target_contract_id": target_contract()["id"],
                "features": chosen_features,
                "class_order": class_order,
                "recipe": "synthetic fixture " + name,
                "parameters": parameters,
            },
        )
        settings_path = write_json(
            directory / f"{name}_settings.json",
            {
                "synthetic_fixture": True,
                "parameters": parameters,
                "fit_splits": fit_splits,
            },
        )
        artifact = None
        if name == "logistic":
            artifact = dict(
                file_ref(logistic_path), format="sklearn_joblib", num_iterations=None
            )
        elif name == "lightgbm":
            artifact = dict(
                file_ref(model_path),
                format="lightgbm_text",
                num_iterations=booster.current_iteration(),
            )
        receipt_path = write_json(
            directory / f"{name}_receipt.json",
            {
                "schema_version": 1,
                "model": name,
                "prediction_split": evaluation_split,
                "reference_sha256": sha256(reference_path),
                "target_contract_id": target_contract()["id"],
                "training_code_commit": "synthetic_fixture",
                "fit_splits": fit_splits,
                "fit_rows": file_ref(fit_path),
                "preprocessing_fit_splits": ["train"],
                "features": chosen_features,
                "class_order": class_order,
                "probability_kind": "one_hot" if deterministic else "predict_proba",
                "majority_class": majority if deterministic else None,
                "settings": file_ref(settings_path),
                "selection_settings": file_ref(selection_path),
                "selection_source": "none",
                "model_artifact": artifact,
                "runtime_versions": versions,
            },
        )
        receipts[name] = receipt_path
        entries.append(
            {
                "name": name,
                "predictions": file_ref(predictions_path),
                "training_receipt": file_ref(receipt_path),
            }
        )
    manifest_path = write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "dataset_kind": "synthetic",
            "evaluation_split": evaluation_split,
            "reference": file_ref(reference_path),
            "target_contract": target_contract(),
            "models": entries,
            "upstream_review": None,
            "final_selection": None,
        },
    )
    return {
        "manifest_path": manifest_path,
        "reference_path": reference_path,
        "predictions_path": predictions_path,
        "receipts": receipts,
        "model_path": model_path,
        "fit_rows_path": fit_path,
    }


def refresh_hashes(manifest_path):
    """Refresh actual file hashes after a test deliberately edits fixture inputs."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    def refresh(ref, parent):
        if ref is None:
            return
        path = Path(ref["path"])
        if not path.is_absolute():
            path = parent / path
        ref["sha256"] = sha256(path)

    refresh(manifest["reference"], manifest_path.parent)
    for entry in manifest["models"]:
        refresh(entry["predictions"], manifest_path.parent)
        path = Path(entry["training_receipt"]["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        receipt = json.loads(path.read_text())
        receipt["reference_sha256"] = manifest["reference"]["sha256"]
        for key in ("fit_rows", "settings", "selection_settings", "model_artifact"):
            refresh(receipt[key], path.parent)
        write_json(path, receipt)
        refresh(entry["training_receipt"], manifest_path.parent)
    write_json(manifest_path, manifest)
    return manifest
