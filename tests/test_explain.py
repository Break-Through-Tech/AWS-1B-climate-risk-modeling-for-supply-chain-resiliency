"""Behavioral and real three-class synthetic integration checks for SHAP."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src import explain
from src.baselines import CLASSES
from src.evaluate import EvaluationContractError, run_evaluation, sha256_file

try:
    from tests.evaluation_fixtures import build_fixture
except ModuleNotFoundError:
    from evaluation_fixtures import build_fixture


class ShapeAndSelectionTests(unittest.TestCase):
    def test_both_supported_shapes_map_native_classes_explicitly(self):
        values = np.arange(24, dtype=float).reshape(2, 4, 3)
        classes = ["warming", "cooling", "neutral"]
        expected = values[:, :, [1, 2, 0]]
        for representation in (values, [values[:, :, k] for k in range(3)]):
            actual, base = explain._normalise_shap(
                representation, [10, 20, 30], 2, 4, classes
            )
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(base, [[20, 30, 10], [20, 30, 10]])
        row_base = np.array([[10, 20, 30], [11, 21, 31]])
        _, base = explain._normalise_shap(values, row_base, 2, 4, classes)
        np.testing.assert_array_equal(base, [[20, 30, 10], [21, 31, 11]])

    def test_unsupported_shapes_are_rejected_not_guessed(self):
        for values, base in [
            (np.zeros((3, 2, 4)), np.zeros(3)),
            (np.zeros((2, 4)), np.zeros(3)),
            ([np.zeros((2, 4))] * 2, np.zeros(3)),
            (np.zeros((2, 4, 3)), np.zeros((3, 2))),
            (np.full((2, 4, 3), np.nan), np.zeros(3)),
            (np.zeros((2, 4, 3)), np.full(3, np.inf)),
        ]:
            with self.subTest(shape=np.shape(base)):
                with self.assertRaises(EvaluationContractError) as caught:
                    explain._normalise_shap(values, base, 2, 4, list(CLASSES))
                self.assertEqual(caught.exception.code, "E_SHAP_SHAPE")

    def test_invalid_class_mapping_fails(self):
        with self.assertRaises(EvaluationContractError) as caught:
            explain._normalise_shap(
                np.zeros((2, 4, 3)), np.zeros(3), 2, 4, ["cooling"] * 3
            )
        self.assertEqual(caught.exception.code, "E_MODEL_SCHEMA")

    def test_independent_additivity_checks_all_classes(self):
        values = np.arange(24, dtype=float).reshape(2, 4, 3) / 10
        base = np.array([[0.1, 0.2, 0.3]] * 2)
        raw = values.sum(axis=1) + base
        self.assertEqual(explain._check_additivity(values, base, raw), 0.0)
        for class_id in range(3):
            bad = values.copy()
            bad[1, 0, class_id] += 0.01
            with self.assertRaises(EvaluationContractError) as caught:
                explain._check_additivity(bad, base, raw)
            self.assertEqual(caught.exception.code, "E_SHAP_ADDITIVITY")
        with self.assertRaises(EvaluationContractError):
            explain._check_additivity(values, base + 0.1, raw)
        with self.assertRaises(EvaluationContractError):
            explain._check_additivity(values, base, np.full((2, 3), np.nan))

    def test_global_sampling_and_local_cases_are_shuffle_invariant(self):
        count = 1200
        rows = pd.DataFrame(
            {
                "site_id": [f"SITE_{i % 4:03d}" for i in range(count)],
                "date": pd.date_range("1990-01-01", periods=count).strftime("%Y-%m-%d"),
                "y_true": np.resize(
                    ["warming", "warming", "cooling", "cooling"], count
                ),
                "y_pred": np.resize(
                    ["warming", "neutral", "warming", "cooling"], count
                ),
            }
        )

        def keyed(frame):
            positions, mask, cases = explain._select_rows(frame)
            keys = [(frame.iloc[i].site_id, frame.iloc[i].date) for i in positions]
            local = {
                case: (frame.iloc[i].site_id, frame.iloc[i].date)
                if i is not None
                else None
                for case, i in cases.items()
            }
            return keys, np.asarray(keys)[mask].tolist(), local

        a = keyed(rows)
        b = keyed(rows.sample(frac=1, random_state=41).reset_index(drop=True))
        self.assertEqual(a, b)
        self.assertEqual(len(a[1]), 1000)
        self.assertLessEqual(len(a[0]), 1004)
        self.assertTrue(all(key is not None for key in a[2].values()))
        expected = sorted(
            zip(rows.site_id, rows.date),
            key=lambda key: (
                hashlib.sha256(f"0|{key[0]}|{key[1]}".encode()).hexdigest(),
                *key,
            ),
        )[:1000]
        self.assertEqual(a[1], [list(k) for k in expected])

    def test_empty_local_categories_are_not_substituted(self):
        rows = pd.DataFrame(
            {
                "site_id": ["A", "B"],
                "date": ["1995-01-01"] * 2,
                "y_true": ["neutral"] * 2,
                "y_pred": ["neutral"] * 2,
            }
        )
        _, mask, cases = explain._select_rows(rows)
        self.assertTrue(mask.all())
        self.assertEqual(cases, {name: None for name in explain.CASE_CLASSES})

    def test_missing_optional_dependencies_report_contract_error(self):
        with patch.object(
            explain.importlib,
            "import_module",
            side_effect=OSError("native runtime missing"),
        ):
            with self.assertRaises(EvaluationContractError) as caught:
                explain._dependencies()
        self.assertEqual(caught.exception.code, "E_DEPENDENCY")

    def test_metric_import_does_not_load_optional_explanation_packages(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import src.evaluate; assert 'lightgbm' not in sys.modules; assert 'shap' not in sys.modules",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class ModelReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        reference = pd.DataFrame(
            {
                "site_id": ["A", "B", "C"],
                "date": ["1995-01-01"] * 3,
                "f1": [0.0, 1.0, 2.0],
                "f2": [np.nan, 0.2, 0.3],
            }
        )
        reference.to_csv(self.directory / "reference.csv", index=False)
        self.proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
        self.order = ["warming", "cooling", "neutral"]
        self.rows = pd.DataFrame(
            {
                "model": ["lightgbm"] * 3,
                "site_id": ["A", "B", "C"],
                "date": ["1995-01-01"] * 3,
                "split": ["validation"] * 3,
                "target_date": ["1995-04-01"] * 3,
                "observed_target_date": ["1995-04-01"] * 3,
                "y_true": list(CLASSES),
                "y_pred": ["warming", "cooling", "neutral"],
            }
        )
        for c in CLASSES:
            self.rows[f"proba_{c}"] = self.proba[:, self.order.index(c)]
        self.rows.to_csv(self.directory / "aligned_predictions.csv", index=False)
        self.receipt = {"features": ["f1", "f2"], "class_order": self.order}
        self.manifest = {
            "evaluation_split": "validation",
            "inputs": {
                "reference": {"path": str(self.directory / "reference.csv")},
                "models": {
                    "lightgbm": {
                        "receipt": self.receipt,
                        "model_artifact": {
                            "path": "fixture.txt",
                            "format": "lightgbm_text",
                            "num_iterations": 2,
                        },
                    }
                },
            },
        }
        self.booster = Mock()
        self.booster.dump_model.return_value = {"objective": "multiclass num_class:3"}
        self.booster.num_model_per_iteration.return_value = 3
        self.booster.current_iteration.return_value = 2
        self.booster.num_feature.return_value = 2
        self.booster.feature_name.return_value = ["f1", "f2"]
        self.booster.predict.side_effect = lambda X, **kw: (
            np.log(self.proba) if kw.get("raw_score") else self.proba
        )
        self.lgb = Mock()
        self.lgb.Booster.return_value = self.booster

    def load(self):
        return explain._load_and_reproduce(self.manifest, self.directory, self.lgb)

    def test_reproduction_maps_classes_and_preserves_missing_features(self):
        _, _, X, raw, _, _ = self.load()
        self.assertTrue(np.isnan(X.iloc[0, 1]))
        self.assertEqual(X.columns.tolist(), ["f1", "f2"])
        np.testing.assert_allclose(raw, np.log(self.proba)[:, [1, 2, 0]])
        for call in self.booster.predict.call_args_list:
            self.assertEqual(call.kwargs["num_iteration"], 2)
            self.assertEqual(call.kwargs["start_iteration"], 0)

    def test_wrong_feature_order_or_iteration_is_rejected(self):
        for change in ("order", "iterations", "objective"):
            with self.subTest(change=change):
                original = copy.deepcopy(self.manifest)
                if change == "order":
                    self.receipt["features"] = ["f2", "f1"]
                elif change == "iterations":
                    self.manifest["inputs"]["models"]["lightgbm"]["model_artifact"][
                        "num_iterations"
                    ] = 3
                else:
                    self.booster.dump_model.return_value = {
                        "objective": "multiclassova num_class:3"
                    }
                with self.assertRaises(EvaluationContractError) as caught:
                    self.load()
                self.assertEqual(caught.exception.code, "E_MODEL_SCHEMA")
                self.manifest = original
                self.receipt = self.manifest["inputs"]["models"]["lightgbm"]["receipt"]
                self.booster.dump_model.return_value = {
                    "objective": "multiclass num_class:3"
                }

    def test_forbidden_or_missing_or_nonnumeric_feature_rejected(self):
        for features in (["date", "f2"], ["missing", "f2"]):
            with self.subTest(features=features):
                self.receipt["features"] = features
                self.booster.feature_name.return_value = features
                with self.assertRaises(EvaluationContractError) as caught:
                    self.load()
                self.assertEqual(caught.exception.code, "E_MODEL_SCHEMA")
        self.receipt["features"] = ["f1", "f2"]
        self.booster.feature_name.return_value = ["f1", "f2"]
        ref = pd.read_csv(self.directory / "reference.csv")
        ref["f1"] = ["no", "numeric", "encoder"]
        ref.to_csv(self.directory / "reference.csv", index=False)
        with self.assertRaises(EvaluationContractError) as caught:
            self.load()
        self.assertEqual(caught.exception.code, "E_MODEL_SCHEMA")

    def test_stale_probabilities_or_wrong_native_mapping_fails(self):
        bad = self.rows.copy()
        bad.loc[0, "proba_warming"] -= 0.01
        bad.loc[0, "proba_cooling"] += 0.01
        bad.to_csv(self.directory / "aligned_predictions.csv", index=False)
        with self.assertRaises(EvaluationContractError) as caught:
            self.load()
        self.assertEqual(caught.exception.code, "E_MODEL_REPRODUCTION")
        self.rows.to_csv(self.directory / "aligned_predictions.csv", index=False)
        self.receipt["class_order"] = list(CLASSES)
        with self.assertRaises(EvaluationContractError) as caught:
            self.load()
        self.assertEqual(caught.exception.code, "E_MODEL_REPRODUCTION")


class WorkflowGateTests(unittest.TestCase):
    def test_test_gate_is_enforced_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                explain,
                "load_verified_run",
                side_effect=EvaluationContractError(
                    "E_TEST_GATE", "Test requires explicit flag"
                ),
            ):
                with self.assertRaises(EvaluationContractError) as caught:
                    explain.run_explanations(tmp)
            self.assertEqual(caught.exception.code, "E_TEST_GATE")
            self.assertFalse((Path(tmp) / "shap").exists())

    def test_failure_has_only_failed_audit_and_preserves_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "report.md"
            parent.write_text("existing metric report")
            with patch.object(
                explain,
                "load_verified_run",
                return_value={
                    "evaluation_split": "validation",
                    "inputs": {"models": {}},
                },
            ):
                with self.assertRaises(EvaluationContractError) as caught:
                    explain.run_explanations(tmp)
            self.assertEqual(caught.exception.code, "E_DEPENDENCY")
            self.assertEqual(
                [p.name for p in (Path(tmp) / "shap").iterdir()], ["audit.json"]
            )
            self.assertEqual(
                json.loads((Path(tmp) / "shap/audit.json").read_text())["status"],
                "failed",
            )
            self.assertEqual(parent.read_text(), "existing metric report")

    def test_existing_shap_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "shap"
            out.mkdir()
            (out / "marker.txt").write_text("preserve")
            with patch.object(explain, "load_verified_run", return_value={}):
                with self.assertRaises(EvaluationContractError) as caught:
                    explain.run_explanations(tmp)
            self.assertEqual(caught.exception.code, "E_OUTPUT_EXISTS")
            self.assertEqual((out / "marker.txt").read_text(), "preserve")


class RealMulticlassIntegrationTests(unittest.TestCase):
    def test_model_change_during_explanation_prevents_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_fixture(root / "inputs", include_lightgbm=True)
            run = run_evaluation(fixture["manifest_path"], root / "reports")
            original = explain._write_report

            def change_artifact(*args, **kwargs):
                original(*args, **kwargs)
                with fixture["model_path"].open("a") as stream:
                    stream.write("\n")

            with patch.object(explain, "_write_report", side_effect=change_artifact):
                with self.assertRaises(EvaluationContractError) as caught:
                    explain.run_explanations(run)
            self.assertEqual(caught.exception.code, "E_HASH")
            self.assertEqual([p.name for p in (run / "shap").iterdir()], ["audit.json"])

    def test_actual_saved_model_and_shap_work_even_when_persistence_wins(self):
        # No skip: the full evaluation environment must prove the native path.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_fixture(
                root / "inputs",
                include_lightgbm=True,
                lightgbm_class_order=["warming", "cooling", "neutral"],
            )
            run = run_evaluation(fixture["manifest_path"], root / "reports")
            selection = json.loads((run / "selection.json").read_text())
            self.assertEqual(selection["selected_model"], "persistence")
            before = sha256_file(run / "manifest.json")
            random_state = np.random.get_state()
            result = explain.run_explanations(run)
            for expected, actual in zip(random_state, np.random.get_state()):
                if isinstance(expected, np.ndarray):
                    np.testing.assert_array_equal(expected, actual)
                else:
                    self.assertEqual(expected, actual)
            self.assertEqual(before, sha256_file(run / "manifest.json"))
            metadata = json.loads(result.read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["class_order"], list(CLASSES))
            self.assertEqual(
                metadata["model_class_order"], ["warming", "cooling", "neutral"]
            )
            self.assertLess(metadata["max_absolute_additivity_error"], 1e-6)
            for relative, digest in metadata["outputs"].items():
                self.assertEqual(sha256_file(result.parent / relative), digest)
            with np.load(result.parent / "values.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["values"].shape[2], 3)
                self.assertEqual(arrays["feature_names"].tolist(), ["f1", "f2"])
                self.assertTrue(np.isnan(arrays["feature_values"]).any())
                np.testing.assert_allclose(
                    arrays["base_values"] + arrays["values"].sum(axis=1),
                    arrays["raw_scores"],
                    rtol=1e-5,
                    atol=1e-6,
                )
            global_frame = pd.read_csv(result.parent / "global_importance.csv")
            self.assertEqual(len(global_frame), 6)
            self.assertEqual(set(global_frame["class"]), set(CLASSES))
            local = pd.read_csv(
                result.parent / "local_cases.csv", keep_default_na=False
            )
            self.assertEqual(len(local), 4)
            for row in local.itertuples():
                plot = result.parent / "figures" / f"{row.case_id}_waterfall.png"
                self.assertEqual(plot.exists(), row.status == "available")
                if row.status == "unavailable":
                    self.assertEqual(row.reason, "no_matching_forecast")
            for label in CLASSES:
                self.assertGreater(
                    (result.parent / "figures" / f"{label}_importance.png")
                    .stat()
                    .st_size,
                    1000,
                )
                self.assertGreater(
                    (result.parent / "figures" / f"{label}_beeswarm.png")
                    .stat()
                    .st_size,
                    1000,
                )
            # A later read validates the unchanged metric bundle despite SHAP's
            # separately completed subdirectory.
            explain.load_verified_run(run)
            with self.assertRaises(EvaluationContractError) as caught:
                explain.run_explanations(run)
            self.assertEqual(caught.exception.code, "E_OUTPUT_EXISTS")


if __name__ == "__main__":
    unittest.main()
