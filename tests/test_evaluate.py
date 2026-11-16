"""Behavioral tests for saved-prediction evaluation; no project data are read."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import evaluate as ev
from src.baselines import C_GRID, CLASSES
from tests.evaluation_fixtures import (
    build_fixture,
    refresh_hashes,
    target_contract,
    write_json,
)


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False)


def metric_frame(truth, predicted):
    rows = []
    for i, (actual, guess) in enumerate(zip(truth, predicted)):
        rows.append(
            {
                "model": "logistic",
                "site_id": f"S{i}",
                "date": "1995-01-01",
                "split": "validation",
                "target_date": "1995-04-01",
                "observed_target_date": "1995-04-01",
                "y_true": actual,
                "y_pred": guess,
                **{f"proba_{c}": float(guess == c) for c in CLASSES},
            }
        )
    return pd.DataFrame(rows, columns=ev.ALIGNED_COLUMNS)


class MetricTest(unittest.TestCase):
    def test_hand_calculated_all_neutral_predictions(self):
        result = ev.score_predictions(
            metric_frame(["cooling", "neutral", "warming", "neutral"], ["neutral"] * 4)
        )["logistic"]["validation"]
        self.assertAlmostEqual(result["macro_f1"], 2 / 9)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(
            result["confusion_matrix"]["rows_true_cols_pred"],
            [[0, 1, 0], [0, 2, 0], [0, 1, 0]],
        )
        self.assertEqual(result["per_class"]["neutral"]["precision"], 0.5)
        self.assertEqual(result["per_class"]["neutral"]["recall"], 1.0)
        self.assertEqual(
            [result["per_class"][c]["support"] for c in CLASSES], [1, 2, 1]
        )

    def test_perfect_three_class_predictions(self):
        result = ev.score_predictions(metric_frame(CLASSES, CLASSES))["logistic"][
            "validation"
        ]
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(
            result["confusion_matrix"]["rows_true_cols_pred"],
            np.eye(3, dtype=int).tolist(),
        )

    def test_absent_classes_remain_in_macro_denominator(self):
        result = ev.score_predictions(metric_frame(["neutral"], ["neutral"]))[
            "logistic"
        ]["validation"]
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["per_class"]["warming"]["support"], 0)


class AlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        fixture = build_fixture(Path(cls.temp.name))
        cls.reference = read_csv(fixture["reference_path"])
        cls.predictions = read_csv(fixture["predictions_path"])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_alignment(self, reference=None, predictions=None, models=None):
        ref = self.reference.copy() if reference is None else reference
        pred = self.predictions.copy() if predictions is None else predictions
        mapping = models or {
            name: pred.copy() for name in ("majority", "persistence", "logistic")
        }
        return ev.validate_and_align(ref, mapping, target_contract(), "validation")

    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(ev.EvaluationContractError) as raised:
            function(*args, **kwargs)
        self.assertEqual(raised.exception.code, code, raised.exception)

    def test_shuffle_and_nonunique_dataframe_index_invariance(self):
        baseline, audit = self.run_alignment()
        ref = self.reference.sample(frac=1, random_state=10)
        ref.index = [0] * len(ref)
        predictions = {
            n: self.predictions.sample(frac=1, random_state=i)
            for i, n in enumerate(["majority", "persistence", "logistic"])
        }
        for frame in predictions.values():
            frame.index = [0] * len(frame)
        actual, checked = self.run_alignment(reference=ref, models=predictions)
        pd.testing.assert_frame_equal(baseline, actual)
        self.assertEqual(audit["population_sha256"], checked["population_sha256"])

    def test_duplicate_reference_and_prediction_keys_rejected(self):
        self.assert_code(
            "E_DUPLICATE_KEY",
            self.run_alignment,
            reference=pd.concat(
                [self.reference, self.reference.iloc[[0]]], ignore_index=True
            ),
        )
        self.assert_code(
            "E_DUPLICATE_KEY",
            self.run_alignment,
            predictions=pd.concat(
                [self.predictions, self.predictions.iloc[[0]]], ignore_index=True
            ),
        )

    def test_missing_or_extra_forecasts_rejected(self):
        self.assert_code(
            "E_PREDICTION_COVERAGE",
            self.run_alignment,
            predictions=self.predictions.iloc[1:].copy(),
        )
        extra = self.predictions.iloc[[0]].copy()
        extra["site_id"] = "EXTRA"
        self.assert_code(
            "E_PREDICTION_COVERAGE",
            self.run_alignment,
            predictions=pd.concat([self.predictions, extra], ignore_index=True),
        )

    def test_redundant_truth_split_and_target_fields_must_agree(self):
        for field, bad in [
            ("y_true", "warming"),
            ("split", "train"),
            ("target_date", "1995-04-02"),
        ]:
            with self.subTest(field=field):
                pred = self.predictions.copy()
                pred.loc[0, field] = bad
                self.assert_code(
                    "E_TRUTH_MISMATCH", self.run_alignment, predictions=pred
                )
        pred = self.predictions.copy()
        pred["observed_target_date"] = pred.target_date
        pred.loc[0, "observed_target_date"] = "1995-04-02"
        self.assert_code("E_TRUTH_MISMATCH", self.run_alignment, predictions=pred)

    def test_prediction_metadata_collision_is_a_schema_error(self):
        for field in ["target_class", "current_class", "nominal_lat"]:
            with self.subTest(field=field):
                pred = self.predictions.copy()
                pred[field] = "unexpected"
                self.assert_code("E_SCHEMA", self.run_alignment, predictions=pred)

    def test_unknown_and_missing_prediction_labels_fail(self):
        for value in ["Warming", "unknown", "", None]:
            with self.subTest(value=value):
                pred = self.predictions.copy()
                pred.loc[0, "y_pred"] = value
                self.assert_code("E_LABEL", self.run_alignment, predictions=pred)

    def test_missing_target_pair_is_excluded_with_reconciled_counts(self):
        ref = self.reference.copy()
        index = ref.index[(ref.split == "validation") & ref.target_class.ne("")][0]
        key = tuple(ref.loc[index, ev.KEYS])
        ref.loc[index, ["target_class", "observed_target_date"]] = ""
        pred = self.predictions.loc[
            ~(self.predictions.site_id.eq(key[0]) & self.predictions.date.eq(key[1]))
        ].copy()
        _, audit = self.run_alignment(ref, pred)
        counts = audit["splits"]["validation"]
        self.assertEqual(counts["eligible_n"], 23)
        self.assertEqual(
            counts["reference_n"],
            counts["eligible_n"] + sum(counts["primary_exclusions"].values()),
        )

    def test_one_sided_target_missingness_fails(self):
        for field in ["target_class", "observed_target_date"]:
            ref = self.reference.copy()
            ref.loc[0, field] = ""
            self.assert_code("E_TARGET_PAIR", self.run_alignment, reference=ref)

    def test_requested_inside_actual_outside_training_boundary_is_excluded(self):
        row = self.reference.iloc[0].to_dict()
        row.update(
            site_id="SITE_056",
            date="1994-10-02",
            split="train",
            target_date="1994-12-31",
            observed_target_date="1995-01-01",
            target_class="warming",
            current_class="neutral",
        )
        ref = pd.concat([self.reference, pd.DataFrame([row])], ignore_index=True)
        _, audit = self.run_alignment(reference=ref)
        self.assertEqual(
            audit["splits"]["train"]["primary_exclusions"][
                "observed_target_after_split_end"
            ],
            1,
        )
        self.assertEqual(audit["splits"]["train"]["eligible_n"], 24)

    def test_actual_target_at_split_end_is_eligible(self):
        row = self.reference.iloc[0].to_dict()
        row.update(
            site_id="SITE_056",
            date="1994-10-02",
            split="train",
            target_date="1994-12-31",
            observed_target_date="1994-12-31",
            target_class="warming",
            current_class="neutral",
        )
        donor = dict(
            row,
            date="1994-12-31",
            target_date="1995-03-31",
            observed_target_date="",
            target_class="",
            current_class="warming",
        )
        ref = pd.concat([self.reference, pd.DataFrame([row, donor])], ignore_index=True)
        _, audit = self.run_alignment(
            reference=ref, models={"logistic": self.predictions}
        )
        self.assertEqual(audit["splits"]["train"]["eligible_n"], 25)

    def test_target_window_rejects_before_request_and_eighth_day(self):
        for offset in [-1, 8]:
            ref = self.reference.copy()
            ref.loc[0, "observed_target_date"] = (
                pd.Timestamp(ref.loc[0, "target_date"]) + pd.Timedelta(days=offset)
            ).strftime("%Y-%m-%d")
            self.assert_code("E_TARGET_WINDOW", self.run_alignment, reference=ref)

    def test_seventh_day_is_accepted_with_exact_donor(self):
        ref = self.reference.copy()
        index = ref.index[ref.split.eq("validation") & ref.target_class.ne("")][0]
        observed = pd.Timestamp(ref.loc[index, "target_date"]) + pd.Timedelta(days=7)
        ref.loc[index, "observed_target_date"] = observed.strftime("%Y-%m-%d")
        donor = ref.loc[index].to_dict()
        donor.update(
            date=observed.strftime("%Y-%m-%d"),
            target_date=(observed + pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
            current_class=ref.loc[index, "target_class"],
            target_class="",
            observed_target_date="",
        )
        ref = pd.concat([ref, pd.DataFrame([donor])], ignore_index=True)
        _, audit = self.run_alignment(reference=ref)
        self.assertEqual(audit["splits"]["validation"]["eligible_n"], 24)

    def test_target_contract_and_horizon_are_fixed(self):
        for key, value in [
            ("horizon_unit", "calendar_months"),
            ("horizon_value", 89),
            ("tolerance_days", 8),
        ]:
            contract = target_contract()
            contract[key] = value
            self.assert_code(
                "E_TARGET_CONTRACT",
                ev.validate_and_align,
                self.reference,
                {"logistic": self.predictions},
                contract,
                "validation",
            )
        ref = self.reference.copy()
        ref.loc[0, "target_date"] = "1993-04-02"
        self.assert_code("E_TARGET_CONTRACT", self.run_alignment, reference=ref)

    def test_wrong_same_site_donor_rejected(self):
        ref = self.reference.copy()
        donor_mask = ref.site_id.eq(ref.loc[0, "site_id"]) & ref.date.eq(
            ref.loc[0, "observed_target_date"]
        )
        ref.loc[donor_mask, "current_class"] = "warming"
        self.assert_code("E_TARGET_PAIR", self.run_alignment, reference=ref)

    def test_probability_validation_never_repairs_values(self):
        for value in [-0.1, 1.1, np.inf, np.nan, "bad", 0.25]:
            with self.subTest(value=value):
                pred = self.predictions.copy()
                pred["proba_cooling"] = pred.proba_cooling.astype(object)
                pred.loc[0, "proba_cooling"] = value
                self.assert_code("E_PROBABILITY", self.run_alignment, predictions=pred)

    def test_probability_column_order_and_model_class_ties(self):
        pred = (
            self.predictions[self.predictions.model.eq("logistic")]
            .copy()
            .reset_index(drop=True)
        )
        pred.loc[0, ev.PROBA] = [0.5, 0.5, 0.0]
        pred.loc[0, "y_pred"] = "neutral"
        pred = pred[list(reversed(pred.columns))]
        pred.attrs["class_order"] = ["neutral", "cooling", "warming"]
        aligned, _ = self.run_alignment(models={"logistic": pred})
        self.assertEqual(aligned.iloc[0].y_pred, "neutral")
        pred.loc[0, "y_pred"] = "cooling"
        self.assert_code("E_PROBABILITY", self.run_alignment, models={"logistic": pred})

    def test_persistence_fallback_rule_is_checked(self):
        ref = self.reference.copy()
        mask = (
            ref.split.eq("validation")
            & ref.date.eq("1995-01-01")
            & ref.site_id.eq("SYNTH_002")
        )
        ref.loc[mask, "current_class"] = ""
        pred = self.predictions.copy()
        row = (
            pred.model.eq("persistence")
            & pred.site_id.eq("SYNTH_002")
            & pred.date.eq("1995-01-01")
        )
        self.assert_code("E_LABEL", self.run_alignment, reference=ref, predictions=pred)
        pred.loc[row, "y_pred"] = "cooling"
        pred.loc[row, ev.PROBA] = [1.0, 0.0, 0.0]
        self.run_alignment(reference=ref, predictions=pred)

    def test_test_outcomes_are_not_inspected_by_validation(self):
        baseline, audit = self.run_alignment()
        ref = self.reference.copy()
        ref.loc[
            ref.split.eq("test"),
            ["target_class", "current_class", "observed_target_date", "target_date"],
        ] = "uninspected"
        extra = self.predictions.iloc[[0]].copy()
        extra.loc[:, "date"] = "1997-01-01"
        extra.loc[:, "split"] = "test"
        for column in ["y_true", "y_pred", *ev.PROBA]:
            extra[column] = "uninspected"
        actual, changed = self.run_alignment(
            reference=ref,
            predictions=pd.concat([self.predictions, extra], ignore_index=True),
        )
        pd.testing.assert_frame_equal(actual, baseline)
        self.assertEqual(audit["population_sha256"], changed["population_sha256"])
        self.assertNotIn("test", changed["splits"])

    def test_empty_evaluation_population_is_an_error(self):
        ref = self.reference.copy()
        ref.loc[
            ref.split.eq("validation"), ["target_class", "observed_target_date"]
        ] = ""
        self.assert_code("E_EMPTY_POPULATION", self.run_alignment, reference=ref)

    def test_malformed_dates_and_changing_site_coordinates_fail(self):
        for field, value in [("date", "1993-01-01T00:00:00"), ("nominal_lon", 181.0)]:
            ref = self.reference.copy()
            ref.loc[0, field] = value
            self.assert_code("E_SCHEMA", self.run_alignment, reference=ref)
        ref = self.reference.copy()
        ref.loc[0, "nominal_lat"] += 0.1
        self.assert_code("E_SCHEMA", self.run_alignment, reference=ref)


class RunTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = build_fixture(self.root / "input")
        self.manifest = self.fixture["manifest_path"]

    def tearDown(self):
        self.temp.cleanup()

    def expect_failure(self, code, **kwargs):
        with self.assertRaises(ev.EvaluationContractError) as raised:
            ev.run_evaluation(self.manifest, self.root / "outputs", **kwargs)
        self.assertEqual(raised.exception.code, code, raised.exception)
        run = self.root / "outputs" / "synthetic-validation"
        self.assertEqual([p.name for p in run.iterdir()], ["audit.json"])
        self.assertEqual(
            json.loads((run / "audit.json").read_text())["status"], "failed"
        )

    def test_successful_bundle_hashes_and_no_overwrite(self):
        run = ev.run_evaluation(self.manifest, self.root / "outputs")
        manifest = ev.load_verified_run(run)
        self.assertEqual(manifest["evaluation_split"], "validation")
        self.assertEqual(
            manifest["inputs"]["models"]["logistic"]["receipt"]["fit_splits"], ["train"]
        )
        self.assertEqual(
            json.loads((run / "selection.json").read_text())["selected_model"],
            "persistence",
        )
        before = {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        with self.assertRaises(ev.EvaluationContractError) as raised:
            ev.run_evaluation(self.manifest, self.root / "outputs")
        self.assertEqual(raised.exception.code, "E_OUTPUT_EXISTS")
        self.assertEqual(
            before, {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        )
        (run / "shap").mkdir()
        (run / "shap" / "placeholder").write_text(
            "Separate SHAP outputs do not alter recorded metric files"
        )
        ev.load_verified_run(run)

    def test_reference_hash_change_fails_before_reports(self):
        with self.fixture["reference_path"].open("a") as stream:
            stream.write("\n")
        self.expect_failure("E_HASH")

    def test_fit_keys_including_real_boundary_pattern_are_blocked(self):
        ref = read_csv(self.fixture["reference_path"])
        row = ref.iloc[0].to_dict()
        row.update(
            site_id="SITE_056",
            date="1994-10-02",
            split="train",
            target_date="1994-12-31",
            observed_target_date="1995-01-01",
            target_class="warming",
            current_class="neutral",
        )
        pd.concat([ref, pd.DataFrame([row])], ignore_index=True).to_csv(
            self.fixture["reference_path"], index=False
        )
        keys = read_csv(self.fixture["fit_rows_path"])
        pd.concat(
            [keys, pd.DataFrame([{"site_id": "SITE_056", "date": "1994-10-02"}])],
            ignore_index=True,
        ).to_csv(self.fixture["fit_rows_path"], index=False)
        refresh_hashes(self.manifest)
        self.expect_failure("E_FIT_POPULATION")

    def test_duplicate_fit_keys_are_blocked(self):
        keys = read_csv(self.fixture["fit_rows_path"])
        pd.concat([keys, keys.iloc[[0]]], ignore_index=True).to_csv(
            self.fixture["fit_rows_path"], index=False
        )
        refresh_hashes(self.manifest)
        self.expect_failure("E_DUPLICATE_KEY")

    def test_test_gate_both_directions_and_missing_freeze(self):
        self.expect_failure("E_TEST_GATE", use_test=True)
        # Use distinct run IDs because failed bundles are also immutable.
        config = json.loads(self.manifest.read_text())
        for run_id, flag in [
            ("test-without-flag", False),
            ("test-without-freeze", True),
        ]:
            config.update(run_id=run_id, evaluation_split="test")
            write_json(self.manifest, config)
            with self.assertRaises(ev.EvaluationContractError) as raised:
                ev.run_evaluation(self.manifest, self.root / "outputs", use_test=flag)
            self.assertEqual(raised.exception.code, "E_TEST_GATE")

    def test_final_project_data_without_agreement_is_blocked(self):
        config = json.loads(self.manifest.read_text())
        config["dataset_kind"] = "final"
        config["target_contract"].update(status="team_final", builder_commit="1" * 40)
        write_json(self.manifest, config)
        self.expect_failure("E_DEPENDENCY")

    def test_legacy_validation_tuning_is_blocked(self):
        receipt = json.loads(self.fixture["receipts"]["logistic"].read_text())
        receipt["selection_source"] = "validation"
        write_json(self.fixture["receipts"]["logistic"], receipt)
        refresh_hashes(self.manifest)
        self.expect_failure("E_UPSTREAM_INVALID")

    def test_stale_embedded_receipt_and_resolved_input_fail_reloading(self):
        run = ev.run_evaluation(self.manifest, self.root / "outputs")
        original = json.loads((run / "manifest.json").read_text())
        stale = copy.deepcopy(original)
        stale["inputs"]["models"]["logistic"]["receipt"]["features"] = ["wrong"]
        write_json(run / "manifest.json", stale)
        with self.assertRaises(ev.EvaluationContractError) as raised:
            ev.load_verified_run(run)
        self.assertEqual(raised.exception.code, "E_HASH")
        stale = copy.deepcopy(original)
        stale["inputs"]["models"]["logistic"]["model_artifact"] = None
        write_json(run / "manifest.json", stale)
        with self.assertRaises(ev.EvaluationContractError) as raised:
            ev.load_verified_run(run)
        self.assertEqual(raised.exception.code, "E_HASH")

    def test_modified_completed_output_is_detected(self):
        run = ev.run_evaluation(self.manifest, self.root / "outputs")
        with (run / "metrics.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaises(ev.EvaluationContractError) as raised:
            ev.load_verified_run(run)
        self.assertEqual(raised.exception.code, "E_HASH")

    def test_output_cleanup_after_unexpected_writer_failure(self):
        with patch(
            "src._evaluation_reports.write_reports",
            side_effect=RuntimeError("fixture writer failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "fixture writer failure"):
                ev.run_evaluation(self.manifest, self.root / "outputs")
        run = self.root / "outputs" / "synthetic-validation"
        self.assertEqual([p.name for p in run.iterdir()], ["audit.json"])

    def test_unknown_manifest_field_rejected(self):
        config = json.loads(self.manifest.read_text())
        config["skip_validation"] = True
        write_json(self.manifest, config)
        self.expect_failure("E_SCHEMA")

    def test_wrong_json_types_are_contract_errors(self):
        original = json.loads(self.manifest.read_text())
        mutations = [
            lambda c: c.update(dataset_kind={}),
            lambda c: c.update(evaluation_split=[]),
            lambda c: c["models"][0].update(name=[]),
            lambda c: c["target_contract"].update(status={}),
        ]
        for mutate in mutations:
            config = copy.deepcopy(original)
            mutate(config)
            with self.assertRaises(ev.EvaluationContractError):
                ev._validate_config(config, False)

    def test_input_change_during_report_generation_prevents_completion(self):
        def change_input(*args, **kwargs):
            with self.fixture["reference_path"].open("a") as stream:
                stream.write("\n")
            return {"warnings": [], "files": []}

        with patch("src._evaluation_reports.write_reports", side_effect=change_input):
            self.expect_failure("E_HASH")


class CVEvidenceTest(unittest.TestCase):
    def cv(self):
        folds = []
        for i, dates in enumerate(ev.FOLDS):
            folds.append(
                dict(
                    zip(["fit_start", "fit_end", "score_start", "score_end"], dates),
                    fit_keys_sha256="1" * 64,
                    scoring_keys_sha256="2" * 64,
                    fit_n=30,
                    scoring_n=3 * (i + 1),
                    scoring_reference_n=30,
                    scoring_class_support={c: i + 1 for c in CLASSES},
                )
            )
        candidates = [
            {
                "id": str(c),
                "parameters": {"C": c},
                "fold_macro_f1": [0.2, 0.4, 0.9],
                "mean_macro_f1": 0.5,
            }
            for c in C_GRID
        ]
        return {
            "model_selection": {
                "protocol_id": "expanding_calendar_3fold_v1",
                "folds": folds,
                "candidates": candidates,
                "selected_candidate": str(C_GRID[0]),
            }
        }

    def test_equal_fold_weight_and_regularization_tie(self):
        settings = self.cv()
        recipe = {"parameters": {"C": C_GRID[0]}}
        ev._cv_evidence(settings, recipe, "logistic")
        settings["model_selection"]["candidates"][0]["mean_macro_f1"] = (
            0.2 + 0.8 + 2.7
        ) / 6
        with self.assertRaises(ev.EvaluationContractError) as raised:
            ev._cv_evidence(settings, recipe, "logistic")
        self.assertEqual(raised.exception.code, "E_UPSTREAM_INVALID")

    def test_wrong_fold_or_candidate_selection_is_rejected(self):
        for mutate in [
            lambda cv: cv["folds"][0].update(fit_end="1994-12-31"),
            lambda cv: cv.update(selected_candidate=str(C_GRID[-1])),
        ]:
            settings = self.cv()
            mutate(settings["model_selection"])
            with self.assertRaises(ev.EvaluationContractError):
                ev._cv_evidence(settings, {"parameters": {"C": C_GRID[0]}}, "logistic")


if __name__ == "__main__":
    unittest.main()
