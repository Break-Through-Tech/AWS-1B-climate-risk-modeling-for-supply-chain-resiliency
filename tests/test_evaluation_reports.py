"""Report behavior on small, independently understandable forecast populations."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src._evaluation_reports import (
    DIAGNOSTIC_COLUMNS,
    LEADERBOARD_COLUMNS,
    PER_CLASS_COLUMNS,
    RULE_ID,
    TIMELINE_COLUMNS,
    diagnostic_table,
    metric_tables,
    normalized_confusion,
    select_model,
    select_timeline_sites,
    timeline_grid,
    timeline_table,
    write_reports,
)
from src.baselines import CLASSES, score


def fixture():
    observations = [
        ("SITE_A", "1995-01-01", "cooling", "neutral", 150.0),
        ("SITE_A", "1995-01-03", "neutral", "", 150.0),
        ("SITE_B", "1995-01-01", "warming", "warming", 160.0),
        ("SITE_B", "1995-01-03", "neutral", "cooling", 160.0),
        ("SITE_C", "1995-01-01", "warming", "warming", -180.0),
        ("SITE_D", "1995-01-01", "neutral", "neutral", -120.0),
    ]
    ref = pd.DataFrame(
        observations,
        columns=["site_id", "date", "target_class", "current_class", "nominal_lon"],
    )
    ref["nominal_lat"] = 0.0
    ref["split"] = "validation"
    ref["target_date"] = (
        pd.to_datetime(ref["date"]) + pd.Timedelta(days=90)
    ).dt.strftime("%Y-%m-%d")
    ref["observed_target_date"] = (
        pd.to_datetime(ref["target_date"]) + pd.Timedelta(days=2)
    ).dt.strftime("%Y-%m-%d")
    parts, metrics = [], {}
    for model in ["majority", "persistence", "logistic", "lightgbm"]:
        part = ref[
            [
                "site_id",
                "date",
                "split",
                "target_date",
                "observed_target_date",
                "target_class",
            ]
        ].rename(columns={"target_class": "y_true"})
        part.insert(0, "model", model)
        part["y_pred"] = "neutral" if model == "majority" else part["y_true"]
        for label in CLASSES:
            part[f"proba_{label}"] = part["y_pred"].eq(label).astype(float)
        parts.append(part)
        metrics[model] = {"validation": score(part["y_true"], part["y_pred"])}
    aligned = pd.concat(parts, ignore_index=True)
    audit = {
        "splits": {
            "validation": {
                "reference_n": 8,
                "eligible_n": 6,
                "excluded_n": 2,
                "primary_exclusions": {"missing_target": 2},
            }
        },
        "models": {name: {"prediction_coverage": 1.0} for name in metrics},
        "final_comparison_ready": False,
    }
    config = {
        "run_id": "report-fixture",
        "dataset_kind": "synthetic",
        "evaluation_split": "validation",
        "target_contract": {"id": "local-ssta-3class-90d-tol7-v1", "status": "interim"},
    }
    return aligned, ref, metrics, audit, config


class MetricReportTest(unittest.TestCase):
    def test_unrounded_values_shared_denominator_and_class_order(self):
        _, _, metrics, audit, _ = fixture()
        board, per_class = metric_tables(metrics, "validation", audit)
        self.assertEqual(board.columns.tolist(), LEADERBOARD_COLUMNS)
        self.assertEqual(per_class.columns.tolist(), PER_CLASS_COLUMNS)
        majority = board.set_index("model").loc["majority"]
        self.assertAlmostEqual(majority.macro_f1, 2 / 9)
        self.assertEqual(majority.accuracy, 0.5)
        self.assertEqual(majority.label_coverage, 0.75)
        self.assertAlmostEqual(majority.delta_macro_f1_vs_persistence, 2 / 9 - 1)
        self.assertEqual(
            per_class.loc[per_class.model.eq("majority"), "class"].tolist(), CLASSES
        )
        self.assertEqual(
            board.model.tolist(), ["lightgbm", "logistic", "persistence", "majority"]
        )

    def test_missing_persistence_delta_is_undefined(self):
        _, _, metrics, audit, _ = fixture()
        board, _ = metric_tables({"majority": metrics["majority"]}, "validation", audit)
        self.assertTrue(pd.isna(board.iloc[0].delta_macro_f1_vs_persistence))

    def test_diagnostics_share_subsets_and_preserve_support(self):
        aligned, ref, _, _, _ = fixture()
        result = diagnostic_table(aligned, ref)
        self.assertEqual(result.columns.tolist(), DIAGNOSTIC_COLUMNS)
        for model in ["majority", "persistence", "logistic", "lightgbm"]:
            rows = result[result.model.eq(model)]
            for kind in ["issue_year", "site_id", "persistence_context", "transition"]:
                self.assertEqual(rows.loc[rows.group_type.eq(kind), "n"].sum(), 6)
            year = rows[rows.group_type.eq("issue_year")].iloc[0]
            self.assertEqual(
                (year.cooling_support, year.neutral_support, year.warming_support),
                (1, 3, 2),
            )
            transitions = rows[rows.group_type.eq("transition")].set_index(
                "group_value"
            )
            self.assertEqual(transitions.loc["current_class_missing", "n"], 1)
            self.assertEqual(transitions.loc["changed_class", "n"], 2)
            self.assertEqual(transitions.loc["same_class", "n"], 3)
        pd.testing.assert_frame_equal(
            result,
            diagnostic_table(
                aligned.sample(frac=1, random_state=4),
                ref.sample(frac=1, random_state=5),
            ),
        )

    def test_zero_support_normalization_is_nan(self):
        result = normalized_confusion([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
        np.testing.assert_array_equal(result[1], [0, 1, 0])
        self.assertTrue(np.isnan(result[[0, 2]]).all())


class SelectionTest(unittest.TestCase):
    def test_tie_then_simplicity_and_no_lightgbm_privilege(self):
        aligned, ref, metrics, audit, config = fixture()
        result = select_model(
            metrics, "validation", config, audit, select_timeline_sites(aligned, ref)
        )
        self.assertEqual(result["selected_model"], "persistence")
        self.assertEqual(
            result["macro_f1_tied_models"], ["lightgbm", "logistic", "persistence"]
        )
        self.assertEqual(result["status"], "provisional")
        self.assertEqual(result["rule_id"], RULE_ID)

    def test_extreme_recall_breaks_numerical_f1_tie(self):
        _, _, metrics, audit, config = fixture()
        metrics["persistence"]["validation"]["per_class"]["warming"]["recall"] = 0.7
        metrics["logistic"]["validation"]["per_class"]["cooling"]["recall"] = 0.8
        result = select_model(metrics, "validation", config, audit, [])
        self.assertEqual(result["selected_model"], "lightgbm")
        self.assertEqual(result["recall_tied_models"], ["lightgbm"])

    def test_tradeoff_does_not_invent_veto(self):
        _, _, metrics, audit, config = fixture()
        metrics["lightgbm"]["validation"]["macro_f1"] = 0.95
        metrics["logistic"]["validation"]["macro_f1"] = 0.97
        metrics["persistence"]["validation"]["macro_f1"] = 0.9
        metrics["logistic"]["validation"]["per_class"]["warming"]["recall"] = 0.4
        result = select_model(metrics, "validation", config, audit, [])
        self.assertEqual(result["selected_model"], "logistic")
        self.assertEqual(
            result["recall_tradeoffs"],
            [
                {
                    "class": "warming",
                    "selected_recall": 0.4,
                    "persistence_recall": 1.0,
                    "delta": -0.6,
                }
            ],
        )

    def test_final_status_requires_evidence(self):
        _, _, metrics, audit, config = fixture()
        config.update(dataset_kind="final")
        config["target_contract"]["status"] = "team_final"
        self.assertEqual(
            select_model(metrics, "validation", config, audit, [])["status"],
            "provisional",
        )
        audit["final_comparison_ready"] = True
        self.assertEqual(
            select_model(metrics, "validation", config, audit, [])["status"],
            "frozen_ready",
        )

    def test_test_does_not_reselect_winner_or_sites(self):
        aligned, ref, metrics, audit, config = fixture()
        frozen = select_model(
            metrics, "validation", config, audit, select_timeline_sites(aligned, ref)
        )
        frozen["status"] = "frozen_ready"
        original = copy.deepcopy(frozen)
        test_metrics = {
            m: {"test": copy.deepcopy(v["validation"])} for m, v in metrics.items()
        }
        test_metrics["lightgbm"]["test"]["macro_f1"] = 1.0
        test_metrics["persistence"]["test"]["macro_f1"] = 0.0
        result = select_model(test_metrics, "test", config, audit, [], frozen)
        self.assertEqual(result["selected_model"], "persistence")
        self.assertEqual(result["timeline_sites"], original["timeline_sites"])
        self.assertEqual(result["status"], "test_assessment")
        self.assertEqual(frozen, original)
        with self.assertRaises(ValueError):
            select_model(test_metrics, "test", config, audit, [], None)


class TimelineTest(unittest.TestCase):
    def test_sites_use_coverage_geographic_endpoints_and_lexical_ties(self):
        aligned, ref, _, _, _ = fixture()
        selected = select_timeline_sites(aligned, ref)
        self.assertEqual(
            [s["site_id"] for s in selected], ["SITE_A", "SITE_C", "SITE_D"]
        )
        self.assertEqual([s["eligible_validation_n"] for s in selected], [2, 1, 1])
        shuffled = aligned.sample(frac=1, random_state=1).copy()
        shuffled["y_pred"] = "warming"
        self.assertEqual(
            selected,
            select_timeline_sites(shuffled, ref.sample(frac=1, random_state=2)),
        )
        csv_reference = ref.copy()
        csv_reference["nominal_lon"] = csv_reference["nominal_lon"].astype(str)
        self.assertEqual(selected, select_timeline_sites(aligned, csv_reference))

    def test_target_alignment_actual_delay_and_blank_day(self):
        aligned, ref, _, _, _ = fixture()
        frame = timeline_table(aligned, select_timeline_sites(aligned, ref))
        self.assertEqual(frame.columns.tolist(), TIMELINE_COLUMNS)
        self.assertTrue(
            (
                (pd.to_datetime(frame.target_date) - pd.to_datetime(frame.date)).dt.days
                == 90
            ).all()
        )
        self.assertTrue(
            (
                (
                    pd.to_datetime(frame.observed_target_date)
                    - pd.to_datetime(frame.target_date)
                ).dt.days
                == 2
            ).all()
        )
        west = frame[frame.region.eq("west")]
        grid = timeline_grid(
            west, ["majority", "persistence"], pd.date_range("1995-04-01", "1995-04-03")
        )
        self.assertTrue(np.isnan(grid[:, 1]).all())
        np.testing.assert_array_equal(grid[0, [0, 2]], [0, 1])
        np.testing.assert_array_equal(grid[2, [0, 2]], [0, 1])

    def test_empty_region_is_explicit_and_disagreeing_truth_fails(self):
        aligned, ref, _, _, _ = fixture()
        selected = select_timeline_sites(aligned[aligned.site_id.eq("SITE_A")], ref)
        self.assertEqual(
            selected[1],
            {
                "region": "central",
                "site_id": None,
                "eligible_validation_n": 0,
                "reason": "no_eligible_site",
            },
        )
        frame = timeline_table(aligned, selected)
        frame.loc[frame.index[0], "y_true"] = "warming"
        with self.assertRaisesRegex(ValueError, "disagree"):
            timeline_grid(
                frame,
                sorted(aligned.model.unique()),
                pd.date_range("1995-04-01", "1995-04-03"),
            )


class ReportBundleTest(unittest.TestCase):
    def test_complete_report_artifacts_and_no_state_mutation(self):
        aligned, ref, metrics, audit, config = fixture()
        before = aligned.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            result = write_reports(aligned, ref, audit, config, metrics, directory)
            self.assertEqual(result["selection"]["selected_model"], "persistence")
            self.assertEqual(len(result["files"]), 16)
            for relative in result["files"]:
                path = Path(directory) / relative
                self.assertTrue(path.is_file(), relative)
                self.assertGreater(path.stat().st_size, 0, relative)
                if path.suffix == ".png":
                    self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            saved = json.loads((Path(directory) / "selection.json").read_text())
            self.assertEqual(saved, result["selection"])
            report = (Path(directory) / "report.md").read_text()
            self.assertIn("**synthetic**", report)
            self.assertIn("even if another model won", report)
            self.assertIn("not event-onset", report)
            self.assertNotIn("statistically significant improvement", report)
        pd.testing.assert_frame_equal(aligned, before)


if __name__ == "__main__":
    unittest.main()
