"""Exercise final-handoff plumbing with synthetic data and test-only declarations.

These temporary metadata records are not team acknowledgments or upstream
scientific evidence. They test how the evaluator checks and binds such records.
"""

import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src import evaluate as ev
from src.baselines import C_GRID
from tests.evaluation_fixtures import (
    build_fixture,
    file_ref,
    refresh_hashes,
    write_json,
)


def declared_handoff(fixture):
    """Add explicitly artificial declaration fixtures, never project approvals."""
    manifest = json.loads(fixture["manifest_path"].read_text())
    manifest["dataset_kind"] = "final"
    manifest["target_contract"].update(status="team_final", builder_commit="1" * 40)
    directory = fixture["manifest_path"].parent
    review = {
        "schema_version": 1,
        "team_agreement": {
            "status": "confirmed",
            "target_definition": copy.deepcopy(ev.TARGET),
            "cleaned_input": file_ref(fixture["reference_path"]),
            "approvals": [
                {
                    "member": "SYNTHETIC TEST PRINCIPAL",
                    "roles": [
                        "target",
                        "cleaning",
                        "features",
                        "baselines",
                        "lightgbm",
                    ],
                    "confirmed_at_utc": "2026-01-01T00:00:00Z",
                    "evidence": "Unit-test declaration only; not a real team approval.",
                }
            ],
        },
        "checks": {
            f"U{i:02d}": {
                "status": "passed",
                "evidence": "Synthetic declaration for contract testing only",
            }
            for i in range(1, 7)
        },
        "source_commits": ["1" * 40],
        "known_limitations": [
            "All observations and evidence in this temporary test are synthetic."
        ],
    }
    review_path = write_json(directory / "review.json", review)
    manifest["upstream_review"] = file_ref(review_path)
    for name, path in fixture["receipts"].items():
        receipt = json.loads(path.read_text())
        receipt["training_code_commit"] = "1" * 40
        if name in ("logistic", "lightgbm"):
            receipt["selection_source"] = "train_cv"
            recipe = json.loads(Path(receipt["selection_settings"]["path"]).read_text())
            settings_path = Path(receipt["settings"]["path"])
            settings = json.loads(settings_path.read_text())
            parameters = recipe["parameters"]
            grid = (
                [dict(parameters, C=c) for c in C_GRID]
                if name == "logistic"
                else [parameters]
            )
            candidates = [
                {
                    "id": str(i),
                    "parameters": p,
                    "fold_macro_f1": [0.9] * 3 if p == parameters else [0.1] * 3,
                    "mean_macro_f1": 0.9 if p == parameters else 0.1,
                }
                for i, p in enumerate(grid)
            ]
            folds = [
                dict(
                    zip(("fit_start", "fit_end", "score_start", "score_end"), dates),
                    fit_keys_sha256="2" * 64,
                    scoring_keys_sha256="3" * 64,
                    fit_n=3,
                    scoring_n=3,
                    scoring_reference_n=3,
                    scoring_class_support={c: 1 for c in ev.CLASSES},
                )
                for dates in ev.FOLDS
            ]
            settings["model_selection"] = {
                "protocol_id": "expanding_calendar_3fold_v1",
                "folds": folds,
                "candidates": candidates,
                "selected_candidate": next(
                    c["id"] for c in candidates if c["parameters"] == parameters
                ),
            }
            write_json(settings_path, settings)
        write_json(path, receipt)
    write_json(fixture["manifest_path"], manifest)
    refresh_hashes(fixture["manifest_path"])
    return manifest


class HandoffTest(unittest.TestCase):
    def test_full_frozen_test_handoff_and_recipe_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = build_fixture(root / "validation-input", include_lightgbm=True)
            declared_handoff(validation)
            validation_run = ev.run_evaluation(
                validation["manifest_path"], root / "reports"
            )
            selected = json.loads((validation_run / "selection.json").read_text())
            self.assertEqual(selected["status"], "frozen_ready")

            test = build_fixture(
                root / "test-input",
                include_lightgbm=True,
                evaluation_split="test",
                run_id="synthetic-test-handoff",
            )
            config = declared_handoff(test)
            freeze = {
                "schema_version": 1,
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "target_contract_id": ev.TARGET["id"],
                "reference_sha256": config["reference"]["sha256"],
                "validation_run_manifest": file_ref(validation_run / "manifest.json"),
                "rule_id": selected["rule_id"],
                "selected_model": selected["selected_model"],
                "models": [
                    {
                        "name": name,
                        "selection_settings": json.loads(path.read_text())[
                            "selection_settings"
                        ],
                    }
                    for name, path in validation["receipts"].items()
                ],
                "decision_note": "Synthetic test of freeze binding; not a project selection.",
            }
            frozen_path = write_json(root / "frozen-selection.json", freeze)
            config = json.loads(test["manifest_path"].read_text())
            config["final_selection"] = file_ref(frozen_path)
            write_json(test["manifest_path"], config)
            test_run = ev.run_evaluation(
                test["manifest_path"], root / "reports", use_test=True
            )
            ev.load_verified_run(test_run, use_test=True)
            test_selection = json.loads((test_run / "selection.json").read_text())
            self.assertEqual(test_selection["status"], "test_assessment")
            self.assertEqual(
                test_selection["selected_model"], selected["selected_model"]
            )
            self.assertEqual(
                test_selection["timeline_sites"], selected["timeline_sites"]
            )
            self.assertEqual(
                json.loads((test_run / "metrics.json").read_text())["persistence"][
                    "test"
                ]["n"],
                15,
            )

            # A changed recipe cannot be silently accepted after the freeze.
            receipt = json.loads(test["receipts"]["majority"].read_text())
            recipe_path = Path(receipt["selection_settings"]["path"])
            recipe = json.loads(recipe_path.read_text())
            recipe["recipe"] += " changed after freeze"
            write_json(recipe_path, recipe)
            config["run_id"] = "changed-test-recipe"
            write_json(test["manifest_path"], config)
            refresh_hashes(test["manifest_path"])
            with self.assertRaises(ev.EvaluationContractError) as caught:
                ev.run_evaluation(
                    test["manifest_path"], root / "reports", use_test=True
                )
            self.assertEqual(caught.exception.code, "E_TEST_GATE")

    def test_owner_coverage_and_upstream_uniqueness_evidence_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_fixture(tmp)
            config = declared_handoff(fixture)
            review_path = Path(config["upstream_review"]["path"])
            original = json.loads(review_path.read_text())
            for mutate in (
                lambda r: r["team_agreement"]["approvals"][0].update(roles=["target"]),
                lambda r: r["checks"]["U06"].update(status="failed"),
            ):
                review = copy.deepcopy(original)
                mutate(review)
                write_json(review_path, review)
                config["upstream_review"] = file_ref(review_path)
                with self.assertRaises(ev.EvaluationContractError) as caught:
                    ev._review(config, fixture["manifest_path"].parent)
                self.assertEqual(caught.exception.code, "E_UPSTREAM_INVALID")


if __name__ == "__main__":
    unittest.main()
