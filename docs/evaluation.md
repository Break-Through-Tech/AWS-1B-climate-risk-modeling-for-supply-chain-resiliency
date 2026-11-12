# Evaluate saved forecasts and explain LightGBM

The evaluation commands consume saved predictions. They check shared truth, target dates, fitting populations, probabilities, and artifact provenance before publishing a comparison. They do not train a project model, fix an upstream dataset, or choose new hyperparameters.

This guide documents the implemented input contracts, validation checks, and final-test gate. Passing synthetic checks establishes software readiness; project results require actual data, training artifacts, and upstream evidence. This implementation depends on the baseline module introduced by PR #25. Its existing scorer is reused; its current requested-date-only training filter is not reused as the evaluation eligibility rule.

## Set up the evaluation environment

Use Python 3.13 and the exact package versions recorded in `requirements-evaluation.txt`:

```bash
python3.13 -m venv .venv-evaluation
.venv-evaluation/bin/python -m pip install -r requirements-evaluation.txt
```

On macOS, LightGBM needs a compatible OpenMP runtime, normally supplied by `libomp` (for example, `brew install libomp` in an environment managed with Homebrew). If importing LightGBM reports that `libomp.dylib` cannot be loaded, resolve that runtime dependency before running SHAP.

For the verified macOS development environment, scikit-learn's installed wheel also includes a compatible runtime under `.venv-evaluation/lib/python3.13/site-packages/sklearn/.dylibs`. The following per-command environment setting was used to make that existing runtime discoverable; it is an environment workaround, not application code or a portable replacement for installing OpenMP:

```bash
DYLD_LIBRARY_PATH="$PWD/.venv-evaluation/lib/python3.13/site-packages/sklearn/.dylibs" \
  .venv-evaluation/bin/python -c 'import lightgbm, shap; print(lightgbm.__version__, shap.__version__)'
```

Use the same prefix for commands requiring LightGBM if this workaround is needed. The metric-only command imports neither LightGBM nor SHAP. In restricted environments, give Matplotlib/fontconfig a writable cache with `MPLCONFIGDIR=/tmp/evaluation-mpl XDG_CACHE_HOME=/tmp/evaluation-cache` as a command prefix.

## Start with a synthetic smoke run

Run from the repository root. This generates small synthetic inputs and actually fits small fixture models without reading the buoy dataset. The synthetic relationship intentionally lets persistence beat the learned models, so the SHAP path is exercised even when LightGBM does not win.

```bash
.venv-evaluation/bin/python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "tests")
from evaluation_fixtures import build_fixture

paths = build_fixture(Path("reports/evaluation/synthetic-inputs"), include_lightgbm=True)
print(paths["manifest_path"])
PY

.venv-evaluation/bin/python -m src.evaluate \
  --manifest reports/evaluation/synthetic-inputs/manifest.json

.venv-evaluation/bin/python -m src.explain \
  --evaluation-run reports/evaluation/synthetic-validation
```

Apply the macOS runtime prefix above when necessary. Synthetic results verify the software path; they are not real model-performance evidence. Runs refuse to overwrite an existing result directory. For another run, generate a fixture in a new input directory with a different `run_id`; preserve both previous inputs and outputs so recorded hashes remain reproducible.

## Project-data inputs

The evaluation pipeline consumes and verifies a shared reference table and split-specific model artifacts and predictions. The following items remain upstream requirements until actual producer evidence is delivered:

- Recorded team agreement on cooling/neutral/warming, neutral endpoints at ±0.5°C, a 90-day horizon, the first valid target through day 97 inclusive, and the exact cleaned-input version/hash. This guide is not a substitute for the owners' acknowledgments.
- An integrated table built from PR #24's cleaned sensor output with measured SST unchanged, the current canonical site mapping, and the selected target/features. Rows lacking usable targets must remain present so coverage can be reported.
- Passing U01–U06 upstream evidence, including `(site_id,date)` uniqueness before and after feature/target construction. Any multiple-measurement site/day policy belongs upstream and must be recorded/tested; the evaluator rejects duplicates.
- Regenerated baseline models/predictions after actual observed target dates are purged at fitting boundaries. Removing leaked training rows only from a report does not repair a fitted model.
- Training-only chronological CV evidence for logistic/LightGBM and the actual LightGBM model, predictions, ordered feature list, class mapping, and selected iteration count.

The final reference CSV requires `site_id,date,split,target_date,observed_target_date,target_class,current_class,nominal_lat,nominal_lon`, plus the model's feature columns. Dates are ISO `YYYY-MM-DD`; missing labels/observed dates are blank together. Identifiers and dates must be unique, split assignments must match `src.split`, and nominal coordinates must be stable within a site. Blank feature values may remain missing. Every model must predict every eligible selected-split key exactly once.

The baseline-compatible prediction CSV requires:

```text
model,site_id,date,target_date,split,y_true,y_pred,proba_cooling,proba_neutral,proba_warming
```

`observed_target_date` may be supplied; otherwise it is joined by the reference key. Truth comes from the reference, never from a prediction file alone. Probability columns are matched by name. Majority/persistence probabilities must be one-hot indicators; learned-model probabilities must be finite, in range and sum to one within tolerance.

### Manifest and path/hash rules

The input manifest has exactly these top-level fields:

```text
schema_version, run_id, dataset_kind, evaluation_split, reference,
target_contract, models, upstream_review, final_selection
```

`schema_version` is 1. `dataset_kind` is `synthetic`, `interim`, or `final`. The default evaluation split is supplied as `validation`; test requires both `evaluation_split="test"` and `--use-test`. The run ID uses lowercase letters/digits/underscores/hyphens and determines the new output directory.

Ordinary file references have exactly `path` and `sha256` (model artifacts additionally require `format` and `num_iterations`, described below), where the digest is the lowercase SHA-256 of the actual file bytes. Resolve manifest paths relative to the manifest's directory and receipt paths relative to the receipt's directory. Real producers must hash the completed files; do not replace missing hashes with guesses. The synthetic fixture generator is a runnable example of complete records with real hashes.

Each model entry has exactly `name`, `predictions`, and `training_receipt`. Names are `majority`, `persistence`, `logistic`, and `lightgbm`; the three baseline entries can share one prediction file. Project-data comparisons require all three baselines; final completion additionally requires LightGBM. The target contract and receipt fields are documented below. The model receipt identifies actual fitting keys, input hash, source commit, feature/class order, selection recipe, runtime versions and split-specific artifact.

Logistic artifacts use `sklearn_joblib` and are hashed by the evaluator without deserialization. LightGBM artifacts use `lightgbm_text`, already exported at the selected positive number of boosting iterations. A validation prediction file must refer to the train-fitted model, not a subsequently refitted train+validation model. Keep run-specific `settings` separate from frozen `selection_settings` so the final refit preserves the prediction recipe while legitimately changing fit membership.

### Target contract and training receipt

The manifest's `target_contract` contains every field in the `target_definition` JSON below, plus exactly `status` and `builder_commit`. Status is `interim` or `team_final`; `dataset_kind="final"` requires `team_final`. The builder commit is the actual full 40-character source commit. Only synthetic fixtures may use `synthetic_fixture` as a source commit.

Each `training_receipt` JSON has exactly these fields:

```text
schema_version, model, prediction_split, reference_sha256, target_contract_id,
training_code_commit, fit_splits, fit_rows, preprocessing_fit_splits, features,
class_order, probability_kind, majority_class, settings, selection_settings,
selection_source, model_artifact, runtime_versions
```

- `schema_version` is 1. `model` and `prediction_split` match the manifest entry and evaluation split. `reference_sha256` matches the shared CSV, and `target_contract_id` matches the target definition.
- `training_code_commit` identifies the actual training source. `fit_splits` is `["train"]` for validation or `["train", "validation"]` for final test. `preprocessing_fit_splits` remains `["train"]`.
- `fit_rows` references a CSV with exactly `site_id,date`. Its unique keys must equal the complete eligible fitting population. Both requested and observed target dates must stay inside each row's assigned partition; missing targets are ineligible.
- `features` is an ordered, unique, nonempty feature-name list for logistic/LightGBM and `[]` for deterministic baselines. Features must be numeric, may contain missing values, and cannot contain infinities or reserved metadata/target columns defined in `src.baselines.RESERVED`.
- `class_order` is a permutation of `cooling,neutral,warming` matching the saved model. `probability_kind` is `predict_proba` for learned models or `one_hot` for deterministic baselines.
- `majority_class` is null for learned models. Otherwise it is the fitting population's modal target class, with lexicographic tie-breaking, used as the majority/persistence fallback.
- `settings` and `selection_settings` are file references to JSON objects. `settings` records the actual run and, for learned project models, the chronological selection evidence below.
- `selection_source` is `train_cv` for learned project models and `none` for deterministic baselines. Synthetic learned fixtures alone may use `none`.
- `model_artifact` is null for deterministic baselines. Learned-model references have exactly `path,sha256,format,num_iterations`: logistic uses `sklearn_joblib` and null iterations; LightGBM uses `lightgbm_text` and a positive integer matching the selected recipe.
- `runtime_versions` is a nonempty object of actual version strings, including `python`.

The `selection_settings` JSON has exactly `schema_version,model,target_contract_id,features,class_order,recipe,parameters`. Version is 1; identity, features, and class order match the receipt. `recipe` is a nonempty description of the predictive procedure and `parameters` is its resolved parameter object. Preserve this file unchanged between validation and final test.

See [the synthetic fixture generator](../tests/evaluation_fixtures.py) for executable examples of complete manifests, receipts, recipes, and artifact references. Its artificial training records are test examples, not project evidence.

### Upstream review JSON

The existing `upstream_review` file is machine-readable JSON with exactly:

| Key | Content |
| --- | --- |
| `schema_version` | Integer 1 |
| `team_agreement` | Object described below |
| `checks` | Exactly `U01` through `U06`, each with `status` and `evidence` |
| `source_commits` | Nonempty array of actual full 40-character source commits |
| `known_limitations` | Array of factual strings, including relevant prior validation/test exposure |

`team_agreement` has exactly `status`, `target_definition`, `cleaned_input`, and `approvals`. Accepted project runs require `status="confirmed"`. Populate that status only from actual owner acknowledgments. `cleaned_input` is a verified file reference resolved relative to the review JSON. `target_definition` equals the following fixed scientific definition (no builder revision or maturity status in this nested object):

```json
{
  "id": "local-ssta-3class-90d-tol7-v1",
  "classes": ["cooling", "neutral", "warming"],
  "threshold_c": 0.5,
  "threshold_policy": "strict_outer_inclusive_neutral",
  "horizon_unit": "days",
  "horizon_value": 90,
  "match_policy": "first_nonmissing_class_on_or_after",
  "tolerance_days": 7,
  "climatology_fit_splits": ["train"]
}
```

Each approval has exactly `member`, `roles`, `confirmed_at_utc`, and `evidence`. Use the real member identity, actual UTC acknowledgment time, and a source reference to that acknowledgment. Roles across the recorded approvals must cover `target`, `cleaning`, `features`, `baselines`, and `lightgbm`; one owner can cover multiple roles. Each U01–U06 check must record `status="passed"` with a source/test-run evidence reference. The evaluator validates this record's structure and file binding; it cannot authenticate a person's agreement or infer that a claimed test really ran. Human review remains necessary. Never fabricate this record to bypass a missing handoff. Synthetic fixtures alone may use `upstream_review=null`.

The required upstream checks are:

| Check | Required evidence |
| --- | --- |
| U01 | With fitted preprocessing, site mapping, and model frozen, perturbing measurements after issue time leaves earlier/current features, missingness, predictions, and probabilities unchanged (`rtol=0`, `atol=1e-12`, equal missing values). Future target truth may change. |
| U02 | With the same frozen objects, full-record and issue-time-truncated feature computation agree at issue time, covering lags, windows, fills, site separation, and gaps. |
| U03 | Measurements after a fold's fit cutoff cannot affect fitted preprocessing, fitting labels/keys, or the deterministically fitted model; boundary-crossing labels are purged. |
| U04 | Target matching chooses the first same-site nonmissing class from day 90 through day 97 inclusive; missing donors, other sites, day 98, and duplicates are tested. |
| U05 | Both ±0.5°C endpoints are neutral; values beyond them are cooling/warming; missing anomalies remain missing. Requested-inside/observed-outside split and fold boundaries are tested independently. |
| U06 | Nonmissing, unique `(site_id,date)` keys are proven after the canonical site/date join and in the final feature/target table, including unlabeled rows. Any duplicate-resolution policy is documented and tested upstream. |

### Chronological selection evidence

Real logistic/LightGBM receipts use `selection_source="train_cv"`. Their `settings` JSON contains a `model_selection` object with exactly `protocol_id`, `folds`, `candidates`, and `selected_candidate`. The protocol ID is `expanding_calendar_3fold_v1`.

| Fold | Fit period | Scoring period |
| --- | --- | --- |
| 1 | 1980-03-07–1988-12-31 | 1989-01-01–1990-12-31 |
| 2 | 1980-03-07–1990-12-31 | 1991-01-01–1992-12-31 |
| 3 | 1980-03-07–1992-12-31 | 1993-01-01–1994-12-31 |

Each fold object has `fit_start,fit_end,score_start,score_end,fit_keys_sha256,scoring_keys_sha256,fit_n,scoring_n,scoring_reference_n,scoring_class_support`. The support object has exactly cooling/neutral/warming counts; these must be positive and sum to `scoring_n`, which cannot exceed `scoring_reference_n`. Each candidate has `id,parameters,fold_macro_f1,mean_macro_f1`. Record three finite fold scores and their unweighted arithmetic mean. The selected candidate must match the ranking/tie rule and the frozen recipe's actual parameters.

Logistic evidence records the existing seven C values: `1e-5,1e-4,1e-3,1e-2,0.1,1.0,10.0`; numerical ties within `1e-12` prefer smaller C. LightGBM candidates include fixed positive `num_iterations`; ties prefer fewer iterations, then canonical parameter JSON order. The LightGBM training settings supply the actual bounded grid. Each fold refits climatology and learned preprocessing on its fitting portion and purges actual label dates at its boundaries. The evaluation code checks the supplied evidence, not the training execution itself.

## Run validation, explanations, and final test

```bash
.venv-evaluation/bin/python -m src.evaluate --manifest path/to/validation-manifest.json
.venv-evaluation/bin/python -m src.explain --evaluation-run reports/evaluation/<run_id>
```

The evaluator writes a new `reports/evaluation/<run_id>/` directory. `--output-root PATH` can select another parent directory. Exit status is 0 for success, 2 for a contract/dependency error and 1 for an unexpected runtime failure. Expected errors include a machine-readable code and specific details. A failed validly identified run retains `audit.json` but has no completed manifest/report bundle. Existing run directories are preserved and never overwritten.

SHAP runs whenever LightGBM is included among the compared models, even if persistence, logistic or majority wins. It reloads the exact native booster, checks features/classes/iteration count, reproduces saved probabilities, and verifies all three classes' raw-score additivity before writing its separate `shap/` completion manifest. It uses a deterministic maximum-1,000-row global sample and predetermined local-case categories. Raw SHAP contributions are not probability-point effects or causal explanations.

For final test, first obtain the actual frozen selection record and split-specific refitted model/prediction receipts. The record references a completed final validation run, all frozen recipes, the prescribed rule ID and its selected model. Shared data and target identity must match; the evaluator does not create this decision retrospectively or retrain anything.

The final-selection JSON has exactly `schema_version,frozen_at_utc,target_contract_id,reference_sha256,validation_run_manifest,rule_id,selected_model,models,decision_note`. Version is 1. The freeze time is UTC and cannot precede the completed validation run. `decision_note` is nonempty. `validation_run_manifest` is a file reference to that run's `manifest.json`; its recorded inputs and outputs must still verify. Target/reference identity and model set must match the test run. `rule_id` and `selected_model` must match validation's `selection.json`, whose status must be `frozen_ready`. Each entry in `models` has exactly `name,selection_settings`, with the recipe hash unchanged from validation and matching the test receipt. File references resolve relative to the final-selection JSON. Validation input manifests use `final_selection=null`.

```bash
.venv-evaluation/bin/python -m src.evaluate \
  --manifest path/to/test-manifest.json --use-test
```

Optional test SHAP also requires `--use-test` with the test run directory. Test reporting retains the validation winner and timeline sites even if another model scores higher on test. It does not retune, choose new sites, or relabel the test winner as the previously selected model.

## Read the outputs

- `metrics.json` keeps the baseline scorer's exact model/split structure. Macro F1 always averages all three labels with zero-division scores set to zero.
- `leaderboard.csv` compares shared-population F1, accuracy, persistence deltas, coverage and warming/cooling recall. `per_class_metrics.csv` includes support and absent-class flags.
- `diagnostics.csv` includes every nonempty site/year/persistence-context/transition slice. Transition groups are retrospective. Small support and absent classes limit interpretation.
- `selection.json` applies validation Macro F1, then minimum warming/cooling recall for numerical ties, then the simplicity order `majority`, `persistence`, `logistic`, `lightgbm`. Numerical ties use an absolute tolerance of `1e-12`. Provisional records do not establish final selection. Test records assess the previously frozen choice.
- `timeline.csv` and `figures/local_condition_timeline.png` align forecasts to requested target dates and retain actual verifying dates. White gaps remain missing; local class sequences are not official ENSO event detection.
- `aligned_predictions.csv`, `exclusions.csv`, `audit.json` and `manifest.json` make population, failure/exclusion accounting and file/source provenance reviewable. Both requested and observed target dates must stay inside the example's own split.
- `report.md` links the tables/figures and states limitations. `shap/report.md` is separate so adding explanations does not mutate the completed metric bundle or invalidate its hashes.

The baseline probabilities are deterministic encodings; learned probabilities are not claimed to be calibrated. No significance, causal influence, business savings or operational event lead is established by these reports. Historical validation/test-era exploratory exposure and retrospective site construction remain disclosed limitations.

## Troubleshoot without weakening the checks

| Error | Resolution |
| --- | --- |
| `E_DEPENDENCY` | Supply the actual required artifact/environment dependency; do not substitute an unrelated model |
| `E_HASH` | Trace stale/mismatched bytes and regenerate correct producer records; do not silently bless changed input |
| `E_DUPLICATE_KEY` | Have upstream owners resolve/test their site/day observation policy |
| `E_FIT_POPULATION` | Regenerate models using the correct actual-target-date eligibility, not just filtered reports |
| `E_UPSTREAM_INVALID` | Complete genuine agreement/leakage/CV evidence or correct the upstream run |
| `E_PREDICTION_COVERAGE` / `E_TRUTH_MISMATCH` | Regenerate predictions from the same shared reference/population; no silent intersection |
| `E_PROBABILITY` | Correct class mapping, malformed probability output or label disagreement at the producer |
| `E_TEST_GATE` | Supply the matching frozen validation decision and explicit test flag |
| `E_OUTPUT_EXISTS` | Choose a new run ID; preserve the existing run |
| `E_MODEL_REPRODUCTION` / `E_MODEL_SCHEMA` | Check split-specific artifact, tree count, ordered features and class encoding |
| `E_SHAP_SHAPE` / `E_SHAP_ADDITIVITY` | Diagnose the model/library/shape mismatch; do not loosen tolerances to force plots |

## Verify the implementation

```bash
.venv-evaluation/bin/python -m unittest discover -s tests -p 'test_evaluate.py'
.venv-evaluation/bin/python -m unittest discover -s tests -p 'test_evaluation_reports.py'
.venv-evaluation/bin/python -m unittest discover -s tests -p 'test_explain.py'
.venv-evaluation/bin/python -m unittest discover -s tests
```

Use the environment prefixes above when required. Tests use synthetic data, including a real small multiclass LightGBM model for SHAP reproduction/additivity. They do not score the real test block. A skipped actual SHAP integration test is not successful verification. Review test output and the generated figures; record actual package/source versions and disclose remaining upstream blockers separately from software test results.

Implementation ownership is narrow: `src/evaluate.py` validates/publishes the run, `src/_evaluation_reports.py` formats already validated results, and `src/explain.py` verifies/explains the saved LightGBM artifact. Upstream preprocessing/training remains with the existing team owners.
