# Baselines (Member 4)

The baselines set the scores LightGBM has to beat. All of them use the same rows, the same splits and the same Macro F1, so the comparison is fair.

| Model | What it predicts | Fit on |
| --- | --- | --- |
| `majority` | The most common training class, every time. This is the floor. | train |
| `persistence` | That the class on the forecast date stays the same three months later. | nothing |
| `logistic` | Multinomial logistic regression on the history features: median imputation plus missing-value flags, standard scaling, `class_weight="balanced"`. `C` is chosen by validation Macro F1. | train |

## How to run

From the repository root, after `python src/site_assignment.py`:

```bash
python -m src.interim_dataset      # stand-in table until Members 1-3 deliver
python -m src.baselines            # scores train + validation only
python -m unittest tests/test_baselines.py
```

To use the team's final table, run `python -m src.baselines --data path/to/table.csv`.

The 1997–98 test block stays untouched until the final model has been chosen. Then run `python -m src.baselines --use-test`. This refits logistic regression on train plus validation with the chosen `C` and scores the test block once.

## Input contract

The input is one row per site and forecast date. LightGBM (Member 5) should read the same table.

| Column | Meaning |
| --- | --- |
| `site_id` | `SITE_001`–`SITE_080`, from `data/processed/site_assignments.csv` |
| `date` | Forecast date (ISO `YYYY-MM-DD`) |
| `split` | `train`, `validation` or `test`, using the boundaries in `src/split.py` |
| `target_date` | The date being forecast (`date` + horizon) |
| `target_class` | `cooling`, `neutral` or `warming` at `target_date`. Leave it blank if there is no reading. |
| `current_class` | The same classification on `date`. Persistence uses it; it is never a feature. |
| any other numeric column | A feature. It may use only data up to `date`. |

`site_id`, `date`, `split`, `target_date`, `observed_target_date`, `target_class`, `current_class` and `obs_id` are never used as features. Text columns are ignored.

**Leakage rule.** Before training or scoring, rows are dropped if they have no label or if their `target_date` falls after the end of their own split. Without this, a training row dated 1994-11-01 would forecast a day in 1995, which is validation data.

## Outputs

All outputs are regenerable and ignored by git.

- `reports/baselines/predictions.csv`: one row per model, site and date. Columns: `model, site_id, date, target_date, split, y_true, y_pred, proba_cooling, proba_neutral, proba_warming`. Persistence and majority have no real probabilities, so their probability columns are 1 for the predicted class and 0 otherwise. Member 6 can read this file directly.
- `reports/baselines/scores.json`: Macro F1, accuracy, per-class precision, recall and F1, and the confusion matrix, for each model and split.
- `reports/baselines/leaderboard.csv`: one row per model and split.
- `reports/baselines/settings.json`: the feature list, the rows used per split, the validation score for each `C`, the chosen `C`, and the model settings.
- `models/logistic_baseline.joblib`: the fitted scikit-learn pipeline.

## Interim results (stand-in table, not final)

These come from `src/interim_dataset.py`:

- 3 classes at ±0.5 °C;
- a site × calendar-month climatology fit on training rows only;
- a 90-day horizon, using the first reading within 7 days after the target date;
- backward-looking lags, plus 7-, 30- and 90-day means and trends.

| Model | Train Macro F1 | Validation Macro F1 |
| --- | --- | --- |
| persistence | 0.519 | **0.517** |
| logistic (C = 1e-5) | 0.530 | 0.479 |
| majority | 0.252 | 0.195 |

Rows used: 76,566 train, 32,324 validation, and 22,938 test (not scored).

What this says:

- **Persistence is the bar to beat** (0.517). Ocean temperature anomalies change slowly, so "same as today" is a strong three-month forecast. On validation, persistence got recall 0.56 for cooling, 0.60 for neutral and 0.40 for warming.
- **Logistic regression loses on validation** even though it does slightly better on training. It is weakest on warming (F1 0.21 vs 0.38 for persistence).
- **The strongest regularization tested scored best.** More regularization kept helping, and C = 1e-5 is the edge of the grid. With SSTA history alone, logistic regression scores 0.50. This suggests the raw weather columns teach era- and site-specific patterns that don't carry forward. For example, humidity is missing before 1989, and air temperature levels differ by site. Members 3 and 5 should test anomaly or within-site versions of these features.
- **Validation (1995–96) has far more cooling than training** (La Niña conditions). Some of the drop from training to validation is distribution shift, not overfitting.

## Open decisions that change these numbers

1. **Number of classes.** The target script on `main` (`notebooks/preprocessing.py`) still produces nine classes, but the agreed plan is three. `CLASSES` in `src/baselines.py` assumes three.
2. **Horizon.** The stand-in uses 90 days. If the team chooses three calendar months, rebuild the table; the baselines don't need changes.
3. **Missing readings near the target date.** The stand-in uses the first reading within 7 days and otherwise leaves the target blank. On `main`, `notebooks/preprocessing.py` uses the next reading with no limit, so a target can come from a reading months or years later. It also lets training targets cross into validation.
