"""Baseline models for the three-month-ahead risk class (Member 4).

Models, all scored with the same rows and splits so LightGBM can be compared:
- majority:    always predicts the most common training class (floor).
- persistence: predicts that the class at the forecast date stays the same.
- logistic:    multinomial logistic regression on the history features,
               median imputation + missing flags + scaling, fit on train,
               C chosen by validation Macro F1.

The 1997-98 test block is left untouched unless --use-test is passed; then the
logistic model is refit on train + validation with the chosen C before scoring
test once.

Input contract (docs/baselines.md): one row per site and forecast date with
site_id, date, split, target_date, target_class, current_class, and numeric
feature columns. Until Members 1-3 deliver, build a stand-in with
    python -m src.interim_dataset

Run from the repository root:
    python -m src.baselines [--data PATH] [--use-test]
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.split import TEST_END, TRAIN_END, VALIDATION_END

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "interim" / "model_table.csv"
OUT_DIR = ROOT / "reports" / "baselines"
MODEL_DIR = ROOT / "models"

CLASSES = ["cooling", "neutral", "warming"]
SPLIT_END = {"train": TRAIN_END, "validation": VALIDATION_END, "test": TEST_END}
# Never used as features: identifiers, the label, and anything about the target date.
RESERVED = {"site_id", "date", "split", "target_date", "observed_target_date", "target_class", "current_class", "obs_id"}
C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
SEED = 0


def load_table(path):
    df = pd.read_csv(path, parse_dates=["date", "target_date"])
    missing = {"site_id", "date", "split", "target_date", "target_class", "current_class"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def usable_rows(df):
    """Rows with a label whose target date stays inside their own split.

    A training row dated 1994-11-01 forecasts 1995-01-30, which is validation
    data; keeping it would let validation information into training.
    """
    end = pd.to_datetime(df["split"].map({k: str(v) for k, v in SPLIT_END.items()}))
    keep = df["target_class"].notna() & (df["target_date"] <= end)
    return df[keep].copy()


def feature_columns(df):
    return [c for c in df.columns if c not in RESERVED and pd.api.types.is_numeric_dtype(df[c])]


def predict_majority(train, rows):
    label = train["target_class"].mode()[0]
    return pd.Series(label, index=rows.index)


def predict_persistence(train, rows):
    # No class today (missing SST) -> fall back to the most common training class.
    return rows["current_class"].fillna(train["target_class"].mode()[0])


def make_logistic(C):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED),
    )


def tune_logistic(train, validation, features, grid=C_GRID):
    """Pick C by validation Macro F1; returns (best_C, {C: score})."""
    scores = {}
    for C in grid:
        model = make_logistic(C).fit(train[features], train["target_class"])
        scores[C] = macro_f1(validation["target_class"], model.predict(validation[features]))
    return max(scores, key=scores.get), scores


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0))


def score(y_true, y_pred):
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "n": int(len(y_true)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(n[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": {"labels": CLASSES, "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()},
    }


def prediction_frame(rows, model_name, y_pred, proba=None):
    out = rows[["site_id", "date", "target_date", "split", "target_class"]].rename(columns={"target_class": "y_true"})
    out.insert(0, "model", model_name)
    out["y_pred"] = np.asarray(y_pred)
    for i, c in enumerate(CLASSES):
        out[f"proba_{c}"] = proba[:, i] if proba is not None else (out["y_pred"] == c).astype(float)
    return out


def run(df, use_test=False):
    df = usable_rows(df)
    parts = {s: df[df["split"] == s] for s in ("train", "validation", "test")}
    train, validation = parts["train"], parts["validation"]
    eval_splits = ["train", "validation"] + (["test"] if use_test else [])
    features = feature_columns(df)

    best_C, c_scores = tune_logistic(train, validation, features)
    logistic = make_logistic(best_C).fit(train[features], train["target_class"])
    # Final test model sees train + validation, never test.
    final_logistic = make_logistic(best_C).fit(pd.concat([train, validation])[features],
                                              pd.concat([train, validation])["target_class"]) if use_test else logistic

    preds, scores = [], {}
    for split in eval_splits:
        rows = parts[split]
        model = final_logistic if split == "test" else logistic
        assert list(model.classes_) == CLASSES
        fit_rows = pd.concat([train, validation]) if split == "test" else train
        runs = {
            "majority": (predict_majority(fit_rows, rows), None),
            "persistence": (predict_persistence(fit_rows, rows), None),
            "logistic": (model.predict(rows[features]), model.predict_proba(rows[features])),
        }
        for name, (y_pred, proba) in runs.items():
            preds.append(prediction_frame(rows, name, y_pred, proba))
            scores.setdefault(name, {})[split] = score(rows["target_class"], y_pred)

    settings = {
        "classes": CLASSES,
        "features": features,
        "rows_used": {s: int(len(parts[s])) for s in parts},
        "test_scored": use_test,
        "logistic": {"C_grid_validation_macro_f1": {str(k): v for k, v in c_scores.items()}, "best_C": best_C,
                     "class_weight": "balanced", "imputer": "median + missing indicators", "scaler": "standard",
                     "fit_on": "train" + (" (test model: train + validation)" if use_test else ""), "random_state": SEED},
        "persistence": "target_class = current_class; missing current_class -> majority training class",
        "majority": "most frequent training target_class",
    }
    return pd.concat(preds, ignore_index=True), scores, settings, final_logistic


def leaderboard(scores):
    rows = [{"model": m, "split": s, "macro_f1": v["macro_f1"], "accuracy": v["accuracy"], "n": v["n"]}
            for m, by_split in scores.items() for s, v in by_split.items()]
    return pd.DataFrame(rows).sort_values(["split", "macro_f1"], ascending=[True, False], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--use-test", action="store_true", help="also score the 1997-98 test block (final run only)")
    args = parser.parse_args()

    preds, scores, settings, model = run(load_table(args.data), use_test=args.use_test)
    settings["data"] = str(args.data.resolve().relative_to(ROOT)) if args.data.resolve().is_relative_to(ROOT) else str(args.data)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    preds.to_csv(OUT_DIR / "predictions.csv", index=False, date_format="%Y-%m-%d")
    (OUT_DIR / "scores.json").write_text(json.dumps(scores, indent=2))
    (OUT_DIR / "settings.json").write_text(json.dumps(settings, indent=2))
    board = leaderboard(scores)
    board.to_csv(OUT_DIR / "leaderboard.csv", index=False)
    joblib.dump(model, MODEL_DIR / "logistic_baseline.joblib")

    print(board.to_string(index=False, float_format="%.3f"))
    print(f"\nlogistic best C = {settings['logistic']['best_C']}; outputs in {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
