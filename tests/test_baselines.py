import unittest

import numpy as np
import pandas as pd

from src import baselines
from src.interim_dataset import add_history, add_target, risk_class


def synthetic_table(step_days=5):
    """Two sites, 1990-1998, SSTA following a slow cycle so all classes appear."""
    dates = pd.date_range("1990-01-01", "1998-06-23", freq=f"{step_days}D")
    rows = []
    for k, site in enumerate(["SITE_001", "SITE_002"]):
        t = np.arange(len(dates))
        ssta = 1.2 * np.sin(2 * np.pi * t * step_days / 900 + k)
        rows.append(pd.DataFrame({"site_id": site, "date": dates, "ssta": ssta, "zon_winds": -ssta * 2 + k}))
    df = pd.concat(rows, ignore_index=True)
    df["split"] = np.select([df["date"] <= "1994-12-31", df["date"] <= "1996-12-31"], ["train", "validation"], "test")
    df["current_class"] = risk_class(df["ssta"])
    df["target_date"] = df["date"] + pd.Timedelta(days=90)
    future = df.set_index(["site_id", "date"])["current_class"]
    df["target_class"] = [future.get((s, d), pd.NA) for s, d in zip(df["site_id"], df["target_date"])]
    return df


class UsableRowsTest(unittest.TestCase):
    def test_drops_missing_labels_and_targets_past_split_end(self):
        df = pd.DataFrame({
            "split": ["train", "train", "train", "validation", "test"],
            "date": pd.to_datetime(["1994-09-01", "1994-11-01", "1994-08-01", "1996-12-01", "1998-03-01"]),
            "target_class": ["neutral", "neutral", pd.NA, "warming", "warming"],
        })
        df["target_date"] = df["date"] + pd.Timedelta(days=90)
        kept = baselines.usable_rows(df)
        # 1994-11-01 + 90 days is in 1995 (validation); 1996-12-01 + 90 is in test.
        self.assertEqual(kept.index.tolist(), [0, 4])


class FeatureColumnsTest(unittest.TestCase):
    def test_excludes_ids_labels_and_text(self):
        df = synthetic_table()
        df["note"] = "x"
        self.assertEqual(baselines.feature_columns(df), ["ssta", "zon_winds"])


class PersistenceTest(unittest.TestCase):
    def test_copies_current_class_and_fills_missing_with_majority(self):
        train = pd.DataFrame({"target_class": ["neutral", "neutral", "warming"]})
        rows = pd.DataFrame({"current_class": pd.array(["cooling", pd.NA, "warming"], dtype="string")})
        self.assertEqual(baselines.predict_persistence(train, rows).tolist(), ["cooling", "neutral", "warming"])


class ScoreTest(unittest.TestCase):
    def test_perfect_and_constant_predictions(self):
        y = ["cooling", "neutral", "warming", "neutral"]
        self.assertEqual(baselines.score(y, y)["macro_f1"], 1.0)
        constant = baselines.score(y, ["neutral"] * 4)
        self.assertAlmostEqual(constant["macro_f1"], (0 + 2 / 3 + 0) / 3)
        self.assertEqual(constant["confusion_matrix"]["rows_true_cols_pred"], [[0, 1, 0], [0, 2, 0], [0, 1, 0]])


class RunTest(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_table()

    def test_test_block_untouched_by_default(self):
        preds, scores, settings, _ = baselines.run(self.df)
        self.assertEqual(set(preds["split"]), {"train", "validation"})
        self.assertNotIn("test", scores["logistic"])
        self.assertFalse(settings["test_scored"])
        self.assertEqual(set(preds["model"]), {"majority", "persistence", "logistic"})

    def test_use_test_scores_every_model_with_valid_probabilities(self):
        preds, scores, _, model = baselines.run(self.df, use_test=True)
        self.assertEqual(set(scores["persistence"]), {"train", "validation", "test"})
        proba = preds[[f"proba_{c}" for c in baselines.CLASSES]].to_numpy()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        self.assertEqual(list(model.classes_), baselines.CLASSES)
        # The tuned model beats the majority floor on this learnable signal.
        self.assertGreater(scores["logistic"]["validation"]["macro_f1"], scores["majority"]["validation"]["macro_f1"])

    def test_one_prediction_per_model_and_row(self):
        preds, _, _, _ = baselines.run(self.df)
        self.assertFalse(preds.duplicated(["model", "site_id", "date"]).any())


class InterimDatasetTest(unittest.TestCase):
    def test_risk_class_thresholds(self):
        ssta = pd.Series([0.51, 0.5, -0.5, -0.51, np.nan])
        self.assertEqual(risk_class(ssta).tolist(), ["warming", "neutral", "neutral", "cooling", pd.NA])

    def test_target_uses_first_reading_within_tolerance_only(self):
        df = pd.DataFrame({
            "site_id": ["A"] * 4,
            "date": pd.to_datetime(["1990-01-01", "1990-01-02", "1990-04-03", "1990-07-20"]),
            "current_class": pd.array(["neutral", "neutral", "warming", "cooling"], dtype="string"),
        })
        out = add_target(df, horizon_days=90, tolerance_days=7)
        # Jan 1 + 90 = Apr 1 -> Apr 3 reading; Jan 2 + 90 = Apr 2 -> Apr 3; Apr 3 + 90 = Jul 2 -> none within 7 days.
        self.assertEqual(out["target_class"].tolist()[:3], ["warming", "warming", pd.NA])

    def test_history_features_never_look_forward(self):
        dates = pd.date_range("1990-01-01", periods=40, freq="D")
        df = pd.DataFrame({"site_id": "A", "date": dates, "month": dates.month, "ssta": np.arange(40.0)})
        for col in ["zon_winds", "mer_winds", "air_temp", "humidity"]:
            df[col] = 0.0
        out = add_history(df)
        self.assertEqual(out.loc[10, "ssta_lag7"], 3.0)
        self.assertTrue(np.isnan(out.loc[0, "ssta_lag1"]))
        self.assertEqual(out.loc[2, "ssta_mean_7d"], 1.0)  # mean of days 0-2, not days 3+


if __name__ == "__main__":
    unittest.main()
