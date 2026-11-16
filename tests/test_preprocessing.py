import unittest

import pandas as pd

from notebooks.preprocessing import (
    add_future_monthly_targets,
    add_monthly_anomalies,
    aggregate_monthly,
)


class MonthlyPreprocessingTest(unittest.TestCase):
    def test_climatology_weights_training_years_equally(self):
        observations = pd.DataFrame(
            {
                "site_id": ["A"] * 4,
                "date": ["900101", "900102", "910101", "950101"],
                "month": [1] * 4,
                "split": ["train", "train", "train", "validation"],
                "ss_temp": [10.0, 10.0, 20.0, 30.0],
                "zon_winds": [1.0] * 4,
                "mer_winds": [2.0] * 4,
                "humidity": [80.0] * 4,
                "air_temp": [25.0] * 4,
                "use_for_climatology": [True] * 4,
            }
        )
        monthly = add_monthly_anomalies(aggregate_monthly(observations), observations)
        observed = monthly.loc[monthly["year"].eq(1995)].iloc[0]
        self.assertEqual(observed["clim_sst"], 15.0)
        self.assertEqual(observed["monthly_sst"], 30.0)
        self.assertEqual(observed["ssta"], 15.0)

    def test_future_target_requires_exact_month_and_stays_within_split(self):
        monthly = pd.DataFrame(
            {
                "site_id": ["A"] * 4,
                "month_start": pd.to_datetime(
                    ["1994-09-01", "1994-10-01", "1994-12-01", "1995-01-01"]
                ),
                "split": ["train", "train", "train", "validation"],
                "ssta": [0.1, 0.2, 1.0, 2.0],
            }
        )
        result = add_future_monthly_targets(monthly).set_index("month_start")
        self.assertEqual(result.loc["1994-09-01", "target_ssta"], 1.0)
        self.assertTrue(pd.isna(result.loc["1994-10-01", "target_ssta"]))
        self.assertTrue(pd.isna(result.loc["1994-12-01", "target_ssta"]))


if __name__ == "__main__":
    unittest.main()
