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
                "site_id": ["A"] * 6,
                "date": ["900101", "900102", "910101", "920101", "950101", "970101"],
                "month": [1] * 6,
                "split": ["train"] * 4 + ["validation", "test"],
                "ss_temp": [10.0, 10.0, 20.0, 100.0, 30.0, 40.0],
                "zon_winds": [2.0, 2.0, 6.0, 100.0, 10.0, 12.0],
                "mer_winds": [1.0, 1.0, 5.0, 100.0, 7.0, 9.0],
                "humidity": [70.0, 70.0, 90.0, 100.0, 95.0, 100.0],
                "air_temp": [25.0] * 6,
                "use_for_climatology": [True, True, True, False, True, True],
            }
        )
        monthly = add_monthly_anomalies(aggregate_monthly(observations), observations)
        observed = monthly.loc[monthly["year"].eq(1995)].iloc[0]
        self.assertEqual(observed["clim_sst"], 15.0)
        self.assertEqual(observed["monthly_sst"], 30.0)
        self.assertEqual(observed["ssta"], 15.0)
        for baseline, anomaly, expected_baseline, expected_anomaly in (
            ("clim_zon_winds", "zon_winds_anomaly", 4.0, 6.0),
            ("clim_mer_winds", "mer_winds_anomaly", 3.0, 4.0),
            ("clim_humidity", "humidity_anomaly", 80.0, 15.0),
        ):
            self.assertTrue(monthly[baseline].eq(expected_baseline).all())
            self.assertEqual(observed[anomaly], expected_anomaly)
        test = monthly.loc[monthly["year"].eq(1997)].iloc[0]
        self.assertEqual(test["zon_winds_anomaly"], 8.0)
        self.assertEqual(test["mer_winds_anomaly"], 6.0)
        self.assertEqual(test["humidity_anomaly"], 20.0)

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
