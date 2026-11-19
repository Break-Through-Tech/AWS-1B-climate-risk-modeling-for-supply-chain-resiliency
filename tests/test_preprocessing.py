import math
import unittest

import pandas as pd

from notebooks.preprocessing import (
    add_future_monthly_targets,
    add_monthly_anomalies,
    aggregate_monthly,
    create_target,
)


class MonthlyPreprocessingTest(unittest.TestCase):
    def test_three_class_thresholds(self):
        self.assertEqual(create_target(0.5), "el_nino")
        self.assertEqual(create_target(0.0), "neutral")
        self.assertEqual(create_target(-0.5), "la_nina")
        self.assertTrue(pd.isna(create_target(float("nan"))))

    def test_month_encoding_wraps_december_to_january(self):
        observations = pd.DataFrame(
            {
                "site_id": ["A"] * 4,
                "date": ["900101", "900201", "900701", "901201"],
                "split": ["train"] * 4,
                "ss_temp": [25.0] * 4,
                "zon_winds": [1.0] * 4,
                "mer_winds": [1.0] * 4,
                "humidity": [80.0] * 4,
                "air_temp": [25.0] * 4,
            }
        )
        monthly = aggregate_monthly(observations).set_index("month")
        self.assertAlmostEqual(monthly.loc[1, "month_sin"], 0.0)
        self.assertAlmostEqual(monthly.loc[1, "month_cos"], 1.0)
        self.assertAlmostEqual(monthly.loc[7, "month_cos"], -1.0)
        jan_to_dec = math.dist(monthly.loc[1, ["month_sin", "month_cos"]],
                               monthly.loc[12, ["month_sin", "month_cos"]])
        jan_to_feb = math.dist(monthly.loc[1, ["month_sin", "month_cos"]],
                               monthly.loc[2, ["month_sin", "month_cos"]])
        self.assertAlmostEqual(jan_to_dec, jan_to_feb)

    def test_noaa_sst_and_training_only_weather_baselines(self):
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
        noaa = pd.DataFrame({"site_id": ["A"], "month": [1], "clim_sst": [18.0]})
        monthly = add_monthly_anomalies(
            aggregate_monthly(observations), observations, noaa
        )
        observed = monthly.loc[monthly["year"].eq(1995)].iloc[0]
        self.assertEqual(observed["clim_sst"], 18.0)
        self.assertEqual(observed["monthly_sst"], 30.0)
        self.assertEqual(observed["ssta"], 12.0)
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
