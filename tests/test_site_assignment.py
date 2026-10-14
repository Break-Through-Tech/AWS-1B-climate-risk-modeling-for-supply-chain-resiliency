import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.site_assignment import (
    flag_drifting,
    haversine_km,
    match_documented,
    nominal_location,
    site_code,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "el_nino_features.csv"
ASSIGN_PATH = ROOT / "data" / "processed" / "site_assignments.csv"
LOOKUP_PATH = ROOT / "data" / "processed" / "site_lookup.csv"


class HelperTest(unittest.TestCase):
    def test_haversine_one_degree_at_equator_and_dateline(self):
        self.assertAlmostEqual(haversine_km(0, 0, 0, 1), 111.195, places=2)
        self.assertAlmostEqual(haversine_km(0, 179.5, 0, -179.5), 111.195, places=2)

    def test_site_code_formats_hemispheres_and_dateline(self):
        self.assertEqual(site_code(0.0, -110.0), "0N110W")
        self.assertEqual(site_code(-2.0, 156.0), "2S156E")
        self.assertEqual(site_code(5.0, -180.0), "5N180")

    def test_drifting_needs_fast_movement_on_both_sides(self):
        days = pd.date_range("1990-01-01", periods=5)
        # A single redeployment jump (row 2) is not drift; continuous fast motion (rows 3-4) is.
        frame = pd.DataFrame({
            "series_id": 0,
            "obs_date": days,
            "latitude": [0.0, 0.0, 0.5, 1.0, 1.5],
            "longitude": [140.0, 140.0, 140.0, 140.0, 140.0],
        })
        self.assertEqual(flag_drifting(frame).tolist(), [False, False, True, True, False])

    def test_documented_match_requires_mutual_nearest_within_limit(self):
        reference = pd.DataFrame({"nominal_lat": [0.0, 2.0], "nominal_lon": [-110.0, -110.0]})
        # Site 0 sits near 0N110W; site 1 is nearer to 0N110W than to 2N110W but site 0 wins 0N110W.
        matches, _, _ = match_documented([0.01, 0.6], [-110.0, -110.0], reference)
        self.assertEqual(matches, [0, None])

    def test_grid_convention_only_when_close(self):
        self.assertEqual(nominal_location(0.02, 143.02)[2], "grid_convention")
        self.assertEqual(nominal_location(-0.30, -96.09)[2], "estimated_medoid")


@unittest.skipUnless(ASSIGN_PATH.exists() and LOOKUP_PATH.exists(), "run src/site_assignment.py first")
class OutputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(RAW_PATH)
        cls.obs = pd.read_csv(ASSIGN_PATH)
        cls.sites = pd.read_csv(LOOKUP_PATH)

    def test_join_keeps_every_row_in_order(self):
        self.assertEqual(len(self.obs), len(self.raw))
        np.testing.assert_array_equal(self.obs["obs_id"], np.arange(len(self.raw)))
        np.testing.assert_allclose(self.obs["latitude"], self.raw["latitude"])
        np.testing.assert_allclose(self.obs["longitude"], self.raw["longitude"])

    def test_every_row_is_assigned_or_explained(self):
        unassigned = self.obs["site_id"].isna()
        reasons = self.obs.loc[unassigned, "review_reason"]
        self.assertTrue(reasons.isin(["unassigned_drifting", "unassigned_noise"]).all())
        self.assertTrue(self.obs.loc[self.obs["is_drifting"], "site_id"].isna().all())

    def test_site_ids_are_sequential_and_consistent(self):
        expected = [f"SITE_{i:03d}" for i in range(1, len(self.sites) + 1)]
        self.assertEqual(self.sites["site_id"].tolist(), expected)
        self.assertEqual(set(self.obs["site_id"].dropna()), set(expected))

    def test_no_site_has_two_rows_on_one_date(self):
        dated = self.obs.assign(date=self.raw["date"]).dropna(subset=["site_id"])
        self.assertEqual(int(dated.duplicated(["site_id", "date"]).sum()), 0)

    def test_nominal_sources_are_known(self):
        allowed = {"documented", "grid_convention", "estimated_medoid"}
        self.assertTrue(set(self.sites["nominal_source"]) <= allowed)


if __name__ == "__main__":
    unittest.main()
