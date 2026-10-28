"""Check whether observations far from their site's nominal location change the SST signal.

Design (a before/after test around each transition):
  - For every assigned row, take the SST difference to the same-day mean SST of
    neighbouring sites (within NEIGHBOUR_KM). Seasonal cycles and ENSO swings are
    shared across neighbours, so this difference changes slowly when the buoy
    stays put.
  - Every time a series switches between unflagged rows (within FLAG_KM of the
    nominal location) and flagged rows (beyond it), compare the mean difference in
    the WINDOW_DAYS before the switch with the WINDOW_DAYS after it.
  - Baseline: the same before/after comparison at every day inside long runs of
    unflagged rows. If flagged rows measured a different water mass, the shifts
    at real transitions would be larger than the baseline shifts.

Run from the repository root, after src/site_assignment.py:
    python src/site_quality_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "el_nino_features.csv"
ASSIGN_PATH = ROOT / "data" / "processed" / "site_assignments.csv"
LOOKUP_PATH = ROOT / "data" / "processed" / "site_lookup.csv"
OUT_PATH = ROOT / "reports" / "site_id" / "flagged_sst_check.csv"

EARTH_RADIUS_KM = 6371.0088
FLAG_KM = 50.0  # same threshold as far_from_nominal in site_assignment.py
NEIGHBOUR_KM = 1000.0
MIN_NEIGHBOURS = 2
WINDOW_DAYS = 30
MIN_WINDOW_OBS = 10


def neighbour_matrix(sites):
    lat = np.radians(sites["nominal_lat"].to_numpy())
    lon = np.radians(sites["nominal_lon"].to_numpy())
    a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    dist = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    return ((dist > 0) & (dist <= NEIGHBOUR_KM)).astype(float)


def neighbour_differences():
    """Assigned rows with SST minus the same-day neighbour mean SST (`diff`)."""
    raw = pd.read_csv(RAW_PATH, usecols=["date", "ss_temp"])
    obs = pd.read_csv(ASSIGN_PATH)
    sites = pd.read_csv(LOOKUP_PATH)
    obs = obs.assign(sst=raw["ss_temp"], day=pd.to_datetime(raw["date"].astype(str).str.zfill(6), format="%y%m%d"))
    obs = obs[obs["site_id"].notna() & obs["sst"].notna()].copy()
    obs["flagged"] = obs["dist_to_nominal_km"] > FLAG_KM

    grid = obs.pivot_table(index="day", columns="site_id", values="sst").reindex(columns=sites["site_id"])
    w = neighbour_matrix(sites)
    total = np.nan_to_num(grid.to_numpy()) @ w
    count = grid.notna().to_numpy().astype(float) @ w
    ref = pd.DataFrame(np.where(count >= MIN_NEIGHBOURS, total / np.maximum(count, 1), np.nan),
                       index=grid.index, columns=grid.columns)
    ref_long = ref.stack().rename("ref").reset_index()
    obs = obs.merge(ref_long, on=["day", "site_id"], how="left")
    obs["diff"] = obs["sst"] - obs["ref"]
    return obs[obs["diff"].notna()].sort_values(["series_id", "day"]).reset_index(drop=True)


def window_shift(g, t):
    """Mean diff in the WINDOW_DAYS after day t minus the WINDOW_DAYS before it."""
    before = g.loc[(g["day"] < t) & (g["day"] >= t - pd.Timedelta(days=WINDOW_DAYS)), "diff"]
    after = g.loc[(g["day"] >= t) & (g["day"] < t + pd.Timedelta(days=WINDOW_DAYS)), "diff"]
    if len(before) < MIN_WINDOW_OBS or len(after) < MIN_WINDOW_OBS:
        return np.nan
    return after.mean() - before.mean()


def transition_shifts(obs):
    rows = []
    for (series, site), g in obs.groupby(["series_id", "site_code"]):
        g = g.reset_index(drop=True)
        switch = g.index[g["flagged"].ne(g["flagged"].shift()) & (g.index > 0)]
        for i in switch:
            t = g.at[i, "day"]
            # Orient every shift as flagged minus unflagged.
            sign = 1 if g.at[i, "flagged"] else -1
            shift = window_shift(g, t)
            if not np.isnan(shift):
                rows.append({"kind": "transition", "site_code": site, "day": t, "shift_c": sign * shift})
        # Baseline: shifts inside unflagged stretches, sampled every WINDOW_DAYS.
        clean = g[~g["flagged"]]
        for t in clean["day"].iloc[::WINDOW_DAYS]:
            window = g[(g["day"] >= t - pd.Timedelta(days=WINDOW_DAYS)) & (g["day"] < t + pd.Timedelta(days=WINDOW_DAYS))]
            if window["flagged"].any():
                continue
            shift = window_shift(g, t)
            if not np.isnan(shift):
                rows.append({"kind": "baseline", "site_code": site, "day": t, "shift_c": shift})
    return pd.DataFrame(rows)


def main():
    shifts = transition_shifts(neighbour_differences())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shifts.round(3).to_csv(OUT_PATH, index=False)
    summary = shifts.groupby("kind")["shift_c"].agg(
        n="size", mean="mean", median="median",
        mean_abs=lambda s: s.abs().mean(), p90_abs=lambda s: s.abs().quantile(0.9),
    ).round(3)
    print(summary.to_string())
    base = shifts.loc[shifts["kind"] == "baseline", "shift_c"].abs()
    trans = shifts[shifts["kind"] == "transition"].copy()
    trans["percentile_vs_baseline"] = trans["shift_c"].abs().apply(lambda v: (base < v).mean() * 100).round(0)
    print("\nTransitions (flagged minus unflagged, deg C) and how unusual each is against the baseline:")
    print(trans.sort_values("day")[["site_code", "day", "shift_c", "percentile_vs_baseline"]].round(3).to_string(index=False))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
