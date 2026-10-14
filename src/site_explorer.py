"""Build an interactive HTML viewer for the buoy site clusters.

Reads the outputs of src/site_assignment.py and writes a single self-contained
page, reports/site_id/site_explorer.html, that can be opened in any browser.

Run from the repository root, after src/site_assignment.py:
    python src/site_explorer.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "el_nino_features.csv"
ASSIGN_PATH = ROOT / "data" / "processed" / "site_assignments.csv"
LOOKUP_PATH = ROOT / "data" / "processed" / "site_lookup.csv"
TEMPLATE_PATH = Path(__file__).resolve().parent / "site_explorer_template.html"
OUT_PATH = ROOT / "reports" / "site_id" / "site_explorer.html"

EPOCH = pd.Timestamp("1980-01-01")
STATUS = {"ok": 0, "far_from_nominal": 1, "unassigned_drifting": 2, "unassigned_noise": 3}


def build_payload():
    raw = pd.read_csv(RAW_PATH, usecols=["date"])
    obs = pd.read_csv(ASSIGN_PATH)
    sites = pd.read_csv(LOOKUP_PATH)
    assert len(raw) == len(obs), "run src/site_assignment.py first"

    obs["day"] = (pd.to_datetime(raw["date"].astype(str).str.zfill(6), format="%y%m%d") - EPOCH).dt.days
    obs["status"] = obs["review_reason"].fillna("ok").map(STATUS)
    site_index = {sid: i for i, sid in enumerate(sites["site_id"])}
    obs["site"] = obs["site_id"].map(site_index).fillna(-1).astype(int)

    # One point per unique coordinate, site and status (the page draws coordinates, not rows).
    pts = (
        obs.groupby(["latitude", "longitude", "site", "status"])
        .agg(n=("obs_id", "size"), first=("day", "min"), last=("day", "max"), dist=("dist_to_nominal_km", "mean"))
        .reset_index()
    )
    points = [
        [round(r.latitude, 2), round(r.longitude, 2), int(r.site), int(r.status), int(r.n), int(r.first), int(r.last),
         None if np.isnan(r.dist) else round(float(r.dist), 1)]
        for r in pts.itertuples()
    ]

    site_rows = []
    for r in sites.itertuples():
        site_rows.append({
            "id": r.site_id, "code": r.site_code, "lat": r.nominal_lat, "lon": r.nominal_lon,
            "source": r.nominal_source, "mlat": r.medoid_lat, "mlon": r.medoid_lon,
            "n": int(r.n_obs), "coords": int(r.n_unique_coords), "first": r.first_obs, "last": r.last_obs,
            "med": r.spread_median_km, "p95": r.spread_p95_km, "max": r.spread_max_km, "far": int(r.n_far_from_nominal),
            "grid": r.nearest_grid_code, "gridKm": r.nearest_grid_km, "clusters": int(r.n_position_clusters),
            "series": str(r.series_ids), "notes": "" if pd.isna(r.review_notes) else r.review_notes,
        })

    summary = {
        "rows": int(len(obs)),
        "pairs": int(obs[["latitude", "longitude"]].drop_duplicates().shape[0]),
        "sites": int(len(sites)),
        "assigned": int(obs["site_id"].notna().sum()),
        "far": int((obs["status"] == 1).sum()),
        "drifting": int((obs["status"] == 2).sum()),
        "noise": int((obs["status"] == 3).sum()),
    }
    return {"summary": summary, "sites": site_rows, "points": points}


def main():
    payload = json.dumps(build_payload(), separators=(",", ":"))
    html = TEMPLATE_PATH.read_text().replace("/*__DATA__*/null", payload)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
