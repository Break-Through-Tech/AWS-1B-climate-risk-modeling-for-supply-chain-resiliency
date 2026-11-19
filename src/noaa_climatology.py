"""Extract buoy-specific monthly SST normals from NOAA ERSSTv5 NetCDF."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
CLIMATOLOGY_PATH = ROOT / "data/reference/sst.mon.ltm.1981-2010.nc"
SITE_LOOKUP_PATH = ROOT / "data/processed/site_lookup.csv"
BASELINE_PATH = ROOT / "data/processed/noaa_buoy_baseline.csv"


def build_buoy_baseline(climatology_path=CLIMATOLOGY_PATH, site_path=SITE_LOOKUP_PATH):
    """Return one NOAA 1981-2010 mean SST for each site and calendar month."""
    sites = pd.read_csv(site_path)[["site_id", "nominal_lat", "nominal_lon"]]
    if sites["site_id"].duplicated().any():
        raise ValueError("site_lookup.csv has duplicate site_id values")

    with xr.open_dataset(climatology_path, decode_times=False) as dataset:
        climatology = dataset["sst"].sortby("lat")
        if climatology.sizes["time"] != 12:
            raise ValueError("NOAA climatology must contain exactly 12 months")
        climatology = climatology.assign_coords(
            month=("time", np.arange(1, 13))
        ).swap_dims(time="month")
        points = climatology.interp(
            lat=xr.DataArray(sites["nominal_lat"].to_numpy(), dims="site"),
            lon=xr.DataArray(sites["nominal_lon"].to_numpy() % 360, dims="site"),
        )
        values = points.transpose("site", "month").to_numpy().ravel()

    baseline = pd.DataFrame({
        "site_id": np.repeat(sites["site_id"].to_numpy(), 12),
        "month": np.tile(np.arange(1, 13), len(sites)),
        "clim_sst": values,
    })
    if baseline["clim_sst"].isna().any():
        missing = baseline.loc[baseline["clim_sst"].isna(), "site_id"].unique()
        raise ValueError(f"NOAA climatology is missing SST for sites: {missing.tolist()}")
    return baseline


if __name__ == "__main__":
    result = build_buoy_baseline()
    result.to_csv(BASELINE_PATH, index=False)
    print(f"Wrote {len(result):,} site/month averages to {BASELINE_PATH}")
