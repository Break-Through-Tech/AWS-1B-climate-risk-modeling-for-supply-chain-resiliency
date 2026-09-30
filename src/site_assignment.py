"""Assign a stable physical-site identifier (site_id) to every buoy observation.

The raw UCI El Nino file has 8,536 distinct latitude/longitude pairs because
moored buoys move inside a watch circle, are redeployed at slightly different
positions, and occasionally break free and drift. This script groups those
coordinates back into physical mooring sites.

Pipeline (see docs/site_id.md for the full rationale and results):
  1. Load the data, parse dates, and check coordinates for missing/invalid values.
  2. Recover a station-series proxy: the file is sorted by buoy then date, so
     every time the date goes backwards a new series starts (series_id).
  3. Flag drifting observations (moving > DRIFT_KM_PER_DAY on both sides).
  4. Cluster the remaining unique coordinates with DBSCAN (haversine distance,
     weighted by observation count), sweeping several eps thresholds.
  5. Merge clusters that are alternative positions of the same series
     (same series_id, within RELOCATION_MERGE_KM).
  6. Choose a nominal coordinate per site: snap to the TAO/TRITON whole-degree
     site grid when the medoid is close to it, otherwise keep the medoid.
  7. Attach site_id + nominal coordinates to each observation, flag uncertain
     rows, validate row counts, and write outputs.

Run from the repository root:
    python src/site_assignment.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "el_nino_features.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "site_id"

EARTH_RADIUS_KM = 6371.0088
EXPECTED_SITES = 219  # figure quoted in Challenge-Project-Overview.md; a benchmark only

# Parameters (justified in docs/site_id.md)
DRIFT_KM_PER_DAY = 10.0  # moored buoys move a median ~1 km/day
EPS_KM = 25.0  # DBSCAN neighbourhood radius
EPS_SWEEP_KM = [2, 5, 10, 15, 25, 40, 60, 80, 100, 150]
MIN_OBS = 30  # min observations (summed weight) for a DBSCAN core point
RELOCATION_MERGE_KM = 150.0  # max distance between two positions of one series
GRID_SNAP_DEG = 0.3  # medoid must be this close (lat and lon) to a grid point
TAO_LATITUDES = (-8, -5, -2, 0, 2, 5, 7, 8, 9)  # latitude lines of the TAO/TRITON array
FAR_FROM_NOMINAL_KM = 50.0  # observation-level review threshold
LOW_SUPPORT_OBS = 100  # site-level review threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; inputs in degrees, broadcastable."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def site_code(lat, lon):
    """PMEL-style site label, e.g. 0N110W, 2S156E, 1.75N157.33W."""
    lat_txt = f"{abs(lat):g}{'S' if lat < 0 else 'N'}"
    lon_e = ((lon + 180) % 360) - 180  # normalise to [-180, 180)
    if abs(lon_e) == 180:
        lon_txt = "180"
    else:
        lon_txt = f"{abs(lon_e):g}{'W' if lon_e < 0 else 'E'}"
    return lat_txt + lon_txt


# ---------------------------------------------------------------------------
# 1-3. Load, check, series proxy, drift flag
# ---------------------------------------------------------------------------
def load_observations(path=RAW_PATH):
    df = pd.read_csv(path)
    df.insert(0, "obs_id", np.arange(len(df)))  # row position in the raw file
    df["obs_date"] = pd.to_datetime(
        df["date"].astype(int).astype(str).str.zfill(6), format="%y%m%d"
    )
    # The file is ordered by buoy, then date: a backwards date step starts a new series.
    df["series_id"] = (df["obs_date"].diff().dt.days < 0).cumsum()
    return df


def coordinate_checks(df):
    lat, lon = df["latitude"], df["longitude"]
    return {
        "rows": len(df),
        "unique_coordinate_pairs": int(df[["latitude", "longitude"]].drop_duplicates().shape[0]),
        "unique_latitudes": int(lat.nunique()),
        "unique_longitudes": int(lon.nunique()),
        "missing_latitude": int(lat.isna().sum()),
        "missing_longitude": int(lon.isna().sum()),
        "latitude_out_of_range": int((~lat.between(-90, 90)).sum()),
        "longitude_out_of_range": int((~lon.between(-180, 180)).sum()),
        "null_island_0_0": int(((lat == 0) & (lon == 0)).sum()),
        "rows_at_lon_minus_180": int((lon == -180).sum()),
        "pairs_after_rounding_to_1_degree": int(
            df[["latitude", "longitude"]].round(0).drop_duplicates().shape[0]
        ),
        "series_from_row_order": int(df["series_id"].nunique()),
        "duplicate_series_date_rows": int(df.duplicated(["series_id", "obs_date"]).sum()),
    }


def flag_drifting(df, threshold=DRIFT_KM_PER_DAY):
    """True where the buoy moved faster than `threshold` km/day to both neighbours."""
    same_prev = df["series_id"].eq(df["series_id"].shift())
    dist = haversine_km(
        df["latitude"].shift(), df["longitude"].shift(), df["latitude"], df["longitude"]
    )
    days = df["obs_date"].diff().dt.days.clip(lower=1)
    speed_prev = pd.Series(np.where(same_prev, dist / days, np.nan), index=df.index)
    speed_next = speed_prev.shift(-1)
    return (speed_prev > threshold) & (speed_next > threshold)


# ---------------------------------------------------------------------------
# 4. DBSCAN on unique coordinates
# ---------------------------------------------------------------------------
def dbscan_labels(coords, eps_km=EPS_KM, min_obs=MIN_OBS):
    """coords: DataFrame of unique latitude/longitude with an `n` (obs count) column."""
    model = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM,
        min_samples=min_obs,
        metric="haversine",
        algorithm="ball_tree",
    )
    return model.fit(np.radians(coords[["latitude", "longitude"]].to_numpy()), sample_weight=coords["n"]).labels_


def cluster_observations(df, eps_km=EPS_KM, min_obs=MIN_OBS):
    """Return a cluster label per row (-1 = noise or drifting)."""
    usable = df[~df["is_drifting"]]
    coords = usable.groupby(["latitude", "longitude"]).size().rename("n").reset_index()
    coords["cluster"] = dbscan_labels(coords, eps_km, min_obs)
    labels = df[["latitude", "longitude"]].merge(coords, on=["latitude", "longitude"], how="left")["cluster"]
    labels = labels.fillna(-1).astype(int).to_numpy()
    labels[df["is_drifting"].to_numpy()] = -1
    return labels


def same_day_conflicts(df, labels):
    """Rows sharing a date inside one cluster: evidence that two buoys were merged."""
    tmp = pd.DataFrame({"c": labels, "d": df["obs_date"].to_numpy()})
    tmp = tmp[tmp["c"] >= 0]
    return int(tmp.duplicated(["c", "d"]).sum())


def split_series(df, labels, min_share=0.05):
    """Series whose assigned rows land in >1 cluster with at least `min_share` each."""
    tmp = pd.DataFrame({"s": df["series_id"].to_numpy(), "c": labels})
    tmp = tmp[tmp["c"] >= 0]
    shares = tmp.groupby("s")["c"].value_counts(normalize=True)
    return int((shares[shares >= min_share].groupby(level=0).size() > 1).sum())


def eps_sweep(df, eps_values=EPS_SWEEP_KM, min_obs_values=(10, MIN_OBS, 100)):
    rows = []
    for min_obs in min_obs_values:
        for eps in eps_values:
            labels = cluster_observations(df, eps, min_obs)
            rows.append(
                {
                    "eps_km": eps,
                    "min_obs": min_obs,
                    "clusters": int(labels.max() + 1),
                    "unassigned_obs": int((labels < 0).sum()),
                    "same_day_conflict_obs": same_day_conflicts(df, labels),
                    "series_split_across_clusters": split_series(df, labels),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Merge successive positions of one station series
# ---------------------------------------------------------------------------
def weighted_medoid(lat, lon, weight):
    """Observed coordinate with the smallest weighted total distance to the others."""
    lat, lon, weight = map(np.asarray, (lat, lon, weight))
    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    best = int(np.argmin(dist @ weight))
    return lat[best], lon[best]


def cluster_summary(df, labels):
    tmp = df.assign(cluster=labels)
    tmp = tmp[tmp["cluster"] >= 0]
    rows = []
    for c, g in tmp.groupby("cluster"):
        coords = g.groupby(["latitude", "longitude"]).size().reset_index(name="n")
        mlat, mlon = weighted_medoid(coords["latitude"], coords["longitude"], coords["n"])
        series = g["series_id"].value_counts()
        rows.append(
            {
                "cluster": c,
                "n_obs": len(g),
                "medoid_lat": mlat,
                "medoid_lon": mlon,
                "start": g["obs_date"].min(),
                "end": g["obs_date"].max(),
                "main_series": int(series.index[0]),
                "main_series_share": series.iloc[0] / len(g),
            }
        )
    return pd.DataFrame(rows).set_index("cluster")


def merge_relocations(clusters, max_km=RELOCATION_MERGE_KM):
    """Union clusters whose main series is the same and whose medoids lie close.

    A series never has two rows on the same date, so its positions are never
    simultaneous: they are redeployments or temporary excursions of one station.
    Returns (mapping cluster -> merged group, list of merge records).
    """
    parent = {c: c for c in clusters.index}

    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c

    merges = []
    for series, grp in clusters.groupby("main_series"):
        grp = grp.sort_values("start")
        ids = list(grp.index)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                A, B = clusters.loc[a], clusters.loc[b]
                dist = haversine_km(A["medoid_lat"], A["medoid_lon"], B["medoid_lat"], B["medoid_lon"])
                if dist <= max_km:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
                        merges.append({"series_id": series, "cluster_a": a, "cluster_b": b, "distance_km": round(float(dist), 1)})
    return {c: find(c) for c in clusters.index}, merges


# ---------------------------------------------------------------------------
# 6. Nominal coordinates
# ---------------------------------------------------------------------------
def nearest_grid_point(lat, lon):
    """Closest TAO/TRITON-style grid point: a TAO latitude line and a whole-degree longitude."""
    glat = min(TAO_LATITUDES, key=lambda t: abs(lat - t))
    glon = round(lon)
    if glon == -180:
        glon = 180
    return float(glat), float(glon)


def nominal_location(lat, lon, tol=GRID_SNAP_DEG):
    """Snap to the TAO/TRITON site grid when close, else keep the medoid."""
    glat, glon = nearest_grid_point(lat, lon)
    lon_diff = abs(((lon - glon) + 180) % 360 - 180)
    if abs(lat - glat) <= tol and lon_diff <= tol:
        return glat, glon, "tao_grid"
    return float(lat), float(lon), "estimated_medoid"


# ---------------------------------------------------------------------------
# 7. Assemble sites and attach to observations
# ---------------------------------------------------------------------------
def build_sites(df, labels):
    clusters = cluster_summary(df, labels)
    group_of, merges = merge_relocations(clusters)
    clusters["group"] = clusters.index.map(group_of)
    tmp = df.assign(group=pd.Series(labels).map(group_of).to_numpy())

    sites = []
    for grp, g in tmp[tmp["group"].notna()].groupby("group"):
        coords = g.groupby(["latitude", "longitude"]).size().reset_index(name="n")
        mlat, mlon = weighted_medoid(coords["latitude"], coords["longitude"], coords["n"])
        nlat, nlon, source = nominal_location(mlat, mlon)
        glat, glon = nearest_grid_point(mlat, mlon)
        member = clusters[clusters["group"] == grp]
        sites.append(
            {
                "group": grp,
                "nominal_lat": nlat,
                "nominal_lon": nlon,
                "nominal_source": source,
                "medoid_lat": mlat,
                "medoid_lon": mlon,
                "nearest_grid_code": site_code(glat, glon),
                "nearest_grid_km": round(float(haversine_km(mlat, mlon, glat, glon)), 1),
                "n_position_clusters": len(member),
                "series_ids": ";".join(str(s) for s in sorted(g["series_id"].unique())),
            }
        )
    sites = pd.DataFrame(sites)
    # Stable ordering: south -> north, then west -> east on a 0-360 longitude.
    sites["_lon360"] = sites["nominal_lon"] % 360
    sites = sites.sort_values(["nominal_lat", "_lon360"]).reset_index(drop=True)
    sites["site_id"] = [f"SITE_{i + 1:03d}" for i in range(len(sites))]
    sites["site_code"] = [site_code(a, o) for a, o in zip(sites["nominal_lat"], sites["nominal_lon"])]
    return sites.drop(columns="_lon360"), clusters, merges


def attach_sites(df, labels, sites, clusters):
    group = pd.Series(labels).map(clusters["group"]).to_numpy()
    lookup = sites.set_index("group")[["site_id", "site_code", "nominal_lat", "nominal_lon", "nominal_source"]]
    out = df.join(lookup, on=pd.Series(group, index=df.index).rename("group"))
    out["dist_to_nominal_km"] = haversine_km(out["latitude"], out["longitude"], out["nominal_lat"], out["nominal_lon"])

    reasons = pd.Series("", index=out.index)
    reasons[out["is_drifting"]] = "unassigned_drifting"
    reasons[out["site_id"].isna() & ~out["is_drifting"]] = "unassigned_noise"
    reasons[out["site_id"].notna() & (out["dist_to_nominal_km"] > FAR_FROM_NOMINAL_KM)] = "far_from_nominal"
    out["review_reason"] = reasons.replace("", np.nan)
    out["needs_review"] = out["review_reason"].notna()
    return out


def site_statistics(obs, sites):
    a = obs[obs["site_id"].notna()]
    stats = a.groupby("site_id").agg(
        n_obs=("obs_id", "size"),
        n_unique_coords=("latitude", lambda s: a.loc[s.index, ["latitude", "longitude"]].drop_duplicates().shape[0]),
        first_obs=("obs_date", "min"),
        last_obs=("obs_date", "max"),
        spread_median_km=("dist_to_nominal_km", "median"),
        spread_p95_km=("dist_to_nominal_km", lambda s: s.quantile(0.95)),
        spread_max_km=("dist_to_nominal_km", "max"),
        n_far_from_nominal=("needs_review", "sum"),
    )
    same_day = a.groupby("site_id")["obs_date"].apply(lambda s: int(s.duplicated().sum())).rename("same_day_conflicts")
    sites = sites.join(stats, on="site_id").join(same_day, on="site_id")

    notes = []
    for _, r in sites.iterrows():
        n = []
        if r["n_position_clusters"] > 1:
            n.append("merged_relocation")
        if len(r["series_ids"].split(";")) > 1:
            n.append("multiple_series")
        if r["n_obs"] < LOW_SUPPORT_OBS:
            n.append("low_support")
        if r["same_day_conflicts"] > 0:
            n.append("same_day_conflicts")
        notes.append(";".join(n))
    sites["review_notes"] = notes
    for col in ["spread_median_km", "spread_p95_km", "spread_max_km", "medoid_lat", "medoid_lon"]:
        sites[col] = sites[col].round(3)
    return sites.drop(columns="group")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
BLUE, ORANGE, RED, GRAY, INK = "#2a78d6", "#eb6834", "#e34948", "#9a9993", "#52514e"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_map(obs, sites, path):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    lon360 = lambda s: s % 360
    ok = obs[obs["site_id"].notna()].drop_duplicates(["latitude", "longitude"])
    drift = obs[obs["review_reason"] == "unassigned_drifting"]
    noise = obs[obs["review_reason"] == "unassigned_noise"]
    ax.scatter(lon360(ok["longitude"]), ok["latitude"], s=3, c=GRAY, lw=0, label=f"Observed coordinate, assigned ({len(ok):,} unique)")
    ax.scatter(lon360(drift["longitude"]), drift["latitude"], s=8, c=RED, lw=0, label=f"Unassigned: drifting ({len(drift):,} obs)")
    ax.scatter(lon360(noise["longitude"]), noise["latitude"], s=8, c=ORANGE, lw=0, label=f"Unassigned: isolated ({len(noise):,} obs)")
    for src, marker in (("tao_grid", "o"), ("estimated_medoid", "D")):
        s = sites[sites["nominal_source"] == src]
        ax.scatter(lon360(s["nominal_lon"]), s["nominal_lat"], s=40, marker=marker, facecolors="none", edgecolors=BLUE, linewidths=1.5,
                   label=f"Nominal site: {'TAO grid' if src == 'tao_grid' else 'estimated medoid'} ({len(s)})")
    ticks = np.arange(130, 280, 10)
    ax.set_xticks(ticks, [f"{t}°E" if t < 180 else ("180°" if t == 180 else f"{360 - t}°W") for t in ticks])
    ax.set_ylabel("Latitude (°)", color=INK)
    ax.set_title(f"TAO/TRITON buoy observations grouped into {len(sites)} physical sites", loc="left", color="#0b0b0b")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(sweep, chosen_eps, path):
    s = sweep[sweep["min_obs"] == MIN_OBS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [
        ("clusters", "DBSCAN clusters"),
        ("same_day_conflict_obs", "Same-day conflicts (merged buoys)"),
        ("unassigned_obs", "Unassigned observations"),
    ]
    for ax, (col, title) in zip(axes, panels):
        ax.plot(s["eps_km"], s[col], color=BLUE, lw=2, marker="o", ms=5)
        ax.axvline(chosen_eps, color=GRAY, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(s["eps_km"], [str(v) for v in s["eps_km"]])
        ax.set_xlabel("eps (km)", color=INK)
        ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
        _style(ax)
    axes[0].axhline(EXPECTED_SITES, color=ORANGE, lw=1)
    axes[0].text(s["eps_km"].iloc[0], EXPECTED_SITES, " expected 219", color=INK, va="bottom", fontsize=8)
    axes[1].set_yscale("symlog")
    axes[1].set_ylim(bottom=0)
    fig.suptitle(f"eps sensitivity (min_obs = {MIN_OBS}); dashed line = chosen eps {chosen_eps:g} km", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_site_zooms(obs, site_codes, path):
    fig, axes = plt.subplots(1, len(site_codes), figsize=(4 * len(site_codes), 3.8))
    for ax, code in zip(np.atleast_1d(axes), site_codes):
        s = obs[obs["site_code"] == code]
        nlat, nlon = s["nominal_lat"].iloc[0], s["nominal_lon"].iloc[0]
        box = obs[(obs["latitude"].sub(nlat).abs() < 1.5) & (((obs["longitude"] - nlon + 180) % 360 - 180).abs() < 1.5)]
        other = box[box["site_code"] != code]
        ax.scatter(other["longitude"], other["latitude"], s=6, c=np.where(other["site_id"].isna(), RED, GRAY), lw=0)
        ax.scatter(s["longitude"], s["latitude"], s=6, c=BLUE, lw=0)
        ax.scatter([nlon], [nlat], s=80, marker="+", c="#0b0b0b")
        ax.set_title(f"{code} ({s['site_id'].iloc[0]}): {len(s):,} obs", loc="left", fontsize=10, color="#0b0b0b")
        _style(ax)
    fig.suptitle("Blue = assigned to site, gray = other sites, red = unassigned, + = nominal", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_site_grid(obs, sites, path, ncols=10):
    """One panel per site: original coordinates coloured by assignment status."""
    coords = (
        obs.assign(status=np.select(
            [obs["site_id"].isna(), obs["review_reason"] == "far_from_nominal"],
            ["unassigned", "far"], "ok"))
        .groupby(["latitude", "longitude", "site_id", "status"], dropna=False).size()
        .rename("n").reset_index()
    )
    nrows = int(np.ceil(len(sites) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    for ax, (_, site) in zip(axes.flat, sites.iterrows()):
        own = coords[coords["site_id"] == site["site_id"]]
        nlon = site["nominal_lon"]
        rel = lambda lon: (lon - nlon + 180) % 360 - 180 + nlon  # unwrap across the dateline
        pad = 0.1
        lat0, lat1 = own["latitude"].min() - pad, own["latitude"].max() + pad
        lon0, lon1 = rel(own["longitude"]).min() - pad, rel(own["longitude"]).max() + pad
        box = coords[coords["latitude"].between(lat0, lat1) & rel(coords["longitude"]).between(lon0, lon1)]
        other = box[(box["site_id"] != site["site_id"]) & box["site_id"].notna()]
        unassigned = box[box["status"] == "unassigned"]
        size = lambda n: 4 + 3 * np.log10(n)  # marker area grows with observation count
        ax.scatter(rel(other["longitude"]), other["latitude"], s=size(other["n"]), c=GRAY, lw=0)
        for status, color in (("ok", BLUE), ("far", ORANGE)):
            p = own[own["status"] == status]
            ax.scatter(rel(p["longitude"]), p["latitude"], s=size(p["n"]), c=color, lw=0)
        ax.scatter(rel(unassigned["longitude"]), unassigned["latitude"], s=6, c=RED, lw=0)
        ax.scatter([nlon], [site["nominal_lat"]], s=60, marker="+", c="#0b0b0b", lw=1.2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{site['site_id']}  {site['site_code']}\n{site['n_obs']:,} obs · p95 {site['spread_p95_km']:.0f} km",
                     loc="left", fontsize=8, color="#0b0b0b")
        _style(ax)
        ax.tick_params(labelsize=6)
        ax.ticklabel_format(useOffset=False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    for ax in axes.flat[len(sites):]:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], ls="", marker="o", color=BLUE, label="Assigned, within 50 km of nominal"),
        plt.Line2D([], [], ls="", marker="o", color=ORANGE, label="Assigned, flagged far_from_nominal"),
        plt.Line2D([], [], ls="", marker="o", color=RED, label="Unassigned (drifting or isolated)"),
        plt.Line2D([], [], ls="", marker="o", color=GRAY, label="Belongs to another site"),
        plt.Line2D([], [], ls="", marker="+", color="#0b0b0b", label="Nominal location"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Original buoy coordinates by site (axes in degrees, each panel auto-scaled; marker size ∝ log obs count)",
                 x=0.01, y=1.012, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_observations()
    checks = coordinate_checks(df)
    df["is_drifting"] = flag_drifting(df)

    sweep = eps_sweep(df)
    sweep.to_csv(REPORT_DIR / "eps_sweep.csv", index=False)

    labels = cluster_observations(df, EPS_KM, MIN_OBS)
    sites, clusters, merges = build_sites(df, labels)
    obs = attach_sites(df, labels, sites, clusters)
    sites = site_statistics(obs, sites)

    # Row-count guarantee: the join must neither add nor drop observations.
    assert len(obs) == len(df) == checks["rows"], "site join changed the row count"
    assert obs["obs_id"].is_unique and obs["obs_id"].equals(df["obs_id"]), "obs_id order changed"

    site_cols = ["site_id", "site_code", "nominal_lat", "nominal_lon", "nominal_source", "medoid_lat", "medoid_lon",
                 "n_obs", "n_unique_coords", "first_obs", "last_obs", "spread_median_km", "spread_p95_km",
                 "spread_max_km", "n_far_from_nominal", "nearest_grid_code", "nearest_grid_km", "n_position_clusters", "series_ids", "same_day_conflicts", "review_notes"]
    sites[site_cols].to_csv(PROCESSED_DIR / "site_lookup.csv", index=False)

    obs_cols = ["obs_id", "series_id", "latitude", "longitude", "site_id", "site_code", "nominal_lat", "nominal_lon",
                "nominal_source", "dist_to_nominal_km", "is_drifting", "needs_review", "review_reason"]
    out = obs[obs_cols].copy()
    out["dist_to_nominal_km"] = out["dist_to_nominal_km"].round(3)
    out.to_csv(PROCESSED_DIR / "site_assignments.csv", index=False)
    pd.DataFrame(merges).to_csv(REPORT_DIR / "relocation_merges.csv", index=False)
    obs[obs["needs_review"]][["obs_id", "series_id", "obs_date", "latitude", "longitude", "site_id", "site_code",
                              "dist_to_nominal_km", "review_reason"]].to_csv(REPORT_DIR / "review_queue.csv", index=False)

    plot_map(obs, sites, REPORT_DIR / "site_map.png")
    plot_sweep(sweep, EPS_KM, REPORT_DIR / "eps_sweep.png")
    zoom = [c for c in ("0N170W", "0N156E", "0N110W", "2N137E") if c in set(sites["site_code"])]
    plot_site_zooms(obs, zoom, REPORT_DIR / "site_zooms.png")
    plot_site_grid(obs, sites, REPORT_DIR / "site_clusters_grid.png")

    # Console summary (numbers quoted in docs/site_id.md)
    print("Coordinate checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\nChosen parameters: eps={EPS_KM} km, min_obs={MIN_OBS}, drift>{DRIFT_KM_PER_DAY} km/day, relocation merge<={RELOCATION_MERGE_KM} km")
    print(f"DBSCAN position clusters: {len(clusters)}; relocation merges: {len(merges)}")
    print(f"Final sites: {len(sites)} (benchmark {EXPECTED_SITES}, difference {len(sites) - EXPECTED_SITES:+d})")
    print(f"  nominal from TAO grid: {(sites['nominal_source'] == 'tao_grid').sum()}, estimated medoid: {(sites['nominal_source'] == 'estimated_medoid').sum()}")
    print(f"Observations: {len(obs):,} total, {obs['site_id'].notna().sum():,} assigned, {obs['site_id'].isna().sum():,} unassigned")
    print(obs["review_reason"].value_counts(dropna=False).to_string())
    print(f"Sites with review notes: {(sites['review_notes'] != '').sum()}")
    print("\nSweep (min_obs = %d):" % MIN_OBS)
    print(sweep[sweep["min_obs"] == MIN_OBS].to_string(index=False))


if __name__ == "__main__":
    main()
