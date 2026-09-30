# Buoy Site Identification (`site_id`)

**Script:** [`src/site_assignment.py`](../src/site_assignment.py)
**Run:** `python src/site_assignment.py` from the repository root (takes about 10 seconds)

## Summary

The UCI El Niño file has **8,536 distinct latitude/longitude pairs**, but it does not have 8,536 buoys. The script groups those coordinates into **79 physical mooring sites** and gives every observation a `site_id`.

- **177,007 of 178,080 observations (99.4%)** are assigned to a site.
- **1,073 observations are unassigned.** 523 of them come from buoys drifting after breaking free of their moorings. The other 550 are isolated points.
- **5,448 assigned observations are flagged for review** because they lie more than 50 km from their site's nominal position.
- **The join is lossless.** The output has exactly 178,080 rows in the original order, and the script asserts this every time it runs.
- **The "roughly 219 locations" figure in the project overview does not hold up.** It matches what you get by rounding coordinates to the nearest degree, which gives 217 pairs. It does not match the number of physical sites. The TAO/TRITON array has about 70 sites, and this dataset covers 79 because it includes some older and short-lived moorings. See [Why not 219?](#why-not-219).

## Output files

| File | Contents |
| --- | --- |
| `data/processed/site_assignments.csv` | One row per observation (178,080 rows). Contains `obs_id`, `site_id`, `site_code`, nominal coordinates, distance to the nominal position, drift flag and review flag. |
| `data/processed/site_lookup.csv` | One row per site (79 rows). Contains nominal position and where it came from, observation count, date range, geographic spread, series IDs and review notes. |
| `reports/site_id/eps_sweep.csv` / `.png` | How the clustering results change with different distance thresholds (see [Choosing the threshold](#choosing-the-threshold)). |
| `reports/site_id/relocation_merges.csv` | The 8 cluster pairs that were merged because they are the same station at two positions. |
| `reports/site_id/review_queue.csv` | Every observation flagged for review, with its reason. |
| `reports/site_id/site_map.png` | Map of all coordinates, unassigned points and nominal sites. |
| `reports/site_id/site_explorer.html` | Interactive viewer you can open in a browser. It shows the array map and a searchable, filterable list of sites. Each site has a zoomable plot of its original coordinates, and hovering over a point shows its dates and observation count. Rebuild it with `python src/site_explorer.py` after running `src/site_assignment.py`. |
| `reports/site_id/site_clusters_grid.png` | One panel per site showing the original coordinates. Blue means assigned, orange means flagged as far from nominal, red means unassigned, gray means the point belongs to another site, and + marks the nominal position. |
| `reports/site_id/site_zooms.png` | Close-up views of 4 sites that were hard to resolve. |

### How to use it

```python
import pandas as pd

obs = pd.read_csv("data/el_nino_features.csv")
sites = pd.read_csv("data/processed/site_assignments.csv")

# obs_id is the row position in el_nino_features.csv
obs = obs.reset_index(names="obs_id").merge(
    sites.drop(columns=["latitude", "longitude"]), on="obs_id", how="left", validate="1:1"
)
assert len(obs) == 178_080

# For climatology or anomaly baselines, group by site_id.
# The conservative option is to also exclude flagged rows:
clean = obs[obs["site_id"].notna() & ~obs["needs_review"]]
```

The original `latitude` and `longitude` are always kept. `nominal_lat` and `nominal_lon` are added alongside them and do not replace them.

### Column reference (`site_assignments.csv`)

| Column | Meaning |
| --- | --- |
| `obs_id` | Row position (starting from 0) in `data/el_nino_features.csv`. Use it as the join key. |
| `series_id` | Contiguous time series recovered from the row order of the file (see step 2). |
| `latitude`, `longitude` | Original coordinates as recorded. |
| `site_id` | `SITE_001` to `SITE_079`. Empty if the observation is unassigned. |
| `site_code` | Human-readable label in the style PMEL uses, e.g. `0N110W`, `2S156E`. Estimated sites show their medoid position, e.g. `1.75N157.33W`. |
| `nominal_lat`, `nominal_lon` | The site's representative position. |
| `nominal_source` | `tao_grid` means the position was snapped to the TAO/TRITON grid. `estimated_medoid` means it is the observed point closest to the rest of the site's observations. |
| `dist_to_nominal_km` | Great-circle distance from the observation to its site's nominal position. |
| `is_drifting` | The buoy was moving more than 10 km/day, i.e. it was loose from its mooring. |
| `needs_review`, `review_reason` | Set to `unassigned_drifting`, `unassigned_noise` or `far_from_nominal` (more than 50 km from nominal). |

## Why there are 8,536 coordinate pairs

**The coordinates are clean.** No values are missing, out of range, or placed at (0, 0). 72 rows record longitude as −180, which is the same meridian as +180. The haversine distance handles this, and site labels show it as `180`.

**The dataset has no station ID column,** but the file is sorted by buoy and then by date. Whenever the date goes backwards, a new buoy record begins. This splits the file into **76 contiguous series** with no duplicate dates inside any series. The script uses these series as stand-in station metadata, both to validate the clusters and to decide which clusters to merge.

With those series in hand, the coordinate variation comes from four distinct sources:

1. **Watch-circle motion and redeployment offsets.** A moored buoy swings around its anchor, and each new deployment lands a few kilometres from the last one. Positions are recorded to 0.01° (about 1 km), so each site builds up 15 to 400 distinct pairs. The median day-to-day movement is 1.1 km. This is the main source of variation.
2. **Buoys drifting after breaking free.** Some buoys break their moorings and drift at 50 to 500 km/day, leaving tracks across the map (the red lines in `site_map.png`). There are 523 such observations.
3. **Relocations and temporary excursions of the same station.** One series can spend months at a second position 40 to 130 km away and then come back. For example, series 14 sits at 0N95W from 1981 to 1995, moves to 0.3S96.1W from 1995 to 1996, and returns to 0N95W from 1996 to 1998.
4. **Slow excursions.** A buoy dragging its anchor moves under 10 km/day for weeks. For example, the 2S110W buoy moved up to 190 km during 1990. These points stay dense enough to join the site's cluster, so they are kept but flagged.

In addition, **some series are two different stations stitched together.** Series 13, for instance, covers 0.2N176.5W from 1986 to 1989 and then 0N180 from 1993 to 1998, 170 km apart. Because of this, series are treated as evidence about stations rather than as station IDs.

## Method

### 1. Flag drifting observations
An observation counts as drifting if the buoy moved **more than 10 km/day both from the previous day and to the next day** within the same series. A moored buoy cannot keep up that speed, since its median movement is 1.1 km/day and its 99th percentile is 5.7 km/day. Requiring fast movement on both sides means a single jump at redeployment is not mistaken for drift. Drifting observations are left out of clustering and left unassigned.

### 2. DBSCAN clustering on unique coordinates
The clustering runs on the 8,536 unique pairs, not all 178,080 rows. Each pair is weighted by how many observations it has.

- The distance is **haversine** (great-circle), using a ball tree.
- `eps` = **25 km** is the radius used to decide which points count as neighbours.
- `min_samples` = **30 observations** is the total weight needed for a point to seed a cluster. A single stray position cannot start a site. A position occupied for about a month can.

Points that end up in no cluster are labelled `unassigned_noise`. They are mostly the start and end points of drift tracks.

### 3. Merge alternative positions of the same station
Two clusters are merged into one site if they **share the same dominant series and their medoids are no more than 150 km apart**. A series never has two rows on the same date, so its positions cannot be two buoys operating at once. They are redeployments or temporary excursions of a single station. The 150 km cap keeps apart the stitched-together series described above, the closest of which are 170 km apart. **8 merges** took the 87 clusters down to **79 sites**. All 8 are listed in `relocation_merges.csv`.

### 4. Nominal location
No mooring-coordinate metadata file comes with the UCI data. The nominal position is therefore set in one of two ways:

- **`tao_grid` (68 sites).** TAO/TRITON sites follow a documented convention: fixed latitude lines (8S, 5S, 2S, 0, 2N, 5N, 7N, 8N, 9N) at whole-degree longitudes (e.g. 95W, 110W, 125W, 140W, 155W, 170W, 180, 165E, 156E, 147E, 137E). If a site's medoid is within 0.3° of such a point in both latitude and longitude, the site is snapped to it.
- **`estimated_medoid` (11 sites).** Otherwise the nominal position is the **weighted medoid**: the observed coordinate with the smallest total distance to all of the site's observations.

`site_lookup.csv` also includes `nearest_grid_code` and `nearest_grid_km` for every site. These make it easy to check the estimated sites against a PMEL site list.

### 5. Stable IDs
Sites are sorted south to north, then west to east using 0–360° longitude, and numbered `SITE_001` onward. The IDs are deterministic: running the script again on the same data with the same parameters gives the same IDs. **Treat `site_lookup.csv` as the reference list.** Changing a parameter can renumber sites, so if you change one, check the lookup table for differences before downstream code relies on the new IDs.

## Choosing the threshold

`eps` was tested at 2, 5, 10, 15, 25, 40, 60, 80, 100 and 150 km. `min_samples` was tested at 10, 30 and 100 observations.

| eps (km) | clusters | unassigned obs | same-day conflicts | series split across clusters |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 207 | 3,213 | 0 | 45 |
| 5 | 142 | 2,024 | 0 | 27 |
| 10 | 114 | 1,638 | 0 | 16 |
| 15 | 98 | 1,368 | 0 | 14 |
| **25** | **87** | **1,073** | **0** | **8** |
| 40 | 84 | 873 | 1 | 6 |
| 60 | 81 | 734 | 4 | 4 |
| 80 | 78 | 641 | 2,568 | 3 |
| 100 | 75 | 640 | 3,283 | 3 |
| 150 | 71 | 552 | 16,072 | 3 |

Two checks detect clustering mistakes:

- **Same-day conflicts** catch sites that were wrongly merged. A single site cannot have two observations on the same date, so any such pair means two separate buoys were grouped together. Conflicts appear from 40 km upward and rise sharply from 80 km, when neighbouring sites start merging.
- **Series split across clusters** catches sites that were wrongly split. It counts series whose observations are spread across more than one cluster, with at least 5% in each. Small `eps` values break single stations into many pieces.

25 km is the largest threshold with **no** same-day conflicts. The splits that remain at 25 km are either real relocations, which step 3 merges, or genuinely separate stations. The choice of `min_samples` has a much smaller effect: at 25 km, 10, 30 and 100 give 100, 87 and 86 clusters respectively.

Removing drifting observations first matters. Without that step, the drift track from the 0N154E buoy crosses 0N156E and creates same-day conflicts at every threshold of 15 km or more.

## Validation

| Check | Result |
| --- | --- |
| Rows before and after the join | 178,080 → 178,080 (the script asserts this) |
| `obs_id` unique and in original order | Yes (the script asserts this) |
| Assigned observations | 177,007 (99.40%) |
| Unassigned, drifting | 523 |
| Unassigned, isolated points | 550 |
| Assigned but more than 50 km from nominal | 5,448, spread across 15 sites |
| Same-day conflicts within any site | 0 |
| Sites with a single dominant series | 78 of 79 |
| Site spread: median distance to nominal | 4.6 km median across sites (range 0–31 km) |
| Site spread: 95th-percentile distance to nominal | 17 km median across sites (range 0–126 km) |

Every site's spread is listed per site in `site_lookup.csv` (`spread_median_km`, `spread_p95_km`, `spread_max_km`).

### Why not 219?

The final count is 79, 140 short of the benchmark. The evidence says 79 is correct and 219 is an artefact of binning:

- **The file itself contains only 76 buoy series.** Clustering gives 79 sites because 3 series cover two distinct stations each (the stitched-together series described above).
- **Rounding coordinates to 1° gives 217 pairs,** which is very close to 219. Rounding splits sites that sit near a half-degree boundary and turns every drift-track position into its own "location".
- **Getting to about 219 clusters requires `eps` of about 2 km.** At that setting, 45 of the 76 series are broken into several pieces, which means one mooring is being counted many times.
- **The real TAO/TRITON array has about 70 sites.** 68 of the sites found here snap to its grid. The other 11 are early or short-lived moorings from the 1980s and the TOGA-COARE period, such as 0.5S166.92E, 1.75N157.33W, 0.02S160.55E and 0.01S157.51E.

The 219 figure should be revised in `Challenge-Project-Overview.md`.

## Unresolved cases

| Site | Issue | Suggested action |
| --- | --- | --- |
| `SITE_040` 0N110W, `SITE_041` 0N95W, `SITE_037` 0N170W, `SITE_045` 2N137E, `SITE_033` 0N156E, `SITE_066` 5N95W, `SITE_069` 7N147W, `SITE_070` 7N132W | Each was merged from two positions 38–130 km apart. The observations at the secondary position are flagged `far_from_nominal` whenever they are more than 50 km away. | Check against PMEL deployment history. The 0N95W merge covers the largest distance (130 km), so check it first. |
| `SITE_025` 1.94S164.46E | Since 1992 the buoy has sat at about 164.4E rather than 165E. The medoid falls between the two positions, 52 km from 2S165E, so the site was not snapped to the grid. 1,443 observations are more than 50 km from nominal. | This is probably PMEL's 2S165E site. If a metadata source confirms that, set the nominal position to 2S165E. |
| `SITE_027` 0.18S124.41W | The buoy consistently sat about 0.6° east of 0N125W. | This is probably 0N125W. Confirm it against metadata. |
| `SITE_023` 2S110W, `SITE_052` 2N140W, `SITE_054` 2N110W, `SITE_009` 5S165E, `SITE_035` 0N170E | Slow excursions of up to 190 km pulled points into the site's cluster. | These are flagged `far_from_nominal`. Exclude them from climatology calculations, or review them individually. |
| `SITE_004` 8S155W, `SITE_079` 9N140W | The medoid is 28–31 km from the grid point it was snapped to, because the whole record sits consistently offset from it. | These are acceptable as they are. The spread columns show the offset. |
| `SITE_029` 0.01S157.51E | 2 of its 460 observations come from series 7 (the 0N156E buoy passing through). | Low impact. The site's review note is `multiple_series`. |
| 3 stitched series (13, 58, 60) | Each covers two stations more than 150 km apart and is correctly split into two sites. | None. They are documented here for traceability. |

## Parameters

All parameters are constants at the top of `src/site_assignment.py`:

| Constant | Value | Why |
| --- | --- | --- |
| `DRIFT_KM_PER_DAY` | 10 | About 9 times the median daily movement of a moored buoy, and above its 99.5th percentile. |
| `EPS_KM` | 25 | The largest `eps` with no same-day conflicts (see the sweep above). |
| `MIN_OBS` | 30 | Roughly one month of daily observations, enough to seed a site. |
| `RELOCATION_MERGE_KM` | 150 | Covers every observed relocation (38–130 km) and stays below the closest pair of stitched stations (170 km). |
| `GRID_SNAP_DEG` | 0.3 | About 33 km. Allows for consistent deployment offsets without snapping sites that clearly sit off the grid. |
| `FAR_FROM_NOMINAL_KM` | 50 | Flags an observation for review at this distance, twice `eps`. |
