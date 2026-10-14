# Buoy Site Identification (`site_id`)

**Scripts:** [`src/site_assignment.py`](../src/site_assignment.py) (assigns sites), [`src/site_quality_check.py`](../src/site_quality_check.py) (SST check on flagged rows), [`src/site_explorer.py`](../src/site_explorer.py) (interactive viewer)
**Run:** from the repository root, in this order. Each takes under 15 seconds.

```bash
python src/site_assignment.py
python src/site_quality_check.py
python src/site_explorer.py
python -m unittest tests/test_site_assignment.py
```

## Summary

The UCI El Niño file has **8,536 distinct latitude/longitude pairs**, but it does not have 8,536 buoys. The script groups those coordinates into **80 physical mooring sites** and gives every observation a `site_id`.

- **177,007 of 178,080 observations (99.4%)** are assigned to a site.
- **1,073 observations are unassigned.** 523 of them come from buoys drifting after breaking free of their moorings. The other 550 are isolated points.
- **65 of the 80 sites match the official TAO/TRITON site list**, so their nominal coordinates are documented rather than estimated. The other 15 are historical moorings that are no longer in the official list.
- **9,946 assigned observations are flagged** because they lie more than 50 km from their site's nominal position. **They should be kept.** An SST check found no change in the signal when a buoy moved off its position (see [Flagged rows](#flagged-rows-keep-them)).
- **The join is lossless.** The output has exactly 178,080 rows in the original order. The script asserts this, and so do the tests.
- **The "roughly 219 locations" figure in the project overview was wrong.** It matches what you get by rounding coordinates to the nearest degree (217 pairs), not the number of physical sites. See [Why not 219?](#why-not-219).

> **Changed on 2026-10-14:** the site count went from 79 to 80. The 0N95W station's secondary position is now its own site, because it measured colder water (see [step 3](#3-merge-alternative-positions-of-the-same-station)). As a result, **IDs from `SITE_026` onward shifted by one.** Nominal coordinates now come from the official site list, and `use_for_climatology` was added. Re-join from `site_assignments.csv` rather than reusing old IDs.

## Output files

| File | Contents |
| --- | --- |
| `data/processed/site_assignments.csv` | One row per observation (178,080 rows). Contains `obs_id`, `site_id`, `site_code`, nominal coordinates, distance to the nominal position, drift flag, review flag and `use_for_climatology`. |
| `data/processed/site_lookup.csv` | One row per site (80 rows). Contains nominal position and its source, the nearest documented site, observation count, date range, geographic spread, series IDs and review notes. |
| `data/reference/tao_triton_sites.csv` | The official TAO and TRITON site list used for nominal coordinates (76 Pacific sites), with source URLs and the retrieval date. |
| `reports/site_id/flagged_sst_check.csv` | Results of the SST check: one row per before/after window, either around a real on/off-position transition or from the baseline. |
| `reports/site_id/eps_sweep.csv` / `.png` | How the clustering results change with different distance thresholds. |
| `reports/site_id/relocation_merges.csv` | The 7 cluster pairs merged because they are the same station at two positions. |
| `reports/site_id/review_queue.csv` | Every observation flagged for review, with its reason. |
| `reports/site_id/site_explorer.html` | Interactive viewer. Open it in a web browser: double-click it in Finder, or drag it into Chrome. VS Code and GitHub show only the code, because the page is drawn by JavaScript. |
| `reports/site_id/site_map.png`, `site_clusters_grid.png`, `site_zooms.png` | Static figures: an array map, one panel per site, and 4 close-up views. |

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

# For climatology or anomaly baselines, group by site_id and keep recommended rows.
usable = obs[obs["use_for_climatology"]]
```

**Joining onto the train/validation/test split files.** `src/split.py` re-sorts the rows and drops the row number, so `obs_id` won't line up with them. Join on `date`, `latitude` and `longitude` instead. That combination is unique in the raw data, so the join is 1:1.

```python
key = (pd.read_csv("data/el_nino_features.csv")[["date", "latitude", "longitude"]]
       .join(pd.read_csv("data/processed/site_assignments.csv")[["site_id", "use_for_climatology"]]))
validation = validation.merge(key, on=["date", "latitude", "longitude"], how="left", validate="1:1")
```

The original `latitude` and `longitude` are always kept. `nominal_lat` and `nominal_lon` are added alongside them.

### Column reference (`site_assignments.csv`)

| Column | Meaning |
| --- | --- |
| `obs_id` | Row position (starting from 0) in `data/el_nino_features.csv`. Use it as the join key. |
| `series_id` | Contiguous time series recovered from the row order of the file (see below). |
| `latitude`, `longitude` | Original coordinates as recorded. |
| `site_id` | `SITE_001` to `SITE_080`. Empty if the observation is unassigned. |
| `site_code` | Site label in the official style, e.g. `0N110W` or `2S156E`. Sites at an estimated position show it with decimals, e.g. `1.75N157.33W`. |
| `nominal_lat`, `nominal_lon` | The site's representative position. |
| `nominal_source` | `documented` means the position comes from the official TAO/TRITON list. `grid_convention` means it was snapped to the whole-degree naming grid (historical sites). `estimated_medoid` means it is the observed point closest to all of the site's observations. |
| `dist_to_nominal_km` | Great-circle distance from the observation to its site's nominal position. |
| `is_drifting` | The buoy was moving more than 10 km/day, i.e. it was loose from its mooring. |
| `needs_review`, `review_reason` | Set to `unassigned_drifting`, `unassigned_noise` or `far_from_nominal` (more than 50 km from nominal). |
| `use_for_climatology` | Recommended rows for per-site SST work: every row with a site. |

## Why there are 8,536 coordinate pairs

**The coordinates are clean.** No values are missing, out of range, or placed at (0, 0). 72 rows record longitude as −180, which is the same meridian as +180. The haversine distance handles this, and site labels show it as `180`.

**The dataset has no station ID column,** but the file is sorted by buoy and then by date. Whenever the date goes backwards, a new buoy record begins. This splits the file into **76 contiguous series**, with no duplicate dates inside any series. The script uses these series as stand-in station metadata, both to validate the clusters and to decide which clusters to merge.

With those series in hand, the coordinate variation comes from four sources:

1. **Watch-circle motion and redeployment offsets.** A moored buoy swings around its anchor, and each new deployment lands a few kilometres from the last one. Positions are recorded to 0.01° (about 1 km), so each site builds up 15 to 400 distinct pairs. The median day-to-day movement is 1.1 km. This is the main source of variation.
2. **Buoys drifting after breaking free.** These move at 50 to 500 km/day and leave tracks across the map. There are 523 such observations.
3. **Relocations of the same station.** One series can spend months at a second position 40 to 130 km away and then return.
4. **Slow excursions.** A buoy dragging its anchor moves under 10 km/day for weeks. For example, the 2S110W buoy moved up to 190 km during 1990.

In addition, **some series are two different stations stitched together.** Series 13, for instance, covers 0.2N176.5W from 1986 to 1989 and then 0N180 from 1993 to 1998, 392 km apart. So series are treated as evidence about stations, not as station IDs.

## Method

### 1. Flag drifting observations
An observation counts as drifting if the buoy moved **more than 10 km/day both from the previous day and to the next day** within the same series. A moored buoy's median movement is 1.1 km/day and its 99th percentile is 5.7 km/day, so it can't sustain that speed. Requiring fast movement on both sides means a single jump at redeployment isn't mistaken for drift. Drifting observations are left out of clustering and left unassigned.

### 2. DBSCAN clustering on unique coordinates
The clustering runs on the 8,536 unique pairs, each weighted by its number of observations. It uses **haversine** distance with **`eps` = 25 km** and **`min_samples` = 30 observations**, roughly a month at one position. Points that end up in no cluster are labelled `unassigned_noise`. They are mostly the start and end points of drift tracks.

### 3. Merge alternative positions of the same station
Two clusters are merged if they **share the same dominant series and their medoids are no more than 100 km apart**. A series never has two rows on the same date, so its positions can't be two buoys operating at once.

**7 merges** take the 87 clusters down to **80 sites**. The merged positions are 38–85 km apart.

The cap was 150 km until 2026-10-14. Lowering it to 100 km separates the 0N95W station from its 1995–98 secondary position at 0.3S96.1W, 130 km away. That position sits near the equatorial cold tongue and recorded colder water:
- **At the two transitions,** its SST relative to neighbouring buoys dropped by 0.90 °C (larger than 94% of normal shifts) and by 0.45 °C (larger than 76%).
- **Over each position's whole record,** the secondary position averaged about 1.2 °C colder relative to its neighbours.

Merging it would have mixed two water masses into one site's climatology.

### 4. Nominal location
The nominal position is chosen in this order:

1. **`documented` (65 sites).** The official TAO and TRITON site lists are in [`data/reference/tao_triton_sites.csv`](../data/reference/tao_triton_sites.csv). They come from NOAA NDBC's OceanSITES catalogs, retrieved 2026-10-14:
   - TAO (60 sites): <https://dods.ndbc.noaa.gov/thredds/catalog/oceansites/DATA/catalog.html>
   - TRITON (16 Pacific sites): <https://dods.ndbc.noaa.gov/thredds/catalog/oceansites/long_timeseries/TRITON/catalog.html>

   A site takes the documented position when the two are **each other's nearest match and no more than 75 km apart**.
2. **`grid_convention` (5 sites).** These are historical sites with no official entry. If the medoid is within 0.15° of a whole-degree point, the site is snapped to it, following the naming convention (e.g. 0N143E, 7N140W).
3. **`estimated_medoid` (10 sites).** Otherwise the nominal position is the **weighted medoid**: the observed coordinate with the smallest total distance to all of the site's observations.

The documented matches show a clear break in distance:
- **63 sites** are within 32 km of an official site.
- **2 sites are at 60–69 km.** Each is the only candidate in both directions. 2S165E's buoy sat at about 164.4E from 1992 onward, and 0N125W's sat about 0.6° east and 0.2° south of the official position.
- **Every other site** is at least 108 km from any official site.

`site_lookup.csv` lists `nearest_documented_site` and `nearest_documented_km` for every site.

These sources give official site names and positions only. **The 1980s–90s per-deployment positions are not published there.** The OceanSITES files start in 2012, and TRITON's in 1998. So the 15 historical sites can't be checked against an official record.

### 5. Stable IDs
Sites are sorted south to north, then west to east using 0–360° longitude, and numbered `SITE_001` onward. The IDs are deterministic: the same data and parameters always give the same IDs. **Treat `site_lookup.csv` as the reference list.** Changing a parameter can renumber sites, as the 2026-10-14 change did.

## Choosing the threshold

`eps` was tested at 2, 5, 10, 15, 25, 40, 60, 80, 100 and 150 km. `min_samples` was tested at 10, 30 and 100.

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
- **Same-day conflicts** catch wrong merges: one site can't record two observations on the same date.
- **Series split across clusters** catches wrong splits. It counts series spread across more than one cluster, with at least 5% in each.

25 km is the largest threshold with no same-day conflicts. Removing drifting observations first matters: without that step, the drift track from the 0N154E buoy crosses 0N156E and creates conflicts at 15 km and above.

## Flagged rows: keep them

9,946 assigned rows are more than 50 km from their site's nominal position. Most come from two sites whose buoys consistently sat offset from the official position: 0N125W (4,121 rows) and 2S165E (2,099 rows). The rest come from merged relocations and slow excursions.

The question is whether those rows record different SST than the site's on-position rows. If they did, they would contaminate a per-site climatology. `src/site_quality_check.py` tests this:

1. **Remove shared signal.** For every row, take the site's SST minus the same-day mean SST of neighbouring buoys within 1,000 km. This cancels the seasonal cycle and El Niño swings, which neighbours share.
2. **Measure transitions.** At every switch between on-position and off-position rows within a series, compare that difference in the 30 days before the switch with the 30 days after.
3. **Compare with a baseline.** Run the same before/after comparison at regular points inside stretches where the buoy stayed on position.

| | windows | mean shift | mean \|shift\| | 90th percentile \|shift\| |
| --- | ---: | ---: | ---: | ---: |
| Baseline: buoy stayed on position | 4,435 | −0.00 °C | 0.33 °C | 0.71 °C |
| Real transitions on or off position | 33 | +0.08 °C | 0.26 °C | 0.63 °C |

Moving off position doesn't change the SST signal beyond normal day-to-day variation. No transition exceeds the 89th percentile of the baseline. Excluding these rows would remove 5.6% of assigned data, and over half of the 0N125W and 2S165E records, with no measured benefit. So `use_for_climatology` is true for every assigned row, and `needs_review` stays as information.

Two caveats:
- The test covers 33 transitions at 9 sites. Sites whose buoys were off position for their whole record, like 0N125W, have no transition to test. Their offset is a fixed 60–70 km, which is within the range the merged relocations show is safe.
- An earlier version compared anomalies against per-site climatologies. Those were built from different years at different sites, which skewed the results in both directions, so the before/after design replaced it.

## Validation

`tests/test_site_assignment.py` checks the helpers on small synthetic inputs and checks the outputs. Run it with `python -m unittest tests/test_site_assignment.py`.

| Check | Result |
| --- | --- |
| Rows before and after the join | 178,080 → 178,080 (asserted by the script and the tests) |
| `obs_id` unique, in original order, and original coordinates unchanged | Yes (tested) |
| Every row assigned or given an unassigned reason | Yes (tested) |
| Same-day conflicts within any site | 0 (tested) |
| Assigned observations | 177,007 (99.40%) |
| Unassigned: drifting / isolated | 523 / 550 |
| Sites with documented coordinates | 65 of 80 |
| Sites with a single dominant series | 79 of 80 |
| Site spread: median distance to nominal | 4.6 km median across sites (range 0–69 km) |
| Site spread: 95th-percentile distance to nominal | 17 km median across sites (range 0–102 km) |
| Reruns | Byte-identical outputs |

### Why not 219?

The final count is 80, which is 139 short of the benchmark. The evidence says 219 was an artefact of binning:

- **The file contains only 76 buoy series.** The count rises to 80 because four series each cover two separate stations: 13, 14, 58 and 60.
- **Rounding coordinates to 1° gives 217 pairs,** very close to 219. Rounding splits sites that sit near a half-degree boundary and turns every drift-track position into its own "location".
- **Getting to about 219 clusters requires `eps` of about 2 km.** At that setting, 45 of the 76 series break into pieces, so single moorings get counted many times.
- **65 sites match the official TAO/TRITON list.** The remaining 15 are early or short-lived moorings from the 1980s and the TOGA-COARE period (1992–94).

`Challenge-Project-Overview.md` has been updated with the correct figure.

## Remaining limitations

| Item | Detail |
| --- | --- |
| 15 historical sites | No official record of their positions is online in the sources above. Their nominal coordinates are the grid convention (5 sites) or the medoid (10 sites). They hold 13,698 observations. |
| `SITE_046` 2N137E | The nearest official site, TRITON 2N138E, is 108 km away and started in 1999, after this data ends. This is treated as the earlier site at 137E. |
| Merged relocations (7 sites) | Merged on the series evidence and the SST check. There is no deployment-by-deployment record to confirm them. |
| Stitched series 13, 58, 60 | Each covers two stations at least 390 km apart and is correctly split into two sites. |
| `SITE_029` 0.01S157.51E | 2 of its 460 observations come from the 0N156E buoy passing through (review note `multiple_series`). |

## Parameters

All parameters are constants at the top of `src/site_assignment.py`:

| Constant | Value | Why |
| --- | --- | --- |
| `DRIFT_KM_PER_DAY` | 10 | About 9 times the median daily movement of a moored buoy, and above its 99.5th percentile. |
| `EPS_KM` | 25 | The largest `eps` with no same-day conflicts. |
| `MIN_OBS` | 30 | Roughly one month of daily observations at one position. |
| `RELOCATION_MERGE_KM` | 100 | Covers every remaining relocation (38–85 km) and excludes the 130 km 0N95W position, which measured colder water. |
| `DOC_MATCH_KM` | 75 | Sits in the gap between the furthest documented match (69 km) and the nearest non-match (108 km). |
| `GRID_SNAP_DEG` | 0.15 | Applies only to sites with no documented match. Snaps historical sites that sit on the naming grid. |
| `FAR_FROM_NOMINAL_KM` | 50 | Review flag only. It does not exclude rows (see [Flagged rows](#flagged-rows-keep-them)). |
