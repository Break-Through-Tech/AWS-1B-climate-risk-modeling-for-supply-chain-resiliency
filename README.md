# Climate Risk Modeling for Supply Chain Resiliency

A predictive risk model that classifies climate anomalies from Pacific buoy sensor data and gives organizations months of lead time before supply chain disruptions hit.

## Business Problem

Global climate events, particularly the El Nino-Southern Oscillation (ENSO), cause severe supply chain disruptions, logistics delays, and infrastructure vulnerabilities. This project builds a predictive risk model that forecasts SSTAs and classifies climate risk levels, giving organizational stakeholders early warning to mitigate downstream impacts.

## Dataset

**UCI Machine Learning Repository: El Nino Dataset (ID: 122)**
https://archive.ics.uci.edu/dataset/122/el+nino

The dataset contains spatial-temporal oceanographic and atmospheric measurements from buoy sensors, including sea surface temperatures, air temperatures, wind vectors, humidity, and subsurface readings across the tropical Pacific.

### Data Dictionary

| Feature           | Description                         | Unit    |
| ----------------- | ----------------------------------- | ------- |
| `latitude`        | Buoy latitude position              | Degrees |
| `longitude`       | Buoy longitude position             | Degrees |
| `year`            | Observation year                    | YYYY    |
| `month`           | Observation month                   | 1-12    |
| `day`             | Observation day                     | 1-31    |
| `date`            | Full observation date               | Date    |
| `ss_temp`         | Sea surface temperature             | Celsius |
| `air_temp`        | Air temperature                     | Celsius |
| `humidity`        | Relative humidity                   | Percent |
| `zonal_wind`      | East-west wind component (u-wind)   | m/s     |
| `meridional_wind` | North-south wind component (v-wind) | m/s     |
| `subsurface_temp` | Subsurface ocean temperature        | Celsius |

_Note: Some features may contain missing values due to sensor malfunctions. Programmatic imputation is part of the pipeline._

### Buoy sites (`site_id`)

Raw coordinates drift, so the data has 8,536 distinct lat/lon pairs. `python src/site_assignment.py` groups them into 80 physical mooring sites (65 matched to the official TAO/TRITON site list) and writes `data/processed/site_assignments.csv`, with one `site_id` per observation. Group by `site_id`, not by raw coordinates, and keep rows where `use_for_climatology` is true. See [docs/site_id.md](docs/site_id.md) for the method, validation and unresolved cases.

To explore the sites interactively, open `reports/site_id/site_explorer.html` in a web browser (in Finder, double-click it or drag it into Chrome). VS Code and GitHub show only the code, because the page is drawn by JavaScript.

## Chronological split

The buoy record is split on the calendar date built from `year`, `month`, and `day`. A year stored as 80 through 98 means 1900 plus that year. A year stored as 1980 through 1998 is kept. Each range includes its first and last day, and the next range starts on the following day.

| Partition | First day | Last day |
| --- | --- | --- |
| train | 1980-03-07 | 1994-12-31 |
| validation | 1995-01-01 | 1996-12-31 |
| test | 1997-01-01 | 1998-06-23 |

`n_splits` is 5. `python -m src.split` reads `data/el_nino_features.csv` and writes `data/processed/train.csv`, `data/processed/validation.csv`, and `data/processed/test.csv`. Each file keeps the original columns and adds `split`, whose values are `train`, `validation`, and `test`. The raw table stays at `data/el_nino_features.csv`.

Climatology, imputation, and any scaler must be fit only on the fitting partition and then applied forward. During chronological CV, this means refitting inside each fold. Interim baseline preparation exists; the final shared cleaned feature/target dataset remains an upstream handoff.

Lags and rolling windows look backward inside one site.

The evaluation pipeline evaluates a future local anomaly class: the first valid same-site observation from 90 through 97 days after the issue date. Future target information must never be used as an issue-time feature. See the [evaluation contract](docs/evaluation.md) for target thresholds and eligibility rules.

The existing `src.split` utility emits five zero-gap folds on distinct training dates. These legacy split examples are not the evaluation model-selection protocol. On the current raw file:

1. train 1980-03-07 through 1982-11-16, test 1982-11-17 through 1985-09-09
2. train 1980-03-07 through 1985-09-09, test 1985-09-10 through 1988-01-07
3. train 1980-03-07 through 1988-01-07, test 1988-01-08 through 1990-05-06
4. train 1980-03-07 through 1990-05-06, test 1990-05-07 through 1992-09-02
5. train 1980-03-07 through 1992-09-02, test 1992-09-03 through 1994-12-31

The evaluation protocol requires three expanding calendar folds scoring 1989–90, 1991–92, and 1993–94, with requested and observed target dates purged at each partition boundary. Validation covers 1995–96; the final held-out test is the 1997–98 block and is scored only after validation selection is frozen.

## Baselines

`python -m src.baselines` scores three baselines on the three-month-ahead risk class: majority class, persistence (today's class stays the same) and logistic regression. It writes predictions, scores and settings to `reports/baselines/` and does not touch the test block unless `--use-test` is passed. Until the final target and feature table exists, run `python -m src.interim_dataset` first to build a stand-in. See [docs/baselines.md](docs/baselines.md) for the input contract, outputs and interim results. The reported 0.517 validation Macro F1 for persistence is a historical interim result. Final comparisons require regenerated predictions on the shared final dataset after the training-boundary and CV updates described below.

## Project Goals

**Primary:** Build a classification model using Gradient Boosted Trees and time-series feature engineering that significantly outperforms a naive baseline at predicting climate risk thresholds (extreme warming/cooling spikes vs. normal states), evaluated on Macro F1-Score.

**Stretch:** Explore temporal LSTMs (PyTorch) for sequence modeling, or wrap the trained model into a minimal interactive dashboard (Streamlit or Gradio) for real-time risk scoring demonstrations.

## Evaluation Metrics

| Criterion              | What We Measure                                                                                                              |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Pipeline Completeness  | Modular ingestion of raw UCI data, programmatic imputation of sensor logs, and reproducible train/validation/test splits     |
| Predictive Performance | Macro F1-Score that significantly outperforms a naive baseline, capturing rare high-impact anomalies                         |
| Model Interpretability | Feature importance or SHAP value plots explaining which variables drive risk flags for business audiences                    |
| Professional Handoff   | Clean repository, well-commented code, data dictionary, and a final presentation translating findings into business insights |

## Timeline

| Month     | Milestone                                 | Key Activities                                                                                                                                                                         |
| --------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| September | Technical scoping document                | Deconstruct the business problem, initialize coding environments, define team roles and task distributions, confirm initial data loading                                               |
| October   | Modular preprocessing script              | Clean the raw dataset, engineer time-series features, implement programmatic imputation, produce reproducible train/validation/test splits                                             |
| November  | Final model, repository, and presentation | Train and tune models, serialize model artifacts, generate SHAP plots and a validation leaderboard, finalize a production-grade GitHub repo, deliver a live presentation to leadership |

## EXAMPLE Repository Structure

```
.
├── README.md
├── data/
│   ├── raw/              # Original UCI dataset files
│   └── processed/        # Cleaned, feature-engineered outputs
├── notebooks/            # Exploratory analysis and prototyping
├── src/
│   ├── ingestion.py      # Data loading from UCI repository
│   ├── preprocessing.py  # Cleaning, imputation, scaling
│   ├── features.py       # Time-series feature engineering
│   ├── train.py          # Model training and hyperparameter tuning
│   ├── evaluate.py       # Metrics, SHAP plots, leaderboard generation
│   └── predict.py        # Inference on new observations
├── models/               # Serialized model artifacts
├── reports/              # Figures, SHAP plots, presentation materials
├── requirements.txt
└── .gitignore
```

## Getting Started

Our team uses **VS Code locally** as the primary development environment.

1. Clone this repository and open its folder in VS Code.
2. Read `README.md`.
3. Install the requirements from `requirements.txt`.

### Target variable

The dataset doesn't come with a pre-labeled target variable. The target would be climate risk class derived from Sea Surface Temperature Anomalies (SSTAs). An example is:

- Normal: SSTA within a threshold (e.g., -0.5 to +0.5 degrees C)
- Warming risk: SSTA above +0.5 (El Nino signal)
- Cooling risk: SSTA below -0.5 (La Nina signal)

### Prerequisites

- Python 3.10+
- pip or conda
- A virtual environment

### Installation

```bash
git clone https://github.com/Break-Through-Tech/AWS-1B-climate-risk-modeling-for-supply-chain-resiliency.git
cd AWS-1B-climate-risk-modeling-for-supply-chain-resiliency
pip install -r requirements.txt
```

## Tech Stack

- **Runtime:** Python 3.10+ with pandas, NumPy, and scikit-learn
- **Site assignment:** pandas, NumPy, scikit-learn DBSCAN, and matplotlib
- **Data acquisition:** `ucimlrepo` in `notebooks/fetch_el_nino.py`
- **Planned modeling:** XGBoost or LightGBM, with SHAP interpretability
- **Planned extensions:** PyTorch sequence models and a Streamlit or Gradio dashboard

### Choice of primary environment

VS Code

### Team

_Roles and assignments to be defined in the September scoping document._

Tina Zeng
Tejaswi Amatya
Jonathan Cortez
Ayushi Das
Shivangi Manel

## Current Implementation Workflow

The codebase currently implements data acquisition support, physical-site
reconstruction, site-quality validation, chronological splitting, and an
exploratory target-building path. Interim baseline training and prediction,
saved-prediction evaluation, and LightGBM explanation commands are also available.
Final shared data/features, corrected baseline training, and actual LightGBM
artifacts remain integration dependencies; a deployed dashboard is not included.

Run the implemented workflow from the repository root:

```bash
# Optional: fetch the UCI dataset into a pandas dataframe
python notebooks/fetch_el_nino.py

# Assign observations to physical mooring sites
python src/site_assignment.py

# Validate SST behavior for observations far from nominal site locations
python src/site_quality_check.py

# Generate the standalone interactive site explorer
python src/site_explorer.py

# Create chronological train/validation/test files
python -m src.split

# Run the tests
python -m unittest tests/test_site_assignment.py tests/test_split.py
```

### Workflow stages

1. **Site assignment:** `src/site_assignment.py` adds an original-row
   `obs_id`, recovers station-series boundaries from date order, flags fast
   drift, clusters stable coordinates with haversine DBSCAN, merges nearby
   relocations, and attaches stable `site_id` values. It writes
   `data/processed/site_assignments.csv`, `data/processed/site_lookup.csv`, and
   site reports under `reports/site_id/`.
2. **Quality check:** `src/site_quality_check.py` compares SST relative to
   neighboring sites before and after on/off-position transitions. The result is
   `reports/site_id/flagged_sst_check.csv`; assigned rows remain eligible for
   climatology through `use_for_climatology`.
3. **Interactive exploration:** `src/site_explorer.py` combines the assignment
   outputs with `src/site_explorer_template.html` and writes the standalone
   `reports/site_id/site_explorer.html` page.
4. **Chronological split:** `src/split.py` converts two-digit years, validates
   dates, sorts rows by date and coordinates, and writes
   `data/processed/train.csv`, `validation.csv`, and `test.csv`. The split
   boundaries are the ones documented above; random shuffling is not used.
5. **Exploratory target construction:** `notebooks/preprocessing.py` joins
   assignments, fits a site/month SST climatology on training rows, calculates
   SSTA, creates nine current ENSO classes, and looks up the first available
   same-site class at least three months ahead. It returns a modeling dataframe
   but does not save one or train a model.

### Repository outputs

| Location | Description |
| --- | --- |
| `data/processed/site_assignments.csv` | One site-assignment row per raw observation |
| `data/processed/site_lookup.csv` | Site-level nominal coordinates and statistics |
| `data/processed/train.csv` | Chronological training partition |
| `data/processed/validation.csv` | Chronological validation partition |
| `data/processed/test.csv` | Chronological held-out test partition |
| `reports/site_id/` | Clustering diagnostics, maps, review queue, SST check, and explorer |
| `docs/site_id.md` | Detailed site-identification method and validation |

For detailed site-assignment rationale and output columns, see
[docs/site_id.md](docs/site_id.md). The repository tests are in
[tests/test_site_assignment.py](tests/test_site_assignment.py) and
[tests/test_split.py](tests/test_split.py).

## Evaluation and LightGBM explanations

The evaluation commands consume saved predictions and verified training
receipts; they do not train or tune project models. See
[docs/evaluation.md](docs/evaluation.md) for the input contracts, synthetic
smoke workflow, dependency setup, and explicit final-test gate.

```bash
python -m src.evaluate --manifest path/to/validation-manifest.json
python -m src.explain --evaluation-run reports/evaluation/RUN_ID
```

LightGBM explanations run when LightGBM is included in the comparison, even
when another model wins. Real final comparisons require recorded team agreement
on the target and cleaned input, upstream uniqueness/leakage checks, corrected
baseline training, and actual split-specific model artifacts. The
[evaluation guide](docs/evaluation.md) documents the required inputs and checks.
Synthetic test success does not establish project results.

This implementation branch is based on the baseline work in PR #25. Its current
trainer still needs the observed-target boundary and training-only tuning
updates before it can supply an accepted final comparison.

## License

TBD
