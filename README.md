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

Raw coordinates drift, so the data has 8,536 distinct lat/lon pairs. `python src/site_assignment.py` groups them into 79 physical mooring sites and writes `data/processed/site_assignments.csv`, with one `site_id` per observation. Group by `site_id`, not by raw coordinates. See [docs/site_id.md](docs/site_id.md) for the method, validation and unresolved cases.

## Chronological split

The buoy record is split on the calendar date built from `year`, `month`, and `day`. A year stored as 80 through 98 means 1900 plus that year. A year stored as 1980 through 1998 is kept. Each range includes its first and last day, and the next range starts on the following day.

| Partition | First day | Last day |
| --- | --- | --- |
| train | 1980-03-07 | 1994-12-31 |
| validation | 1995-01-01 | 1996-12-31 |
| test | 1997-01-01 | 1998-06-23 |

`n_splits` is 5. `python -m src.split` reads `data/el_nino_features.csv` and writes `data/processed/train.csv`, `data/processed/validation.csv`, and `data/processed/test.csv`. Each file keeps the original columns and adds `split`, whose values are `train`, `validation`, and `test`. The raw table stays at `data/el_nino_features.csv`.

Climatology, imputation, rolling windows, and any scaler must be fit on dates through 1994-12-31 and then applied forward. Climatology is **not-yet**. Imputation is **not-yet**. Feature windows are **not-yet**. Those artifacts do not exist.

Lags and rolling windows look backward inside one site.

The label is the anomaly on the same row, so that row's sea surface temperature is not also a feature.

Time series folds use 5 splits and a gap of 0 on the distinct training dates only. On the current raw file:

1. train 1980-03-07 through 1982-11-16, test 1982-11-17 through 1985-09-09
2. train 1980-03-07 through 1985-09-09, test 1985-09-10 through 1988-01-07
3. train 1980-03-07 through 1988-01-07, test 1988-01-08 through 1990-05-06
4. train 1980-03-07 through 1990-05-06, test 1990-05-07 through 1992-09-02
5. train 1980-03-07 through 1992-09-02, test 1992-09-03 through 1994-12-31

Folds whose training side ends before 1992 are the small early array. The official test score is the 1997–98 block.

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

- **Data processing:** pandas, NumPy, scikit-learn
- **Modeling:** XGBoost or LightGBM (Gradient Boosted Trees)
- **Interpretability:** SHAP
- **Stretch (deep learning):** PyTorch (LSTM)
- **Stretch (dashboard):** Streamlit or Gradio

### Choice of primary environment

VS Code

### Team

_Roles and assignments to be defined in the September scoping document._

Tina Zeng
Tejaswi Amatya
Jonathan Cortez
Ayushi Das

## License

TBD
