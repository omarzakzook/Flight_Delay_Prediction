# Flight Arrival Delay Prediction Using Machine Learning

This project predicts whether a completed U.S. domestic flight will arrive **15 minutes or more late**.

The final system uses an XGBoost classification pipeline and generates predictions **immediately after departure** using actual departure time and departure delay together with scheduled, route, airline, distance, and calendar information.

## Live Application

https://omar-flight-delay-predictor.streamlit.app

## GitHub Repository

https://github.com/omarzakzook/Flight_Delay_Prediction

## Prediction Target

The project is formulated as a binary classification problem:

- `1` — Arrival delay is **15 minutes or more**
- `0` — Arrival delay is **less than 15 minutes**

The target is created using:

```python
IS_DELAYED = (ARR_DELAY >= 15).astype(int)
```

Cancelled and diverted flights are excluded from the modeling population.

## Prediction Scenario

The final model is an **after-departure arrival-delay prediction model**.

It uses 21 features, including:

- Actual departure time
- Departure delay
- Scheduled departure and arrival times
- Scheduled flight duration
- Airline
- Origin and destination airports
- Flight distance
- Calendar and seasonal features
- Automatically derived time-of-day, distance-category, peak-season, and busy-hour features

Because actual departure information is required, this system must not be described as a pre-departure prediction model.

The delayed-class model score displayed by the application is not calibrated as a guaranteed real-world probability.

## Dataset

The project uses the **Flight Delay and Cancellation Dataset** obtained from Kaggle.

### Raw Dataset

- Records: approximately 3,000,000
- Original columns: 32
- Period: January 2019 through August 2023
- Raw filename: `flights_sample_3m.csv`

### Cleaned Modeling Population

- Completed, non-diverted flights: 2,913,802
- On-time flights: 2,379,939
- Delayed flights: 533,863

The raw and processed CSV files are not committed to Git because of their size.

Place the downloaded raw file at:

```text
data/raw/flights_sample_3m.csv
```

The cleaning and feature-engineering notebooks generate the processed datasets locally inside:

```text
data/processed/
```

## Temporal Evaluation Protocol

The project uses chronological periods instead of a random train-test split:

- Training and model development: 2019–2021
- Validation and model comparison: 2022
- Final temporal holdout: January–August 2023

After model selection, the final pipeline was trained using 2019–2022 data and evaluated once on the January–August 2023 holdout.

The temporal holdout was not used for model fitting, hyperparameter tuning, threshold selection, or model comparison.

## Final Model Performance

| Metric | Temporal Holdout Result |
|---|---:|
| Accuracy | 92.24% |
| Delayed-flight precision | 92.07% |
| Delayed-flight recall | 72.55% |
| Delayed-flight F1-score | 81.16% |
| ROC-AUC | 93.42% |

## Models Evaluated

The project compares:

- Logistic Regression
- Random Forest
- XGBoost
- Temporally tuned XGBoost
- An improved after-departure XGBoost model

The final after-departure model substantially outperformed the pre-departure models because departure delay and actual departure time provide strong information about whether a flight can recover before arrival.

## Project Structure

```text
Flight_Delay_Prediction/
├── app.py
├── README.md
├── requirements.txt
├── requirements-notebooks.txt
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── models/
│   ├── final_after_departure_xgboost_pipeline.joblib
│   ├── deployment_metadata.joblib
│   └── final_test_metrics.csv
└── notebooks/
    ├── 01_Business_Understanding.ipynb
    ├── 02_Data_Understanding.ipynb
    ├── 03_Data_Cleaning.ipynb
    ├── 04_Feature_Engineering.ipynb
    ├── 05_Exploratory_Data_Analysis.ipynb
    ├── 06_Data_Preparation_for_Machine_Learning.ipynb
    ├── 07_Model_Building.ipynb
    ├── 08_Model_Improvement.ipynb
    └── 09_Final_Model_Evaluation.ipynb
```

## Notebook Workflow

| Notebook | Purpose |
|---|---|
| 01 | Business understanding, scope, objectives, and limitations |
| 02 | Initial data understanding |
| 03 | Data cleaning and target creation |
| 04 | Feature engineering and leakage control |
| 05 | Exploratory analysis using the training period |
| 06 | Temporal splitting and preprocessing |
| 07 | Baseline comparison and temporal hyperparameter tuning |
| 08 | After-departure model improvement |
| 09 | Final training, temporal holdout evaluation, and artifact saving |

## Installation

Create and activate a Python environment, then install the Streamlit application dependencies:

```bash
pip install -r requirements.txt
```

To run the notebooks, install the extended notebook environment:

```bash
pip install -r requirements-notebooks.txt
```

## Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application loads the final pipeline and deployment metadata from the `models/` directory.

## Main Limitations

- Predictions are available only after the aircraft departs.
- Cancelled and diverted flights are outside the model scope.
- The 2023 holdout ends in August and is not a complete calendar year.
- The dataset does not contain detailed real-time weather, maintenance, or air-traffic information.
- The model score is not probability-calibrated.
- Predictions outside the 2019–2023 development period should be interpreted cautiously.
