# Flight Delay Prediction Using Machine Learning

This project predicts whether a U.S. domestic flight will arrive more than 15 minutes late.

The final system uses an XGBoost classification pipeline and makes predictions immediately after departure using actual departure information, including departure time and departure delay.

## Live Application

Open the deployed Streamlit application:

https://omar-flight-delay-predictor.streamlit.app

## GitHub Repository

https://github.com/omarzakzook/Flight_Delay_Prediction

## Final Model Performance

The final model was trained using flight data from 2019 through 2022 and evaluated once on the untouched 2023 test dataset.

| Metric | Test Result |
|---|---:|
| Accuracy | 93% |
| Precision for delayed flights | 92% |
| Recall for delayed flights | 73% |
| F1-score for delayed flights | 81% |

## Prediction Target

The target variable is:

- `1`: Arrival delay greater than 15 minutes
- `0`: Arrival delay of 15 minutes or less

Cancelled and diverted flights were excluded from model training.

## Prediction Scenario

The final model is an after-departure arrival-delay prediction model.

It uses:

- Actual departure time
- Departure delay
- Scheduled departure and arrival times
- Airline
- Origin and destination airports
- Flight distance
- Scheduled duration
- Date and seasonal features

Because actual departure information is used, the model must not be described as a pre-departure prediction system.

## Project Structure

```text
Flight_Delay_Prediction/
├── app.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
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
