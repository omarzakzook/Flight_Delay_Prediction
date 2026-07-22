from datetime import date, time
import hashlib
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
)


# ---------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------

BASE_DIRECTORY = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIRECTORY / "models" / "final_after_departure_xgboost_pipeline.joblib"

METADATA_PATH = BASE_DIRECTORY / "models" / "deployment_metadata.joblib"


# ---------------------------------------------------------
# Load and validate deployment artifacts
# ---------------------------------------------------------


@st.cache_resource
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file was not found: {METADATA_PATH}")

    model_pipeline = joblib.load(MODEL_PATH)
    deployment_metadata = joblib.load(METADATA_PATH)

    required_metadata_fields = [
        "artifact_version",
        "prediction_moment",
        "target_definition",
        "decision_threshold",
        "feature_names",
        "airlines",
        "origins",
        "destinations",
        "test_metrics",
    ]

    missing_fields = [
        field for field in required_metadata_fields if field not in deployment_metadata
    ]

    if missing_fields:
        raise KeyError(
            "Deployment metadata is missing required fields: " f"{missing_fields}"
        )

    expected_hash = deployment_metadata.get("model_sha256")

    if expected_hash:
        actual_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

        if actual_hash != expected_hash:
            raise RuntimeError(
                "The model file does not match the deployment " "metadata SHA-256 hash."
            )

    pipeline_features = list(
        getattr(
            model_pipeline,
            "feature_names_in_",
            deployment_metadata["feature_names"],
        )
    )

    metadata_features = list(deployment_metadata["feature_names"])

    if pipeline_features != metadata_features:
        raise RuntimeError(
            "The model feature contract does not match " "the deployment metadata."
        )

    return model_pipeline, deployment_metadata


try:
    model, metadata = load_resources()

except Exception as error:
    st.error("The model or deployment metadata could not be loaded.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Feature-engineering helpers
# ---------------------------------------------------------


def time_to_hhmm(selected_time: time) -> int:
    """Convert a time object to the dataset's HHMM format."""
    return selected_time.hour * 100 + selected_time.minute


def determine_time_of_day(hour: int) -> str:
    """Apply the same time-of-day logic used in training."""
    if 5 <= hour < 12:
        return "Morning"

    if 12 <= hour < 17:
        return "Afternoon"

    if 17 <= hour < 21:
        return "Evening"

    return "Night"


def determine_season(month: int) -> str:
    """Convert a month to its engineered season."""
    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Spring"

    if month in [6, 7, 8]:
        return "Summer"

    return "Autumn"


def determine_distance_category(
    distance: float,
) -> str:
    """Apply the distance buckets used during training."""
    if distance < 500:
        return "Short"

    if distance < 1500:
        return "Medium"

    return "Long"


def determine_peak_season(month: int) -> int:
    """Identify the peak-season months used in training."""
    return int(month in [6, 7, 8, 11, 12])


def determine_busy_hour(hour: int) -> int:
    """Identify the busy scheduled-departure hours."""
    return int(7 <= hour <= 10 or 16 <= hour <= 19)


# ---------------------------------------------------------
# Metadata values
# ---------------------------------------------------------

airlines = metadata["airlines"]
origins = metadata["origins"]
destinations = metadata["destinations"]

expected_features = list(metadata["feature_names"])

decision_threshold = float(metadata.get("decision_threshold", 0.5))

default_elapsed_time = float(
    metadata.get(
        "defaults",
        {},
    ).get(
        "CRS_ELAPSED_TIME",
        120.0,
    )
)

default_distance = float(
    metadata.get(
        "defaults",
        {},
    ).get(
        "DISTANCE",
        700.0,
    )
)

test_metrics = metadata.get(
    "test_metrics",
    {},
)

if not airlines or not origins or not destinations:
    st.error(
        "The deployment metadata does not contain valid " "airline and airport options."
    )
    st.stop()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("✈️ Flight Arrival Delay Predictor")

st.write("""
    This application predicts whether a flight will arrive
    **15 minutes or more late**.
    """)

st.warning("""
    This is an **after-departure prediction model** because
    it uses the actual departure time and departure delay.
    It cannot generate a prediction before departure.
    """)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Model Information")

    st.metric(
        "Final Test Accuracy",
        f"{test_metrics.get('Accuracy', 0):.2%}",
    )

    st.metric(
        "Delayed-Flight Precision",
        f"{test_metrics.get('Precision', 0):.2%}",
    )

    st.metric(
        "Delayed-Flight Recall",
        f"{test_metrics.get('Recall', 0):.2%}",
    )

    st.metric(
        "Delayed-Flight F1",
        f"{test_metrics.get('F1', 0):.2%}",
    )

    st.metric(
        "Final Test ROC-AUC",
        f"{test_metrics.get('ROC_AUC', 0):.2%}",
    )

    st.divider()

    st.caption("""
        The final pipeline was trained using flight records
        from 2019 through 2022 and evaluated on the
        January–August 2023 temporal holdout.
        """)

    st.caption(f"Artifact version: " f"{metadata['artifact_version']}")


# ---------------------------------------------------------
# Prediction form
# ---------------------------------------------------------

with st.form("flight_prediction_form"):
    st.subheader("Flight Details")

    first_column, second_column, third_column = st.columns(3)

    with first_column:
        airline = st.selectbox(
            "Airline",
            options=airlines,
        )

        origin = st.selectbox(
            "Origin Airport",
            options=origins,
            index=0,
        )

        destination = st.selectbox(
            "Destination Airport",
            options=destinations,
            index=(1 if len(destinations) > 1 else 0),
        )

    with second_column:
        flight_date = st.date_input(
            "Flight Date",
            value=date(2023, 6, 15),
        )

        scheduled_departure_time = st.time_input(
            "Scheduled Departure Time",
            value=time(10, 0),
        )

        actual_departure_time = st.time_input(
            "Actual Departure Time",
            value=time(10, 0),
        )

    with third_column:
        scheduled_arrival_time = st.time_input(
            "Scheduled Arrival Time",
            value=time(12, 0),
        )

        departure_delay = st.number_input(
            "Departure Delay — Minutes",
            min_value=-120.0,
            max_value=1000.0,
            value=0.0,
            step=1.0,
            help=(
                "Use a negative value when the flight "
                "departed earlier than scheduled."
            ),
        )

        scheduled_elapsed_time = st.number_input(
            "Scheduled Flight Duration — Minutes",
            min_value=1.0,
            max_value=1500.0,
            value=default_elapsed_time,
            step=1.0,
        )

        distance = st.number_input(
            "Flight Distance — Miles",
            min_value=1.0,
            max_value=10000.0,
            value=default_distance,
            step=10.0,
        )

    st.caption("""
        Time of day, season, distance category, weekend,
        peak season, and busy-hour features are calculated
        automatically using the same rules used during
        model training.
        """)

    submitted = st.form_submit_button(
        "Predict Flight Delay",
        width="stretch",
    )


# ---------------------------------------------------------
# Generate prediction
# ---------------------------------------------------------

if submitted:
    if origin == destination:
        st.error("Origin and destination airports must be different.")

    else:
        year = flight_date.year
        month = flight_date.month
        day = flight_date.day

        day_of_week = flight_date.strftime("%A")
        quarter = ((month - 1) // 3) + 1

        scheduled_departure_hhmm = time_to_hhmm(scheduled_departure_time)

        actual_departure_hhmm = time_to_hhmm(actual_departure_time)

        scheduled_arrival_hhmm = time_to_hhmm(scheduled_arrival_time)

        departure_hour = scheduled_departure_time.hour

        time_of_day = determine_time_of_day(departure_hour)

        season = determine_season(month)

        distance_category = determine_distance_category(distance)

        is_weekend = int(
            day_of_week
            in [
                "Saturday",
                "Sunday",
            ]
        )

        is_peak_season = determine_peak_season(month)

        is_busy_hour = determine_busy_hour(departure_hour)

        input_data = pd.DataFrame(
            [
                {
                    "AIRLINE": airline,
                    "ORIGIN": origin,
                    "DEST": destination,
                    "CRS_DEP_TIME": (scheduled_departure_hhmm),
                    "DEP_TIME": (actual_departure_hhmm),
                    "DEP_DELAY": departure_delay,
                    "CRS_ARR_TIME": (scheduled_arrival_hhmm),
                    "CRS_ELAPSED_TIME": (scheduled_elapsed_time),
                    "DISTANCE": distance,
                    "YEAR": year,
                    "MONTH": month,
                    "DAY": day,
                    "DAY_OF_WEEK": day_of_week,
                    "QUARTER": quarter,
                    "DEP_HOUR": departure_hour,
                    "TIME_OF_DAY": time_of_day,
                    "SEASON": season,
                    "DISTANCE_CATEGORY": (distance_category),
                    "IS_WEEKEND": is_weekend,
                    "IS_PEAK_SEASON": (is_peak_season),
                    "IS_BUSY_HOUR": is_busy_hour,
                }
            ]
        )

        missing_features = [
            feature
            for feature in expected_features
            if feature not in input_data.columns
        ]

        unexpected_features = [
            feature
            for feature in input_data.columns
            if feature not in expected_features
        ]

        if missing_features or unexpected_features:
            st.error(
                "The generated model input does not match "
                "the saved feature contract."
            )

            st.write(
                "Missing features:",
                missing_features,
            )

            st.write(
                "Unexpected features:",
                unexpected_features,
            )

            st.stop()

        input_data = input_data.loc[
            :,
            expected_features,
        ]

        try:
            model_score = float(model.predict_proba(input_data)[0, 1])

            prediction = int(model_score >= decision_threshold)

            st.divider()
            st.subheader("Prediction Result")

            result_column, score_column = st.columns(2)

            with result_column:
                st.metric(
                    "Predicted Status",
                    ("Delayed" if prediction == 1 else "On Time"),
                )

            with score_column:
                st.metric(
                    "Delayed-Class Model Score",
                    f"{model_score:.1%}",
                )

            st.progress(model_score)

            st.caption("""
                This is an uncalibrated model score, not a
                guaranteed real-world probability.
                """)

            if prediction == 1:
                st.error("""
                    The model predicts that this flight is
                    likely to arrive 15 minutes or more late.
                    """)

            else:
                st.success("""
                    The model predicts that this flight is
                    likely to arrive less than 15 minutes late.
                    """)

            with st.expander("View Generated Model Input"):
                st.dataframe(
                    input_data,
                    width="stretch",
                )

            if year < 2019 or year > 2023:
                st.info("""
                    This date is outside the period used for
                    model development and evaluation. The
                    prediction should therefore be interpreted
                    cautiously.
                    """)

        except Exception as error:
            st.error("The prediction could not be generated.")
            st.exception(error)
