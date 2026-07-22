from datetime import date, time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="wide")


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIRECTORY = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIRECTORY / "models" / "final_after_departure_xgboost_pipeline.joblib"

METADATA_PATH = BASE_DIRECTORY / "models" / "deployment_metadata.joblib"


# ---------------------------------------------------------
# Load model and metadata
# ---------------------------------------------------------


@st.cache_resource
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file was not found: {METADATA_PATH}")

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    return model, metadata


try:
    model, metadata = load_resources()

except Exception as error:
    st.error("The model or deployment metadata could not be loaded.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------


def time_to_hhmm(selected_time: time) -> int:
    """
    Convert a Python time object into the HHMM numerical format
    used in the training dataset.

    Example:
        14:35 -> 1435
    """
    return selected_time.hour * 100 + selected_time.minute


def determine_season(month: int) -> str:
    """
    Convert a month into a meteorological season.
    """
    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Spring"

    if month in [6, 7, 8]:
        return "Summer"

    return "Autumn"


def get_encoder_categories(model_pipeline):
    """
    Extract the categorical options learned by the fitted
    OneHotEncoder inside the model pipeline.
    """
    feature_names = ["DAY_OF_WEEK", "TIME_OF_DAY", "SEASON", "DISTANCE_CATEGORY"]

    try:
        preprocessor = model_pipeline.named_steps["preprocessor"]

        onehot_pipeline = preprocessor.named_transformers_["onehot"]

        encoder = onehot_pipeline.named_steps["onehot"]

        return {
            feature_name: [str(value) for value in categories]
            for feature_name, categories in zip(feature_names, encoder.categories_)
        }

    except Exception:
        # Safe fallback values
        return {
            "DAY_OF_WEEK": [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            "TIME_OF_DAY": ["Morning", "Afternoon", "Evening", "Night"],
            "SEASON": ["Winter", "Spring", "Summer", "Autumn"],
            "DISTANCE_CATEGORY": ["Short", "Medium", "Long"],
        }


category_options = get_encoder_categories(model)

airlines = metadata.get("airlines", [])
origins = metadata.get("origins", [])
destinations = metadata.get("destinations", [])

default_elapsed_time = float(
    metadata.get("defaults", {}).get("CRS_ELAPSED_TIME", 120.0)
)

default_distance = float(metadata.get("defaults", {}).get("DISTANCE", 700.0))


# ---------------------------------------------------------
# Application header
# ---------------------------------------------------------

st.title("✈️ Flight Arrival Delay Predictor")

st.write("""
    This application predicts whether a flight will arrive
    more than 15 minutes late.
    """)

st.warning("""
    This is an **after-departure prediction model** because it
    uses the actual departure time and departure delay.
    """)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Model Information")

    st.metric("Final Test Accuracy", "93%")

    st.metric("Delayed-Flight Precision", "92%")

    st.metric("Delayed-Flight Recall", "73%")

    st.metric("Delayed-Flight F1", "81%")

    st.divider()

    st.caption("""
        The final model was trained using flight records from
        2019 through 2022 and evaluated on the untouched 2023
        dataset.
        """)


# ---------------------------------------------------------
# Prediction form
# ---------------------------------------------------------

with st.form("flight_prediction_form"):

    st.subheader("Flight Details")

    first_column, second_column, third_column = st.columns(3)

    with first_column:
        airline = st.selectbox("Airline", options=airlines)

        origin = st.selectbox("Origin Airport", options=origins)

        destination = st.selectbox("Destination Airport", options=destinations)

    with second_column:
        flight_date = st.date_input("Flight Date", value=date(2023, 6, 15))

        scheduled_departure_time = st.time_input(
            "Scheduled Departure Time", value=time(10, 0)
        )

        actual_departure_time = st.time_input(
            "Actual Departure Time", value=time(10, 0)
        )

    with third_column:
        scheduled_arrival_time = st.time_input(
            "Scheduled Arrival Time", value=time(12, 0)
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

    st.subheader("Distance and Engineered Features")

    fourth_column, fifth_column, sixth_column = st.columns(3)

    with fourth_column:
        distance = st.number_input(
            "Flight Distance — Miles",
            min_value=1.0,
            max_value=10000.0,
            value=default_distance,
            step=10.0,
        )

        distance_category = st.selectbox(
            "Distance Category",
            options=category_options["DISTANCE_CATEGORY"],
            help=(
                "Select the same category definition used "
                "during feature engineering."
            ),
        )

    with fifth_column:
        time_of_day = st.selectbox(
            "Time of Day",
            options=category_options["TIME_OF_DAY"],
            help=(
                "Select the category corresponding to the " "scheduled departure time."
            ),
        )

        is_peak_season = st.selectbox(
            "Peak Travel Season",
            options={"No": 0, "Yes": 1},
            format_func=lambda label: label,
        )

    with sixth_column:
        is_busy_hour = st.selectbox(
            "Busy Departure Hour",
            options={"No": 0, "Yes": 1},
            format_func=lambda label: label,
        )

    submitted = st.form_submit_button("Predict Flight Delay", use_container_width=True)


# ---------------------------------------------------------
# Run prediction
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

        is_weekend = int(day_of_week in ["Saturday", "Sunday"])

        season = determine_season(month)

        scheduled_departure_hhmm = time_to_hhmm(scheduled_departure_time)

        actual_departure_hhmm = time_to_hhmm(actual_departure_time)

        scheduled_arrival_hhmm = time_to_hhmm(scheduled_arrival_time)

        departure_hour = scheduled_departure_time.hour

        peak_season_value = {"No": 0, "Yes": 1}[is_peak_season]

        busy_hour_value = {"No": 0, "Yes": 1}[is_busy_hour]

        input_data = pd.DataFrame(
            [
                {
                    "AIRLINE": airline,
                    "ORIGIN": origin,
                    "DEST": destination,
                    "CRS_DEP_TIME": scheduled_departure_hhmm,
                    "DEP_TIME": actual_departure_hhmm,
                    "DEP_DELAY": departure_delay,
                    "CRS_ARR_TIME": scheduled_arrival_hhmm,
                    "CRS_ELAPSED_TIME": scheduled_elapsed_time,
                    "DISTANCE": distance,
                    "YEAR": year,
                    "MONTH": month,
                    "DAY": day,
                    "DAY_OF_WEEK": day_of_week,
                    "QUARTER": quarter,
                    "DEP_HOUR": departure_hour,
                    "TIME_OF_DAY": time_of_day,
                    "SEASON": season,
                    "DISTANCE_CATEGORY": distance_category,
                    "IS_WEEKEND": is_weekend,
                    "IS_PEAK_SEASON": peak_season_value,
                    "IS_BUSY_HOUR": busy_hour_value,
                }
            ]
        )

        try:
            prediction = int(model.predict(input_data)[0])

            delay_probability = float(model.predict_proba(input_data)[0, 1])

            st.divider()
            st.subheader("Prediction Result")

            result_column, probability_column = st.columns(2)

            with result_column:
                st.metric(
                    "Predicted Status", ("Delayed" if prediction == 1 else "On Time")
                )

            with probability_column:
                st.metric("Delay Probability", f"{delay_probability:.1%}")

            st.progress(delay_probability)

            if prediction == 1:
                st.error("""
                    The model predicts that this flight is
                    likely to arrive more than 15 minutes late.
                    """)

            else:
                st.success("""
                    The model predicts that this flight is
                    likely to arrive on time or within
                    15 minutes of its scheduled arrival.
                    """)

            with st.expander("View Model Input"):
                st.dataframe(input_data, use_container_width=True)

            if year > 2023:
                st.info("""
                    This date is later than the period used for
                    training and testing. The prediction should
                    therefore be interpreted cautiously.
                    """)

        except Exception as error:
            st.error("The prediction could not be generated.")
            st.exception(error)
