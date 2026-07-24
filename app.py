from datetime import date, time
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Flight Delay Analytics",
    page_icon="✈️",
    layout="wide",
)


# ---------------------------------------------------------
# Paths and project constants
# ---------------------------------------------------------

BASE_DIRECTORY = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIRECTORY
    / "models"
    / "final_after_departure_xgboost_pipeline.joblib"
)

METADATA_PATH = (
    BASE_DIRECTORY
    / "models"
    / "deployment_metadata.joblib"
)

ANALYSIS_DATA_PATH = (
    BASE_DIRECTORY
    / "data"
    / "app"
    / "analysis_sample.csv.gz"
)

FULL_MODELING_ROWS = 2_913_802
FULL_ON_TIME_ROWS = 2_379_939
FULL_DELAYED_ROWS = 533_863
FULL_DELAY_RATE = FULL_DELAYED_ROWS / FULL_MODELING_ROWS

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

TIME_OF_DAY_ORDER = [
    "Morning",
    "Afternoon",
    "Evening",
    "Night",
]

SEASON_ORDER = [
    "Winter",
    "Spring",
    "Summer",
    "Autumn",
]

DISTANCE_CATEGORY_ORDER = [
    "Short",
    "Medium",
    "Long",
]

NUMERICAL_ANALYSIS_COLUMNS = {
    "Departure Delay": "DEP_DELAY",
    "Flight Distance": "DISTANCE",
    "Scheduled Flight Duration": "CRS_ELAPSED_TIME",
    "Scheduled Departure Time": "CRS_DEP_TIME",
    "Actual Departure Time": "DEP_TIME",
    "Scheduled Arrival Time": "CRS_ARR_TIME",
    "Departure Hour": "DEP_HOUR",
    "Year": "YEAR",
    "Month": "MONTH",
    "Quarter": "QUARTER",
}

CATEGORICAL_ANALYSIS_COLUMNS = {
    "Delay Status": "DELAY_STATUS",
    "Airline": "AIRLINE",
    "Airline Code": "AIRLINE_CODE",
    "Origin Airport": "ORIGIN",
    "Origin City": "ORIGIN_CITY",
    "Destination Airport": "DEST",
    "Destination City": "DEST_CITY",
    "Route": "ROUTE",
    "Day of Week": "DAY_OF_WEEK",
    "Time of Day": "TIME_OF_DAY",
    "Season": "SEASON",
    "Distance Category": "DISTANCE_CATEGORY",
    "Weekend Status": "WEEKEND_STATUS",
    "Peak-Season Status": "PEAK_SEASON_STATUS",
    "Busy-Hour Status": "BUSY_HOUR_STATUS",
}


# ---------------------------------------------------------
# Cached artifact and data loading
# ---------------------------------------------------------

@st.cache_resource
def load_resources():
    """Load and validate the fitted model pipeline and metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file was not found: {METADATA_PATH}"
        )

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
        field
        for field in required_metadata_fields
        if field not in deployment_metadata
    ]

    if missing_fields:
        raise KeyError(
            "Deployment metadata is missing required fields: "
            f"{missing_fields}"
        )

    expected_hash = deployment_metadata.get("model_sha256")

    if expected_hash:
        actual_hash = hashlib.sha256(
            MODEL_PATH.read_bytes()
        ).hexdigest()

        if actual_hash != expected_hash:
            raise RuntimeError(
                "The model file does not match the deployment "
                "metadata SHA-256 hash."
            )

    pipeline_features = list(
        getattr(
            model_pipeline,
            "feature_names_in_",
            deployment_metadata["feature_names"],
        )
    )

    metadata_features = list(
        deployment_metadata["feature_names"]
    )

    if pipeline_features != metadata_features:
        raise RuntimeError(
            "The model feature contract does not match "
            "the deployment metadata."
        )

    return model_pipeline, deployment_metadata


@st.cache_data(show_spinner="Loading analysis data...")
def load_analysis_data() -> pd.DataFrame:
    """Load the compressed representative analysis sample."""
    if not ANALYSIS_DATA_PATH.exists():
        raise FileNotFoundError(
            "Analysis data was not found at: "
            f"{ANALYSIS_DATA_PATH}"
        )

    analysis_data = pd.read_csv(
        ANALYSIS_DATA_PATH,
        parse_dates=["FL_DATE"],
    )

    analysis_data["DELAY_STATUS"] = np.where(
        analysis_data["IS_DELAYED"].eq(1),
        "Delayed",
        "On Time",
    )

    analysis_data["WEEKEND_STATUS"] = np.where(
        analysis_data["IS_WEEKEND"].eq(1),
        "Weekend",
        "Weekday",
    )

    analysis_data["PEAK_SEASON_STATUS"] = np.where(
        analysis_data["IS_PEAK_SEASON"].eq(1),
        "Peak Season",
        "Regular Season",
    )

    analysis_data["BUSY_HOUR_STATUS"] = np.where(
        analysis_data["IS_BUSY_HOUR"].eq(1),
        "Busy Hour",
        "Non-Busy Hour",
    )

    analysis_data["MONTH_NAME"] = pd.Categorical(
        analysis_data["FL_DATE"].dt.month_name(),
        categories=MONTH_ORDER,
        ordered=True,
    )

    analysis_data["DAY_OF_WEEK"] = pd.Categorical(
        analysis_data["DAY_OF_WEEK"],
        categories=DAY_ORDER,
        ordered=True,
    )

    analysis_data["TIME_OF_DAY"] = pd.Categorical(
        analysis_data["TIME_OF_DAY"],
        categories=TIME_OF_DAY_ORDER,
        ordered=True,
    )

    analysis_data["SEASON"] = pd.Categorical(
        analysis_data["SEASON"],
        categories=SEASON_ORDER,
        ordered=True,
    )

    analysis_data["DISTANCE_CATEGORY"] = pd.Categorical(
        analysis_data["DISTANCE_CATEGORY"],
        categories=DISTANCE_CATEGORY_ORDER,
        ordered=True,
    )

    return analysis_data


@st.cache_data
def convert_dataframe_to_csv(dataframe: pd.DataFrame) -> bytes:
    """Convert filtered rows to downloadable CSV bytes."""
    return dataframe.to_csv(index=False).encode("utf-8")


try:
    model, metadata = load_resources()

except Exception as error:
    st.error(
        "The model or deployment metadata could not be loaded."
    )
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Feature-engineering helpers used by prediction
# ---------------------------------------------------------

def time_to_hhmm(selected_time: time) -> int:
    """Convert a Python time object to the dataset's HHMM format."""
    return selected_time.hour * 100 + selected_time.minute


def determine_time_of_day(hour: int) -> str:
    """Apply the same time-of-day grouping used during training."""
    if 5 <= hour < 12:
        return "Morning"

    if 12 <= hour < 17:
        return "Afternoon"

    if 17 <= hour < 21:
        return "Evening"

    return "Night"


def determine_season(month: int) -> str:
    """Convert a month into its engineered season."""
    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Spring"

    if month in [6, 7, 8]:
        return "Summer"

    return "Autumn"


def determine_distance_category(distance: float) -> str:
    """Apply the distance categories used during training."""
    if distance < 500:
        return "Short"

    if distance < 1500:
        return "Medium"

    return "Long"


def determine_peak_season(month: int) -> int:
    """Identify the peak-season months used during training."""
    return int(month in [6, 7, 8, 11, 12])


def determine_busy_hour(hour: int) -> int:
    """Identify the busy departure hours used during training."""
    return int(
        7 <= hour <= 10
        or 16 <= hour <= 19
    )


# ---------------------------------------------------------
# General analysis helpers
# ---------------------------------------------------------

def create_status_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return delayed/on-time counts and percentages."""
    status_table = (
        dataframe["DELAY_STATUS"]
        .value_counts(dropna=False)
        .rename_axis("Status")
        .reset_index(name="Flights")
    )

    status_table["Percentage"] = (
        status_table["Flights"]
        / status_table["Flights"].sum()
        * 100
    )

    return status_table


def create_delay_rate_table(
    dataframe: pd.DataFrame,
    group_column: str,
    minimum_flights: int = 1,
) -> pd.DataFrame:
    """Aggregate flight count, delayed count, and delay rate."""
    grouped = (
        dataframe.groupby(
            group_column,
            observed=True,
            dropna=False,
        )
        .agg(
            Flights=("IS_DELAYED", "size"),
            Delayed_Flights=("IS_DELAYED", "sum"),
            Delay_Rate=("IS_DELAYED", "mean"),
        )
        .reset_index()
    )

    grouped = grouped.loc[
        grouped["Flights"] >= minimum_flights
    ].copy()

    grouped["Delay_Rate_Percent"] = (
        grouped["Delay_Rate"] * 100
    )

    return grouped


def filter_analysis_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered dataset."""
    st.sidebar.divider()
    st.sidebar.subheader("Analysis Filters")

    minimum_date = dataframe["FL_DATE"].min().date()
    maximum_date = dataframe["FL_DATE"].max().date()

    selected_dates = st.sidebar.date_input(
        "Flight date range",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
        key="analysis_date_range",
    )

    selected_airlines = st.sidebar.multiselect(
        "Airlines",
        options=sorted(
            dataframe["AIRLINE"].dropna().unique()
        ),
        key="analysis_airlines",
    )

    selected_origins = st.sidebar.multiselect(
        "Origin airports",
        options=sorted(
            dataframe["ORIGIN"].dropna().unique()
        ),
        key="analysis_origins",
    )

    selected_destinations = st.sidebar.multiselect(
        "Destination airports",
        options=sorted(
            dataframe["DEST"].dropna().unique()
        ),
        key="analysis_destinations",
    )

    selected_statuses = st.sidebar.multiselect(
        "Delay status",
        options=["On Time", "Delayed"],
        key="analysis_statuses",
    )

    selected_days = st.sidebar.multiselect(
        "Days of week",
        options=DAY_ORDER,
        key="analysis_days",
    )

    selected_times = st.sidebar.multiselect(
        "Times of day",
        options=TIME_OF_DAY_ORDER,
        key="analysis_times",
    )

    selected_seasons = st.sidebar.multiselect(
        "Seasons",
        options=SEASON_ORDER,
        key="analysis_seasons",
    )

    selected_distance_categories = st.sidebar.multiselect(
        "Distance categories",
        options=DISTANCE_CATEGORY_ORDER,
        key="analysis_distance_categories",
    )

    departure_delay_minimum = int(
        np.floor(dataframe["DEP_DELAY"].min())
    )
    departure_delay_maximum = int(
        np.ceil(dataframe["DEP_DELAY"].max())
    )

    selected_delay_range = st.sidebar.slider(
        "Departure delay range (minutes)",
        min_value=departure_delay_minimum,
        max_value=departure_delay_maximum,
        value=(
            departure_delay_minimum,
            departure_delay_maximum,
        ),
        key="analysis_delay_range",
    )

    distance_minimum = int(
        np.floor(dataframe["DISTANCE"].min())
    )
    distance_maximum = int(
        np.ceil(dataframe["DISTANCE"].max())
    )

    selected_distance_range = st.sidebar.slider(
        "Flight distance range (miles)",
        min_value=distance_minimum,
        max_value=distance_maximum,
        value=(
            distance_minimum,
            distance_maximum,
        ),
        key="analysis_distance_range",
    )

    filtered = dataframe.copy()

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date = pd.Timestamp(selected_dates[0])
        end_date = pd.Timestamp(selected_dates[1])

        filtered = filtered.loc[
            filtered["FL_DATE"].between(
                start_date,
                end_date,
                inclusive="both",
            )
        ]

    if selected_airlines:
        filtered = filtered.loc[
            filtered["AIRLINE"].isin(selected_airlines)
        ]

    if selected_origins:
        filtered = filtered.loc[
            filtered["ORIGIN"].isin(selected_origins)
        ]

    if selected_destinations:
        filtered = filtered.loc[
            filtered["DEST"].isin(selected_destinations)
        ]

    if selected_statuses:
        filtered = filtered.loc[
            filtered["DELAY_STATUS"].isin(selected_statuses)
        ]

    if selected_days:
        filtered = filtered.loc[
            filtered["DAY_OF_WEEK"].isin(selected_days)
        ]

    if selected_times:
        filtered = filtered.loc[
            filtered["TIME_OF_DAY"].isin(selected_times)
        ]

    if selected_seasons:
        filtered = filtered.loc[
            filtered["SEASON"].isin(selected_seasons)
        ]

    if selected_distance_categories:
        filtered = filtered.loc[
            filtered["DISTANCE_CATEGORY"].isin(
                selected_distance_categories
            )
        ]

    filtered = filtered.loc[
        filtered["DEP_DELAY"].between(
            selected_delay_range[0],
            selected_delay_range[1],
            inclusive="both",
        )
    ]

    filtered = filtered.loc[
        filtered["DISTANCE"].between(
            selected_distance_range[0],
            selected_distance_range[1],
            inclusive="both",
        )
    ]

    return filtered


def show_plotly_chart(figure, key: str) -> None:
    """Render a Plotly figure with consistent Streamlit settings."""
    figure.update_layout(
        margin=dict(l=20, r=20, t=70, b=20),
        legend_title_text="",
    )

    st.plotly_chart(
        figure,
        width="stretch",
        key=key,
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )


# ---------------------------------------------------------
# Overview page
# ---------------------------------------------------------

def render_overview_page() -> None:
    st.title("✈️ Flight Delay Analytics Dashboard")

    st.write(
        """
        Explore the project dataset, compare delay patterns, and
        review the final machine-learning model through one
        interactive Streamlit application.
        """
    )

    st.info(
        """
        Overall project counts and final model metrics come from
        the complete cleaned modeling population. Interactive
        charts use a representative 150,000-row sample stratified
        by year and delay class.
        """
    )

    first_metric, second_metric, third_metric, fourth_metric = (
        st.columns(4)
    )

    first_metric.metric(
        "Completed Flights",
        f"{FULL_MODELING_ROWS:,}",
    )

    second_metric.metric(
        "Delayed Flights",
        f"{FULL_DELAYED_ROWS:,}",
    )

    third_metric.metric(
        "On-Time Flights",
        f"{FULL_ON_TIME_ROWS:,}",
    )

    fourth_metric.metric(
        "Overall Delay Rate",
        f"{FULL_DELAY_RATE:.2%}",
    )

    analysis_data = load_analysis_data()
    test_metrics = metadata["test_metrics"]

    st.subheader("Final Model Performance")

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Accuracy",
        f"{test_metrics.get('Accuracy', 0):.2%}",
    )

    metric_columns[1].metric(
        "Precision",
        f"{test_metrics.get('Precision', 0):.2%}",
    )

    metric_columns[2].metric(
        "Recall",
        f"{test_metrics.get('Recall', 0):.2%}",
    )

    metric_columns[3].metric(
        "F1-Score",
        f"{test_metrics.get('F1', 0):.2%}",
    )

    metric_columns[4].metric(
        "ROC-AUC",
        f"{test_metrics.get('ROC_AUC', 0):.2%}",
    )

    left_column, right_column = st.columns(2)

    with left_column:
        status_table = create_status_table(analysis_data)

        status_figure = px.pie(
            status_table,
            names="Status",
            values="Flights",
            hole=0.55,
            title="Delay Status Distribution",
        )

        show_plotly_chart(
            status_figure,
            "overview_status_distribution",
        )

    with right_column:
        yearly_table = create_delay_rate_table(
            analysis_data,
            "YEAR",
        ).sort_values("YEAR")

        yearly_figure = px.line(
            yearly_table,
            x="YEAR",
            y="Delay_Rate_Percent",
            markers=True,
            title="Delay Rate by Year",
            labels={
                "YEAR": "Year",
                "Delay_Rate_Percent": "Delay Rate (%)",
            },
        )

        show_plotly_chart(
            yearly_figure,
            "overview_yearly_delay_rate",
        )

    st.subheader("Project Workflow")

    st.markdown(
        """
        1. **Data understanding:** Inspect approximately three
           million flight records from January 2019 through
           August 2023.
        2. **Data cleaning:** Exclude cancelled and diverted
           flights and create the target
           `IS_DELAYED = ARR_DELAY >= 15`.
        3. **Feature engineering:** Create calendar, seasonal,
           time, route, and distance-related variables.
        4. **Temporal validation:** Train on 2019–2021, validate
           on 2022, and reserve January–August 2023 as the final
           holdout.
        5. **Model development:** Compare Logistic Regression,
           Random Forest, and XGBoost.
        6. **Model improvement:** Add actual departure time and
           departure delay for an after-departure scenario.
        7. **Deployment:** Save the final pipeline and deploy it
           through Streamlit.
        """
    )

    st.warning(
        """
        The deployed model predicts immediately after departure.
        It requires actual departure time and departure delay, so
        it must not be described as a pre-departure model.
        """
    )


# ---------------------------------------------------------
# Dynamic analysis page
# ---------------------------------------------------------

def render_univariate_analysis(dataframe: pd.DataFrame) -> None:
    st.subheader("Univariate Analysis")

    variable_type = st.radio(
        "Variable type",
        options=["Numerical", "Categorical"],
        horizontal=True,
        key="univariate_type",
    )

    if variable_type == "Numerical":
        selected_label = st.selectbox(
            "Numerical variable",
            options=list(NUMERICAL_ANALYSIS_COLUMNS),
            key="univariate_numeric_variable",
        )

        selected_column = NUMERICAL_ANALYSIS_COLUMNS[
            selected_label
        ]

        chart_type = st.selectbox(
            "Chart type",
            options=[
                "Histogram",
                "Box Plot",
                "Violin Plot",
            ],
            key="univariate_numeric_chart",
        )

        chart_data = dataframe[
            [selected_column]
        ].dropna()

        if chart_type == "Histogram":
            number_of_bins = st.slider(
                "Number of bins",
                min_value=10,
                max_value=100,
                value=40,
                key="univariate_bins",
            )

            figure = px.histogram(
                chart_data,
                x=selected_column,
                nbins=number_of_bins,
                marginal="box",
                title=f"Distribution of {selected_label}",
                labels={
                    selected_column: selected_label,
                },
            )

        elif chart_type == "Box Plot":
            figure = px.box(
                chart_data,
                x=selected_column,
                points="outliers",
                title=f"Box Plot of {selected_label}",
                labels={
                    selected_column: selected_label,
                },
            )

        else:
            figure = px.violin(
                chart_data,
                y=selected_column,
                box=True,
                points=False,
                title=f"Violin Plot of {selected_label}",
                labels={
                    selected_column: selected_label,
                },
            )

        show_plotly_chart(
            figure,
            "univariate_numeric_figure",
        )

        statistics = (
            dataframe[selected_column]
            .describe(
                percentiles=[0.25, 0.5, 0.75]
            )
            .rename(
                {
                    "count": "Count",
                    "mean": "Mean",
                    "std": "Standard Deviation",
                    "min": "Minimum",
                    "25%": "25th Percentile",
                    "50%": "Median",
                    "75%": "75th Percentile",
                    "max": "Maximum",
                }
            )
            .to_frame("Value")
        )

        st.dataframe(
            statistics,
            width="stretch",
        )

        st.caption(
            f"Missing values: "
            f"{dataframe[selected_column].isna().sum():,}"
        )

    else:
        selected_label = st.selectbox(
            "Categorical variable",
            options=list(CATEGORICAL_ANALYSIS_COLUMNS),
            key="univariate_categorical_variable",
        )

        selected_column = CATEGORICAL_ANALYSIS_COLUMNS[
            selected_label
        ]

        chart_type = st.selectbox(
            "Chart type",
            options=[
                "Count Bar Chart",
                "Percentage Bar Chart",
                "Pie Chart",
            ],
            key="univariate_categorical_chart",
        )

        maximum_categories = st.slider(
            "Maximum categories",
            min_value=5,
            max_value=30,
            value=15,
            key="univariate_category_limit",
        )

        frequency_table = (
            dataframe[selected_column]
            .astype("object")
            .fillna("Missing")
            .value_counts()
            .head(maximum_categories)
            .rename_axis(selected_label)
            .reset_index(name="Flights")
        )

        frequency_table["Percentage"] = (
            frequency_table["Flights"]
            / len(dataframe)
            * 100
        )

        if chart_type == "Count Bar Chart":
            figure = px.bar(
                frequency_table,
                x=selected_label,
                y="Flights",
                title=f"{selected_label} Frequency",
            )

        elif chart_type == "Percentage Bar Chart":
            figure = px.bar(
                frequency_table,
                x=selected_label,
                y="Percentage",
                title=f"{selected_label} Percentage Distribution",
                labels={
                    "Percentage": "Percentage (%)",
                },
            )

        else:
            figure = px.pie(
                frequency_table,
                names=selected_label,
                values="Flights",
                hole=0.4,
                title=f"{selected_label} Distribution",
            )

        show_plotly_chart(
            figure,
            "univariate_categorical_figure",
        )

        st.dataframe(
            frequency_table,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            f"Unique values: "
            f"{dataframe[selected_column].nunique(dropna=True):,}"
        )


def render_bivariate_analysis(dataframe: pd.DataFrame) -> None:
    st.subheader("Bivariate Analysis")

    all_variables = {
        **NUMERICAL_ANALYSIS_COLUMNS,
        **CATEGORICAL_ANALYSIS_COLUMNS,
    }

    variable_labels = list(all_variables)

    first_label = st.selectbox(
        "First variable",
        options=variable_labels,
        index=variable_labels.index(
            "Flight Distance"
        ),
        key="bivariate_first_variable",
    )

    second_label = st.selectbox(
        "Second variable",
        options=variable_labels,
        index=variable_labels.index(
            "Departure Delay"
        ),
        key="bivariate_second_variable",
    )

    first_column = all_variables[first_label]
    second_column = all_variables[second_label]

    if first_column == second_column:
        st.warning(
            "Select two different variables for bivariate analysis."
        )
        return

    first_is_numeric = (
        first_label in NUMERICAL_ANALYSIS_COLUMNS
    )

    second_is_numeric = (
        second_label in NUMERICAL_ANALYSIS_COLUMNS
    )

    if first_is_numeric and second_is_numeric:
        chart_type = st.selectbox(
            "Chart type",
            options=[
                "Scatter Plot",
                "Density Heatmap",
            ],
            key="bivariate_numeric_numeric_chart",
        )

        plot_data = dataframe[
            [first_column, second_column]
        ].dropna()

        if len(plot_data) > 20_000:
            plot_data = plot_data.sample(
                n=20_000,
                random_state=42,
            )

        correlation = dataframe[
            [first_column, second_column]
        ].corr().iloc[0, 1]

        st.metric(
            "Pearson Correlation",
            f"{correlation:.3f}",
        )

        if chart_type == "Scatter Plot":
            figure = px.scatter(
                plot_data,
                x=first_column,
                y=second_column,
                opacity=0.45,
                title=(
                    f"{second_label} versus {first_label}"
                ),
                labels={
                    first_column: first_label,
                    second_column: second_label,
                },
            )

        else:
            figure = px.density_heatmap(
                plot_data,
                x=first_column,
                y=second_column,
                nbinsx=40,
                nbinsy=40,
                title=(
                    f"Density of {second_label} "
                    f"versus {first_label}"
                ),
                labels={
                    first_column: first_label,
                    second_column: second_label,
                },
            )

        show_plotly_chart(
            figure,
            "bivariate_numeric_numeric_figure",
        )

    elif first_is_numeric != second_is_numeric:
        if first_is_numeric:
            numeric_label = first_label
            numeric_column = first_column
            categorical_label = second_label
            categorical_column = second_column

        else:
            numeric_label = second_label
            numeric_column = second_column
            categorical_label = first_label
            categorical_column = first_column

        chart_type = st.selectbox(
            "Chart type",
            options=[
                "Box Plot",
                "Violin Plot",
                "Mean Bar Chart",
            ],
            key="bivariate_categorical_numeric_chart",
        )

        top_categories = (
            dataframe[categorical_column]
            .astype("object")
            .value_counts()
            .head(15)
            .index
        )

        plot_data = dataframe.loc[
            dataframe[categorical_column]
            .astype("object")
            .isin(top_categories),
            [categorical_column, numeric_column],
        ].dropna()

        if chart_type == "Box Plot":
            figure = px.box(
                plot_data,
                x=categorical_column,
                y=numeric_column,
                points=False,
                title=(
                    f"{numeric_label} by "
                    f"{categorical_label}"
                ),
                labels={
                    categorical_column: categorical_label,
                    numeric_column: numeric_label,
                },
            )

        elif chart_type == "Violin Plot":
            figure = px.violin(
                plot_data,
                x=categorical_column,
                y=numeric_column,
                box=True,
                points=False,
                title=(
                    f"{numeric_label} Distribution by "
                    f"{categorical_label}"
                ),
                labels={
                    categorical_column: categorical_label,
                    numeric_column: numeric_label,
                },
            )

        else:
            mean_table = (
                plot_data.groupby(
                    categorical_column,
                    observed=True,
                )[numeric_column]
                .mean()
                .sort_values(ascending=False)
                .reset_index(name="Mean")
            )

            figure = px.bar(
                mean_table,
                x=categorical_column,
                y="Mean",
                title=(
                    f"Mean {numeric_label} by "
                    f"{categorical_label}"
                ),
                labels={
                    categorical_column: categorical_label,
                    "Mean": f"Mean {numeric_label}",
                },
            )

        show_plotly_chart(
            figure,
            "bivariate_categorical_numeric_figure",
        )

    else:
        chart_type = st.selectbox(
            "Chart type",
            options=[
                "Stacked Bar Chart",
                "Heatmap",
            ],
            key="bivariate_categorical_categorical_chart",
        )

        first_top_categories = (
            dataframe[first_column]
            .astype("object")
            .value_counts()
            .head(12)
            .index
        )

        second_top_categories = (
            dataframe[second_column]
            .astype("object")
            .value_counts()
            .head(12)
            .index
        )

        plot_data = dataframe.loc[
            dataframe[first_column]
            .astype("object")
            .isin(first_top_categories)
            & dataframe[second_column]
            .astype("object")
            .isin(second_top_categories),
            [first_column, second_column],
        ].copy()

        contingency_table = pd.crosstab(
            plot_data[first_column],
            plot_data[second_column],
        )

        if chart_type == "Stacked Bar Chart":
            stacked_table = (
                contingency_table
                .reset_index()
                .melt(
                    id_vars=first_column,
                    var_name=second_column,
                    value_name="Flights",
                )
            )

            figure = px.bar(
                stacked_table,
                x=first_column,
                y="Flights",
                color=second_column,
                barmode="stack",
                title=(
                    f"{second_label} Distribution within "
                    f"{first_label}"
                ),
                labels={
                    first_column: first_label,
                    second_column: second_label,
                },
            )

        else:
            figure = px.imshow(
                contingency_table,
                aspect="auto",
                text_auto=True,
                title=(
                    f"{first_label} versus {second_label}"
                ),
                labels={
                    "x": second_label,
                    "y": first_label,
                    "color": "Flights",
                },
            )

        show_plotly_chart(
            figure,
            "bivariate_categorical_categorical_figure",
        )


def render_time_analysis(dataframe: pd.DataFrame) -> None:
    st.subheader("Time-Based Analysis")

    granularity = st.selectbox(
        "Time dimension",
        options=[
            "Year",
            "Month",
            "Day of Week",
            "Departure Hour",
        ],
        key="time_granularity",
    )

    if granularity == "Year":
        group_column = "YEAR"
        x_label = "Year"

    elif granularity == "Month":
        group_column = "MONTH_NAME"
        x_label = "Month"

    elif granularity == "Day of Week":
        group_column = "DAY_OF_WEEK"
        x_label = "Day of Week"

    else:
        group_column = "DEP_HOUR"
        x_label = "Departure Hour"

    time_table = create_delay_rate_table(
        dataframe,
        group_column,
    )

    if granularity in [
        "Year",
        "Departure Hour",
    ]:
        time_table = time_table.sort_values(
            group_column
        )

    left_column, right_column = st.columns(2)

    with left_column:
        delay_rate_figure = px.line(
            time_table,
            x=group_column,
            y="Delay_Rate_Percent",
            markers=True,
            title=f"Delay Rate by {x_label}",
            labels={
                group_column: x_label,
                "Delay_Rate_Percent": "Delay Rate (%)",
            },
        )

        show_plotly_chart(
            delay_rate_figure,
            "time_delay_rate_figure",
        )

    with right_column:
        flight_count_figure = px.bar(
            time_table,
            x=group_column,
            y="Flights",
            title=f"Flight Count by {x_label}",
            labels={
                group_column: x_label,
            },
        )

        show_plotly_chart(
            flight_count_figure,
            "time_flight_count_figure",
        )

    st.dataframe(
        time_table[
            [
                group_column,
                "Flights",
                "Delayed_Flights",
                "Delay_Rate_Percent",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def render_airline_route_analysis(
    dataframe: pd.DataFrame,
) -> None:
    st.subheader("Airline, Airport, and Route Analysis")

    analysis_dimension = st.selectbox(
        "Comparison dimension",
        options=[
            "Airline",
            "Origin Airport",
            "Destination Airport",
            "Route",
        ],
        key="ranking_dimension",
    )

    group_mapping = {
        "Airline": "AIRLINE",
        "Origin Airport": "ORIGIN",
        "Destination Airport": "DEST",
        "Route": "ROUTE",
    }

    group_column = group_mapping[
        analysis_dimension
    ]

    first_column, second_column, third_column = (
        st.columns(3)
    )

    with first_column:
        ranking_metric = st.selectbox(
            "Ranking metric",
            options=[
                "Delay Rate",
                "Delayed Flights",
                "Total Flights",
            ],
            key="ranking_metric",
        )

    with second_column:
        top_n = st.slider(
            "Number of categories",
            min_value=5,
            max_value=30,
            value=15,
            key="ranking_top_n",
        )

    with third_column:
        maximum_rows = max(1, len(dataframe))
        default_minimum = min(100, maximum_rows)

        minimum_flights = st.number_input(
            "Minimum flights",
            min_value=1,
            max_value=maximum_rows,
            value=default_minimum,
            step=25,
            key="ranking_minimum_flights",
        )

    ranking_table = create_delay_rate_table(
        dataframe,
        group_column,
        int(minimum_flights),
    )

    metric_mapping = {
        "Delay Rate": "Delay_Rate_Percent",
        "Delayed Flights": "Delayed_Flights",
        "Total Flights": "Flights",
    }

    ranking_column = metric_mapping[
        ranking_metric
    ]

    ranking_table = (
        ranking_table
        .sort_values(
            ranking_column,
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            ranking_column,
            ascending=True,
        )
    )

    if ranking_table.empty:
        st.warning(
            "No categories meet the selected minimum-flight "
            "requirement."
        )
        return

    ranking_figure = px.bar(
        ranking_table,
        x=ranking_column,
        y=group_column,
        orientation="h",
        title=(
            f"Top {len(ranking_table)} "
            f"{analysis_dimension} Categories by "
            f"{ranking_metric}"
        ),
        labels={
            group_column: analysis_dimension,
            ranking_column: ranking_metric,
        },
    )

    show_plotly_chart(
        ranking_figure,
        "airline_route_ranking_figure",
    )

    st.dataframe(
        ranking_table[
            [
                group_column,
                "Flights",
                "Delayed_Flights",
                "Delay_Rate_Percent",
            ]
        ].sort_values(
            ranking_column,
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
    )


def render_filtered_data(dataframe: pd.DataFrame) -> None:
    st.subheader("Filtered Data")

    st.write(
        f"Displaying up to 5,000 of "
        f"{len(dataframe):,} filtered rows."
    )

    st.dataframe(
        dataframe.head(5_000),
        width="stretch",
        hide_index=True,
    )

    csv_data = convert_dataframe_to_csv(
        dataframe
    )

    st.download_button(
        "Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_flight_analysis.csv",
        mime="text/csv",
        width="stretch",
    )


def render_dynamic_analysis_page() -> None:
    st.title("📊 Dynamic Flight Analysis")

    st.caption(
        """
        Interactive analysis uses a representative 150,000-row
        sample that preserves the complete dataset's year and
        delay-class distribution.
        """
    )

    try:
        analysis_data = load_analysis_data()

    except Exception as error:
        st.error(
            "The analysis dataset could not be loaded."
        )
        st.exception(error)
        return

    filtered_data = filter_analysis_data(
        analysis_data
    )

    if filtered_data.empty:
        st.warning(
            "No flights match the selected filters. "
            "Change or clear one or more filters."
        )
        return

    first_metric, second_metric, third_metric, fourth_metric = (
        st.columns(4)
    )

    first_metric.metric(
        "Filtered Flights",
        f"{len(filtered_data):,}",
    )

    second_metric.metric(
        "Delay Rate",
        f"{filtered_data['IS_DELAYED'].mean():.2%}",
    )

    third_metric.metric(
        "Average Departure Delay",
        f"{filtered_data['DEP_DELAY'].mean():.1f} min",
    )

    fourth_metric.metric(
        "Median Distance",
        f"{filtered_data['DISTANCE'].median():,.0f} mi",
    )

    (
        univariate_tab,
        bivariate_tab,
        time_tab,
        ranking_tab,
        data_tab,
    ) = st.tabs(
        [
            "Univariate Analysis",
            "Bivariate Analysis",
            "Time Analysis",
            "Airlines and Routes",
            "Filtered Data",
        ]
    )

    with univariate_tab:
        render_univariate_analysis(
            filtered_data
        )

    with bivariate_tab:
        render_bivariate_analysis(
            filtered_data
        )

    with time_tab:
        render_time_analysis(
            filtered_data
        )

    with ranking_tab:
        render_airline_route_analysis(
            filtered_data
        )

    with data_tab:
        render_filtered_data(
            filtered_data
        )


# ---------------------------------------------------------
# Prediction page
# ---------------------------------------------------------

def render_prediction_page() -> None:
    airlines = metadata["airlines"]
    origins = metadata["origins"]
    destinations = metadata["destinations"]

    expected_features = list(
        metadata["feature_names"]
    )

    decision_threshold = float(
        metadata.get(
            "decision_threshold",
            0.5,
        )
    )

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

    test_metrics = metadata["test_metrics"]

    if not airlines or not origins or not destinations:
        st.error(
            "The deployment metadata does not contain valid "
            "airline and airport options."
        )
        return

    st.title("🤖 Flight Delay Prediction")

    st.write(
        """
        Predict whether a completed flight will arrive
        **15 minutes or more late**.
        """
    )

    st.warning(
        """
        This is an **after-departure prediction model** because
        it uses actual departure time and departure delay.
        """
    )

    with st.expander(
        "Final Model Performance",
        expanded=False,
    ):
        metric_columns = st.columns(5)

        metric_columns[0].metric(
            "Accuracy",
            f"{test_metrics.get('Accuracy', 0):.2%}",
        )

        metric_columns[1].metric(
            "Precision",
            f"{test_metrics.get('Precision', 0):.2%}",
        )

        metric_columns[2].metric(
            "Recall",
            f"{test_metrics.get('Recall', 0):.2%}",
        )

        metric_columns[3].metric(
            "F1-Score",
            f"{test_metrics.get('F1', 0):.2%}",
        )

        metric_columns[4].metric(
            "ROC-AUC",
            f"{test_metrics.get('ROC_AUC', 0):.2%}",
        )

    default_origin = origins[0]

    destination_index = next(
        (
            index
            for index, destination_value
            in enumerate(destinations)
            if destination_value != default_origin
        ),
        0,
    )

    with st.form(
        "flight_prediction_form"
    ):
        st.subheader("Flight Details")

        first_column, second_column, third_column = (
            st.columns(3)
        )

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
                index=destination_index,
            )

        with second_column:
            flight_date = st.date_input(
                "Flight Date",
                value=date(2023, 6, 15),
            )

            scheduled_departure_time = (
                st.time_input(
                    "Scheduled Departure Time",
                    value=time(10, 0),
                )
            )

            actual_departure_time = (
                st.time_input(
                    "Actual Departure Time",
                    value=time(10, 0),
                )
            )

        with third_column:
            scheduled_arrival_time = (
                st.time_input(
                    "Scheduled Arrival Time",
                    value=time(12, 0),
                )
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

            scheduled_elapsed_time = (
                st.number_input(
                    "Scheduled Flight Duration — Minutes",
                    min_value=1.0,
                    max_value=1500.0,
                    value=default_elapsed_time,
                    step=1.0,
                )
            )

            distance = st.number_input(
                "Flight Distance — Miles",
                min_value=1.0,
                max_value=10000.0,
                value=default_distance,
                step=10.0,
            )

        st.caption(
            """
            Time of day, season, distance category, weekend,
            peak season, and busy-hour features are calculated
            automatically using the same rules used during
            model training.
            """
        )

        submitted = st.form_submit_button(
            "Predict Flight Delay",
            width="stretch",
        )

    if not submitted:
        return

    if origin == destination:
        st.error(
            "Origin and destination airports must be different."
        )
        return

    year = flight_date.year
    month = flight_date.month
    day = flight_date.day

    day_of_week = flight_date.strftime("%A")
    quarter = ((month - 1) // 3) + 1

    scheduled_departure_hhmm = time_to_hhmm(
        scheduled_departure_time
    )

    actual_departure_hhmm = time_to_hhmm(
        actual_departure_time
    )

    scheduled_arrival_hhmm = time_to_hhmm(
        scheduled_arrival_time
    )

    departure_hour = (
        scheduled_departure_time.hour
    )

    time_of_day = determine_time_of_day(
        departure_hour
    )

    season = determine_season(month)

    distance_category = (
        determine_distance_category(distance)
    )

    is_weekend = int(
        day_of_week in [
            "Saturday",
            "Sunday",
        ]
    )

    is_peak_season = determine_peak_season(
        month
    )

    is_busy_hour = determine_busy_hour(
        departure_hour
    )

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
                "IS_PEAK_SEASON": is_peak_season,
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

        return

    input_data = input_data.loc[
        :,
        expected_features,
    ]

    try:
        model_score = float(
            model.predict_proba(
                input_data
            )[0, 1]
        )

        prediction = int(
            model_score
            >= decision_threshold
        )

        st.divider()
        st.subheader("Prediction Result")

        result_column, score_column = (
            st.columns(2)
        )

        with result_column:
            st.metric(
                "Predicted Status",
                (
                    "Delayed"
                    if prediction == 1
                    else "On Time"
                ),
            )

        with score_column:
            st.metric(
                "Delayed-Class Model Score",
                f"{model_score:.1%}",
            )

        st.progress(model_score)

        st.caption(
            """
            This is an uncalibrated model score, not a
            guaranteed real-world probability.
            """
        )

        if prediction == 1:
            st.error(
                """
                The model predicts that this flight is likely
                to arrive 15 minutes or more late.
                """
            )

        else:
            st.success(
                """
                The model predicts that this flight is likely
                to arrive less than 15 minutes late.
                """
            )

        with st.expander(
            "View Generated Model Input"
        ):
            st.dataframe(
                input_data,
                width="stretch",
            )

        if year < 2019 or year > 2023:
            st.info(
                """
                This date is outside the period used for model
                development and evaluation. Interpret the
                prediction cautiously.
                """
            )

    except Exception as error:
        st.error(
            "The prediction could not be generated."
        )
        st.exception(error)


# ---------------------------------------------------------
# Main navigation
# ---------------------------------------------------------

st.sidebar.title("✈️ Flight Analytics")

selected_page = st.sidebar.radio(
    "Navigation",
    options=[
        "Project Overview",
        "Dynamic Analysis",
        "Flight Prediction",
    ],
)

st.sidebar.caption(
    """
    Dataset period: January 2019–August 2023
    """
)

if selected_page == "Project Overview":
    render_overview_page()

elif selected_page == "Dynamic Analysis":
    render_dynamic_analysis_page()

else:
    render_prediction_page()
