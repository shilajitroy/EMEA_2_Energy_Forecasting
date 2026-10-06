"""Streamlit dashboard for Smart Home Energy Forecasting."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed_energy.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "energy_lstm.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

LOOKBACK = 96
FORECAST_MINUTES = (15, 30, 45, 60)
INTERVAL_HOURS = 0.25

st.set_page_config(
    page_title="Smart Home Energy Forecasting",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e7ebf2;
        padding: 16px 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(25, 45, 80, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_processed_data(file_path: str, modified_time: float) -> pd.DataFrame:
    """Load and check the saved 15-minute household power readings."""
    del modified_time  # File modification time is part of Streamlit's cache key.
    data = pd.read_csv(file_path, parse_dates=["Datetime"])
    required_columns = {"Datetime", "Global_active_power"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Processed data is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = data[["Datetime", "Global_active_power"]].copy()
    data["Global_active_power"] = pd.to_numeric(
        data["Global_active_power"], errors="coerce"
    )
    data = data.dropna(subset=["Datetime", "Global_active_power"])
    data = data.sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)
    if len(data) < LOOKBACK:
        raise ValueError(
            f"The dashboard needs at least {LOOKBACK} readings; found {len(data)}."
        )
    if not np.isfinite(data["Global_active_power"].to_numpy(dtype=float)).all():
        raise ValueError("Processed power readings contain non-finite values.")
    if not data["Datetime"].diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError(
            "The processed dataset must contain continuous 15-minute timestamps."
        )
    return data


@st.cache_resource
def load_model_and_scaler(model_modified: float, scaler_modified: float):
    """Load saved artifacts once per app process; this function never trains."""
    del model_modified, scaler_modified  # Modification times invalidate the cache.
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


st.title("🏠 Smart Home Energy Forecasting")
st.caption(
    "Household electricity history and next-hour forecasts from the trained LSTM."
)

if not DATA_PATH.is_file():
    st.error(f"Processed dataset not found: `{DATA_PATH}`")
    st.info(
        "Run the preprocessing section in `energy_forecasting.ipynb` to create "
        "`data/processed_energy.csv`."
    )
    st.stop()

if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
    st.error("The trained model or fitted scaler is missing.")
    st.info(
        "Run the training and save sections in `energy_forecasting.ipynb` first. "
        "The Streamlit application does not train or create these files."
    )
    st.stop()

try:
    energy_data = load_processed_data(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime,
    )
except Exception as error:
    st.error(f"Could not load the processed energy data: {error}")
    st.stop()

try:
    model, scaler = load_model_and_scaler(
        MODEL_PATH.stat().st_mtime,
        SCALER_PATH.stat().st_mtime,
    )
except Exception as error:
    st.error(f"Could not load the saved LSTM model or scaler: {error}")
    st.info(
        "Check that the model was saved as `models/energy_lstm.keras` and the "
        "matching scaler as `models/scaler.pkl`."
    )
    st.stop()

latest = energy_data.iloc[-1]
latest_timestamp = latest["Datetime"]
metrics_date = latest_timestamp.date()
latest_day_data = energy_data[energy_data["Datetime"].dt.date == metrics_date]

current_power_kw = float(latest["Global_active_power"])
daily_energy_kwh = float(
    latest_day_data["Global_active_power"].sum() * INTERVAL_HOURS
)
daily_peak_kw = float(latest_day_data["Global_active_power"].max())
daily_average_kw = float(latest_day_data["Global_active_power"].mean())

st.header("Current Energy Status")
if metrics_date != pd.Timestamp.now().date():
    st.info(
        f"The dataset ends at {latest_timestamp:%Y-%m-%d %H:%M}. "
        f"Daily metrics use the latest date available in the dataset "
        f"({metrics_date:%Y-%m-%d}); these are historical, not live readings."
    )

status_columns = st.columns(4)
status_columns[0].metric("Current Consumption", f"{current_power_kw:.2f} kW")
status_columns[1].metric("Today's Consumption", f"{daily_energy_kwh:.2f} kWh")
status_columns[2].metric("Today's Peak Consumption", f"{daily_peak_kw:.2f} kW")
status_columns[3].metric("Today's Average Consumption", f"{daily_average_kw:.2f} kW")
st.caption(
    "Daily energy is estimated as the sum of each 15-minute mean power reading "
    "multiplied by 0.25 hours (kW × hours = kWh). Power values are in kW."
)

# The loaded artifacts produce the forecast; no placeholder values are used.
latest_readings = energy_data["Global_active_power"].tail(LOOKBACK).to_numpy(dtype=float)
try:
    scaled_readings = scaler.transform(latest_readings.reshape(-1, 1))
    model_input = scaled_readings.reshape(1, LOOKBACK, 1)
    scaled_forecast = np.asarray(model.predict(model_input, verbose=0))
    if scaled_forecast.size != len(FORECAST_MINUTES):
        raise ValueError(
            "The loaded model should return four values; "
            f"it returned shape {scaled_forecast.shape}."
        )
    predictions_kw = scaler.inverse_transform(
        scaled_forecast.reshape(-1, 1)
    ).ravel()
    if not np.isfinite(predictions_kw).all():
        raise ValueError("The loaded model returned non-finite predictions.")
except Exception as error:
    st.error(f"Could not generate a forecast: {error}")
    st.stop()

forecast_timestamps = [
    latest_timestamp + pd.Timedelta(minutes=minutes)
    for minutes in FORECAST_MINUTES
]

st.header("LSTM Forecast")
forecast_columns = st.columns(4)
for column, minutes, prediction in zip(
    forecast_columns, FORECAST_MINUTES, predictions_kw
):
    column.metric(f"Next {minutes} Minutes", f"{prediction:.2f} kW")
st.caption("Forecast values are predictions from the trained LSTM, not measurements.")

recent_data = energy_data.tail(7 * 24 * 4)
st.header("Historical Consumption")
historical_chart = recent_data.set_index("Datetime")[["Global_active_power"]]
historical_chart = historical_chart.rename(
    columns={"Global_active_power": "Observed power (kW)"}
)
st.line_chart(historical_chart)

st.header("Future Forecast")
forecast_chart = recent_data.set_index("Datetime")[["Global_active_power"]].rename(
    columns={"Global_active_power": "Observed power (kW)"}
)
forecast_series = pd.Series(
    predictions_kw,
    index=pd.DatetimeIndex(forecast_timestamps, name="Datetime"),
    name="Predicted power (kW)",
)
forecast_chart = pd.concat([forecast_chart, forecast_series], axis=1)
st.line_chart(forecast_chart)
st.caption(
    "Predictions are timestamped 15, 30, 45, and 60 minutes after the latest "
    f"available reading ({latest_timestamp:%Y-%m-%d %H:%M})."
)

st.header("High-Consumption Warning")
max_observed_kw = float(energy_data["Global_active_power"].max())
slider_max_kw = max(1.0, float(np.ceil(max_observed_kw * 2) / 2))
default_threshold_kw = min(
    max(float(energy_data["Global_active_power"].quantile(0.90)), 0.1),
    slider_max_kw,
)
threshold_kw = st.slider(
    "High-consumption threshold",
    min_value=0.0,
    max_value=slider_max_kw,
    value=default_threshold_kw,
    step=0.1,
    format="%.1f kW",
)

st.header("Energy-Saving Recommendation")
peak_forecast_kw = float(np.max(predictions_kw))
if peak_forecast_kw > threshold_kw:
    st.warning("⚠️ High Energy Consumption Expected")
    st.write("Consider postponing high-power appliances until consumption is lower.")
elif peak_forecast_kw >= threshold_kw * 0.8:
    st.info(
        "Moderate consumption is expected. Avoid running multiple high-power "
        "appliances at the same time."
    )
else:
    st.success("Consumption is expected to remain moderate.")
    st.write("Continue normal use and switch off devices that are not needed.")

st.caption("Forecasts and recommendations are estimates based on historical data.")
