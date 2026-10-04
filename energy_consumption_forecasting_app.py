import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Consumption Forecaster", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.main-title {font-size:40px;font-weight:bold;text-align:center;margin-bottom:10px;}
.subtitle {text-align:center;color:gray;font-size:18px;margin-bottom:30px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ Energy Consumption Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Time Series Analysis and Future Energy Prediction</div>', unsafe_allow_html=True)

st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Energy Consumption CSV", type=["csv"])

if "data" not in st.session_state:
    st.session_state.data = None
if "models" not in st.session_state:
    st.session_state.models = {}
if "results" not in st.session_state:
    st.session_state.results = {}
if "best_model" not in st.session_state:
    st.session_state.best_model = None

if uploaded_file is not None:
    try:
        st.session_state.data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read CSV file: {e}")

use_example = st.sidebar.checkbox("Use Example Data")

if use_example:
    dates = pd.date_range("2025-01-01", periods=24 * 30, freq="h")
    np.random.seed(42)
    hour = dates.hour
    consumption = (
        250
        + 40 * np.sin((hour - 6) * 2 * np.pi / 24)
        + 30 * np.exp(-((hour - 19) ** 2) / 10)
        + np.random.normal(0, 8, len(dates))
    )
    st.session_state.data = pd.DataFrame({
        "timestamp": dates,
        "consumption": consumption
    })

if st.session_state.data is None:
    st.info("Upload a CSV file from the sidebar or select 'Use Example Data'.")
    st.markdown("""
### Required CSV format
Your CSV should contain:
- `timestamp`
- `consumption`

Example:
| timestamp | consumption |
|---|---:|
| 2025-01-01 00:00:00 | 210.51 |
| 2025-01-01 01:00:00 | 203.89 |
| 2025-01-01 02:00:00 | 211.71 |
""")
    st.stop()

df = st.session_state.data.copy()

st.sidebar.subheader("Column Mapping")
timestamp_column = st.sidebar.selectbox("Timestamp Column", df.columns)
consumption_column = st.sidebar.selectbox("Consumption Column", df.columns)

try:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
except Exception:
    st.error("The selected timestamp column cannot be converted to datetime.")
    st.stop()

df[consumption_column] = pd.to_numeric(df[consumption_column], errors="coerce")
df = df.dropna(subset=[timestamp_column, consumption_column])
df = df.sort_values(timestamp_column)
df = df.drop_duplicates(subset=[timestamp_column])
df = df.set_index(timestamp_column)
df = df.rename(columns={consumption_column: "consumption"})

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data", "Analysis", "Model Training", "Evaluation", "Forecast"
])

with tab1:
    st.header("Data Upload & Preprocessing")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Average Consumption", f"{df['consumption'].mean():.2f}")
    with col3:
        st.metric("Maximum Consumption", f"{df['consumption'].max():.2f}")

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum(), use_container_width=True)

    st.subheader("Resampling")
    frequency = st.selectbox(
        "Select Frequency",
        ["Original", "Hourly", "Daily", "Weekly", "Monthly"]
    )

    if frequency == "Hourly":
        processed_df = df.resample("h").mean()
    elif frequency == "Daily":
        processed_df = df.resample("D").mean()
    elif frequency == "Weekly":
        processed_df = df.resample("W").mean()
    elif frequency == "Monthly":
        processed_df = df.resample("ME").mean()
    else:
        processed_df = df.copy()

    processed_df["consumption"] = (
        processed_df["consumption"].interpolate().ffill().bfill()
    )
    st.session_state.processed_data = processed_df

    st.subheader("Processed Data")
    st.dataframe(processed_df.head(20), use_container_width=True)

    st.subheader("Energy Consumption")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=processed_df.index,
        y=processed_df["consumption"],
        mode="lines",
        name="Consumption"
    ))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Energy Consumption",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Processed Data",
        processed_df.to_csv(),
        "processed_energy_data.csv",
        "text/csv"
    )

with tab2:
    st.header("Time Series Analysis")

    if "processed_data" not in st.session_state:
        st.warning("Please process the data in the Data tab first.")
    else:
        data = st.session_state.processed_data
        series = data["consumption"].dropna()

        st.subheader("Trend Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name="Energy Consumption"
        ))
        fig.update_layout(
            title="Energy Consumption Trend",
            xaxis_title="Time",
            yaxis_title="Consumption"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Time Series Decomposition")
        period = st.number_input(
            "Seasonal Period",
            min_value=2,
            max_value=365,
            value=24,
            step=1
        )

        if len(series) >= period * 2:
            try:
                decomposition = seasonal_decompose(
                    series, model="additive", period=period
                )
                fig, ax = plt.subplots(figsize=(12, 8))
                decomposition.plot(ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Decomposition failed: {e}")
        else:
            st.warning("Not enough data for decomposition.")

        st.subheader("Autocorrelation")
        max_lags = min(100, max(5, len(series) // 3))
        default_lags = min(30, max_lags)
        lags = st.slider("Number of Lags", 5, max_lags, default_lags)

        fig, ax = plt.subplots()
        plot_acf(series, lags=lags, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Partial Autocorrelation")
        safe_lags = min(lags, max(1, len(series) // 2 - 1))
        fig, ax = plt.subplots()
        plot_pacf(series, lags=safe_lags, ax=ax, method="ywm")
        st.pyplot(fig)
        plt.close(fig)

with tab3:
    st.header("Model Training")

    if "processed_data" not in st.session_state:
        st.warning("Please process the data first.")
    else:
        data = st.session_state.processed_data
        series = data["consumption"].dropna()

        st.subheader("Train/Test Split")
        test_percentage = st.slider("Test Data Percentage", 10, 40, 20)
        split_index = int(len(series) * (1 - test_percentage / 100))
        train = series.iloc[:split_index]
        test = series.iloc[split_index:]

        st.write(f"Training records: {len(train)}")
        st.write(f"Testing records: {len(test)}")

        st.subheader("ARIMA")
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("ARIMA p", 0, 10, 1)
        with col2:
            d = st.number_input("ARIMA d", 0, 3, 1)
        with col3:
            q = st.number_input("ARIMA q", 0, 10, 1)

        st.subheader("SARIMA")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            P = st.number_input("Seasonal P", 0, 5, 1)
        with col2:
            D = st.number_input("Seasonal D", 0, 2, 1)
        with col3:
            Q = st.number_input("Seasonal Q", 0, 5, 1)
        with col4:
            seasonal_period = st.number_input("Seasonal Period m", 2, 365, 24)

        st.subheader("Exponential Smoothing")
        trend_type = st.selectbox("Trend", ["add", "mul", "None"])
        seasonal_type = st.selectbox("Seasonality", ["add", "mul", "None"])
        trend_type = None if trend_type == "None" else trend_type
        seasonal_type = None if seasonal_type == "None" else seasonal_type

        if st.button("Train All Models", type="primary"):
            results = {}
            models = {}

            with st.spinner("Training ARIMA..."):
                try:
                    fitted = ARIMA(train, order=(p, d, q)).fit()
                    predictions = fitted.forecast(steps=len(test))
                    results["ARIMA"] = {
                        "MAE": mean_absolute_error(test, predictions),
                        "RMSE": np.sqrt(mean_squared_error(test, predictions))
                    }
                    models["ARIMA"] = fitted
                    st.success("ARIMA training completed.")
                except Exception as e:
                    st.error(f"ARIMA failed: {e}")

            with st.spinner("Training SARIMA..."):
                try:
                    fitted = SARIMAX(
                        train,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(disp=False)
                    predictions = fitted.forecast(steps=len(test))
                    results["SARIMA"] = {
                        "MAE": mean_absolute_error(test, predictions),
                        "RMSE": np.sqrt(mean_squared_error(test, predictions))
                    }
                    models["SARIMA"] = fitted
                    st.success("SARIMA training completed.")
                except Exception as e:
                    st.error(f"SARIMA failed: {e}")

            with st.spinner("Training Exponential Smoothing..."):
                try:
                    fitted = ExponentialSmoothing(
                        train,
                        trend=trend_type,
                        seasonal=seasonal_type,
                        seasonal_periods=seasonal_period if seasonal_type else None
                    ).fit()
                    predictions = fitted.forecast(len(test))
                    results["Exponential Smoothing"] = {
                        "MAE": mean_absolute_error(test, predictions),
                        "RMSE": np.sqrt(mean_squared_error(test, predictions))
                    }
                    models["Exponential Smoothing"] = fitted
                    st.success("Exponential Smoothing training completed.")
                except Exception as e:
                    st.error(f"Exponential Smoothing failed: {e}")

            st.session_state.results = results
            st.session_state.models = models

            if results:
                st.session_state.best_model = min(
                    results, key=lambda x: results[x]["RMSE"]
                )
                st.success(f"Best Model: {st.session_state.best_model}")

with tab4:
    st.header("Model Evaluation")

    if not st.session_state.results:
        st.info("Train the models first.")
    else:
        result_df = pd.DataFrame(st.session_state.results).T
        st.subheader("Performance Metrics")
        st.dataframe(
            result_df.style.format("{:.4f}"),
            use_container_width=True
        )

        st.success(f"Best Model: {st.session_state.best_model}")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=result_df.index,
            y=result_df["RMSE"],
            name="RMSE"
        ))
        fig.update_layout(
            title="Model RMSE Comparison",
            xaxis_title="Model",
            yaxis_title="RMSE"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=result_df.index,
            y=result_df["MAE"],
            name="MAE"
        ))
        fig.update_layout(
            title="Model MAE Comparison",
            xaxis_title="Model",
            yaxis_title="MAE"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Future Energy Consumption Forecast")

    if not st.session_state.models:
        st.info("Please train the models first.")
    else:
        model_names = list(st.session_state.models.keys())
        best = st.session_state.best_model
        selected_model = st.selectbox(
            "Select Forecasting Model",
            model_names,
            index=model_names.index(best) if best in model_names else 0
        )

        forecast_periods = st.slider(
            "Forecast Horizon",
            1,
            168,
            24
        )

        if st.button("Generate Forecast", type="primary"):
            model = st.session_state.models[selected_model]

            with st.spinner("Generating forecast..."):
                try:
                    forecast = model.forecast(steps=forecast_periods)
                    data = st.session_state.processed_data
                    last_timestamp = data.index[-1]

                    if len(data.index) > 1:
                        time_difference = data.index[-1] - data.index[-2]
                    else:
                        time_difference = pd.Timedelta(hours=1)

                    future_dates = pd.date_range(
                        start=last_timestamp + time_difference,
                        periods=forecast_periods,
                        freq=time_difference
                    )

                    forecast_df = pd.DataFrame({
                        "timestamp": future_dates,
                        "forecast": forecast.values
                    })

                    st.subheader("Forecast Results")
                    st.dataframe(forecast_df, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Forecast", f"{forecast.mean():.2f}")
                    with col2:
                        st.metric("Maximum Forecast", f"{forecast.max():.2f}")
                    with col3:
                        st.metric("Minimum Forecast", f"{forecast.min():.2f}")

                    historical = data.tail(200)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=historical.index,
                        y=historical["consumption"],
                        mode="lines",
                        name="Historical"
                    ))
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=forecast.values,
                        mode="lines",
                        name="Forecast"
                    ))
                    fig.update_layout(
                        title=f"{selected_model} Energy Consumption Forecast",
                        xaxis_title="Time",
                        yaxis_title="Energy Consumption",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.download_button(
                        "Download Forecast CSV",
                        forecast_df.to_csv(index=False),
                        "energy_forecast.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"Forecast generation failed: {e}")

st.markdown("---")
st.caption("Energy Consumption Forecasting System | Python + Streamlit + Time Series Models")
