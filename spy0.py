import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Custom CSS to shrink the sidebar
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 200px;  /* Change this value to adjust the width */
            max-width: 200px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def get_latest_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        return hist["Close"].iloc[-1]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def main():
    st.title("MA5: Lin_PN Reg with Pred")
    st.sidebar.header("Input Data")
    st.sidebar.divider()

    if 'ticker' not in st.session_state:
        st.session_state.ticker = "SPY"
    if "custom_ticker" not in st.session_state:
        st.session_state.custom_ticker = ""
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False

    if st.sidebar.button(f"Switch to {'QQQ' if st.session_state.ticker == 'SPY' else 'SPY'}"):
        st.session_state.ticker = "QQQ" if st.session_state.ticker == "SPY" else "SPY"
        st.session_state.custom_ticker = ""
        st.session_state.clear_input = True

    def clear_custom_ticker():
        st.session_state.clear_input = True

    custom_ticker = st.sidebar.text_input(
        "Enter a custom ticker symbol (e.g., AAPL, TSLA):",
        value="" if st.session_state.clear_input else st.session_state.custom_ticker,
        key="custom_ticker_input"
    )

    if st.session_state.clear_input:
        st.session_state.custom_ticker = ""
        st.session_state.clear_input = False

    if st.sidebar.button("Clear Ticker"):
        clear_custom_ticker()

    if custom_ticker.strip():
        selected_ticker = custom_ticker.strip().upper()
        st.session_state.custom_ticker = custom_ticker
    else:
        selected_ticker = st.session_state.ticker

   # st.write(f"Selected Ticker: {selected_ticker}")

    st.sidebar.subheader("Historical Data Settings")
    time_period = st.sidebar.selectbox(
        "Select time period:",
        options=["3mo", "6mo", "1y", "2y"],
        index=1
    )

    try:
        stock = yf.Ticker(selected_ticker)
        hist = stock.history(period=time_period)
        hist0 = stock.history(period="5y")
    except Exception as e:
        st.error(f"Error fetching historical data for {selected_ticker}: {e}")
        st.stop()

    if not isinstance(hist.index, pd.DatetimeIndex):
        hist.index = pd.to_datetime(hist.index)

    if not isinstance(hist0.index, pd.DatetimeIndex):
        hist0.index = pd.to_datetime(hist0.index)

    closing_prices = hist["Close"].round(2)
    closing_prices0 = hist0["Close"].round(2)

    if len(closing_prices) == 0:
        st.error(f"No data found for {selected_ticker} for the selected time period. Please try again.")
        st.stop()

    data = closing_prices.tolist()
    data0 = closing_prices0.tolist()

    latest_price = get_latest_price(selected_ticker)

    if len(data) >= 2:
        last_price = data[-1]
        second_last_price = data[-2]
        price_difference = last_price - second_last_price
    else:
        price_difference = None

    st.sidebar.divider()
    st.sidebar.subheader("Prediction Settings")
    prediction_period = st.sidebar.selectbox(
        "Select prediction period:",
        options=["15 days", "30 days", "60 days", "90 days"],
        index=0
    )

    # Add a checkbox to toggle the predicted region
    show_predicted_region = st.sidebar.checkbox("Click to Show Predicted Region", value=False)

    st.sidebar.divider()

    window_size = 5
    if data is None or len(data) < window_size:
        st.error(f"Not enough data points to compute {window_size}-day moving averages. Please provide at least {window_size} data points.")
        st.stop()

    moving_avg = moving_average(data)

    ma9 =  np.convolve(data0, np.ones(9)/9, mode='valid')[-(len(data) - 4):]
    ma20 = np.convolve(data0, np.ones(20)/20, mode='valid')[-(len(data) - 4):]
    ma50 = np.convolve(data0, np.ones(50)/50, mode='valid')[-(len(data) - 4):]
    ma100 =  np.convolve(data0, np.ones(100)/100, mode='valid')[-(len(data) - 4):]
    ma200 = np.convolve(data0, np.ones(200)/200, mode='valid')[-(len(data) - 4):]

    X = np.arange(len(moving_avg)).reshape(-1, 1)
    y = moving_avg

    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_pred_linear = linear_model.predict(X)
    r_squared_linear = r2_score(y, y_pred_linear)

    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)
    y_pred_poly = poly_model.predict(X)
    r_squared_poly = r2_score(y, y_pred_poly)

    if r_squared_linear > r_squared_poly:
        better_model = linear_model
        better_y_pred = y_pred_linear
        better_r_squared = r_squared_linear
        model_type = "Linear"
    else:
        better_model = poly_model
        better_y_pred = y_pred_poly
        better_r_squared = r_squared_poly
        model_type = "Polynomial"

    residuals = y - better_y_pred
    std_dev = np.std(residuals)

    latest_closing_price = data[-1]
    last_model_value = better_y_pred[-1]

    std_dev_levels = [1.0, 1.5, 2.0]
    differences = []

    for level in std_dev_levels:
        lower_bound = last_model_value - level * std_dev
        upper_bound = last_model_value + level * std_dev
        differences.append(abs(latest_closing_price - lower_bound))
        differences.append(abs(latest_closing_price - upper_bound))

    min_diff = min(differences)
    min_index = differences.index(min_diff)
    std_dev_index = min_index // 2
    std_dev_level = std_dev_levels[std_dev_index]

    if min_index % 2 == 0:
        bound_type = "lower"
        bound_price = last_model_value - std_dev_level * std_dev
        color = "red"
    else:
        bound_type = "upper"
        bound_price = last_model_value + std_dev_level * std_dev
        color = "green"

    st.markdown(
        f'<p style="color: {color}; font-size: 18px; font-weight: bold;">{selected_ticker} ({latest_price:.2f}) ___near___ {std_dev_level} Std_Dev___{model_type} Reg (R² = {better_r_squared:.3f})</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p style="color: orange; font-size: 16px; font-weight: bold;">({time_period} period)</p>',
        unsafe_allow_html=True
    )
    hist = stock.history(period=time_period)
    st.write(f"last date in Data___{hist.index[-1].strftime("%Y-%m-%d")}___first date___{hist.index[0].strftime("%Y-%m-%d")}")
    
    st.subheader("Plot")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

    # Plot 1: Price and Moving Averages
    ax1.plot(np.arange(len(data) - 4), data[4:], color='black', label=f'{selected_ticker} Closing Prices', alpha=0.5)
    ax1.plot(np.arange(len(data) - 4 ), ma9, color='red', label=f'{selected_ticker} ma9', alpha=0.5)
    ax1.plot(np.arange(len(data) - 4), ma20, color='blue', label=f'{selected_ticker} ma20', alpha=0.5)
    ax1.plot(np.arange(len(data) - 4), ma50, color='orange', label=f'{selected_ticker} ma50', alpha=0.5)
    ax1.plot(np.arange(len(data) - 4), ma100, color='black', label=f'{selected_ticker} ma100', alpha=0.5)
    ax1.plot(np.arange(len(data) - 4), ma200, color='purple', label=f'{selected_ticker} ma200', alpha=0.5)
    ax1.scatter(X, y, color='blue', label='5-Day Moving Averages')
    ax1.plot(X, y_pred_linear, color='red', label=f'Linear Regression (R² = {r_squared_linear:.3f})')
    ax1.plot(X, y_pred_poly, color='green', label=f'Polynomial Regression (R² = {r_squared_poly:.3f})')

    # Add shaded regions for standard deviations (if enabled)
    if show_predicted_region:
        for level, color, alpha in zip([1.0, 1.5, 2.0], ['gray', 'orange', 'red'], [0.2, 0.1, 0.05]):
            ax1.fill_between(
                X.flatten(),
                better_y_pred - level * std_dev,
                better_y_pred + level * std_dev,
                color=color, alpha=alpha, label=f'±{level} Standard Deviations'
            )

    # Plot future predictions (if enabled)
    if show_predicted_region:
        prediction_days = int(prediction_period.split()[0])
        future_X = np.arange(len(moving_avg), len(moving_avg) + prediction_days).reshape(-1, 1)
        future_y_better = better_model.predict(future_X)
        ax1.scatter(future_X, future_y_better, color='purple', label=f'{model_type} Regression Predictions')

        # Annotate future predictions
        for i, (x, y_pred) in enumerate(zip(future_X, future_y_better)):
            ax1.text(x, y_pred, f'{y_pred:.2f}', color='purple', fontsize=8, ha='left', va='top')

    ax1.set_title(f"MA5: with {model_type} Regression (R² = {better_r_squared:.3f} ({time_period} period)")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid()

    # Plot 2: RSI
    rsi = calculate_rsi(hist)
    rsi2 = calculate_rsi(hist, window=25)
    ax2.plot(np.arange(len(rsi)), rsi, color='blue', label='RSI')
    ax2.plot(np.arange(len(rsi)), rsi2, color="orange", linestyle="--", label="RSI (25)")
    ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax2.axhline(50, color='gray', linestyle='--')  # Gray line at 50
    ax2.set_ylabel("RSI")
    ax2.legend()
    ax2.grid()

    # Plot 3: MACD
    hist = calculate_macd(hist)
    ax3.plot(np.arange(len(hist)), hist['MACD'], color='blue', label='MACD')
    ax3.plot(np.arange(len(hist)), hist['Signal_Line'], color='orange', linestyle="--", label='Signal Line')
    ax3.axhline(0, color='gray', linestyle='-')  # Gray line at 50
    ax3.set_ylabel("MACD")
    ax3.set_xlabel("Time Step")
    ax3.legend()
    ax3.grid()

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
