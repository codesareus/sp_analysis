import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Title of the app
st.title("5-Day Moving Averages: Linear vs Polynomial Regression with Future Predictions")

# Sidebar for user input
st.sidebar.header("Input Data")

# Divider
st.sidebar.divider()

# Add a button to toggle between SPY and QQQ
if 'ticker' not in st.session_state:
    st.session_state.ticker = "SPY"

if st.sidebar.button(f"Switch to {'QQQ' if st.session_state.ticker == 'SPY' else 'SPY'}"):
    st.session_state.ticker = "QQQ" if st.session_state.ticker == "SPY" else "SPY"

######################
# Initialize session state for custom ticker if it doesn't exist
if "custom_ticker" not in st.session_state:
    st.session_state.custom_ticker = ""

# Define a callback function to clear the custom ticker input
def clear_custom_ticker():
    st.session_state.custom_ticker = ""  # Clear the session state

# Add a text input for custom ticker symbol
custom_ticker = st.sidebar.text_input(
    "Enter a custom ticker symbol (e.g., AAPL, TSLA):",
    value=st.session_state.custom_ticker,
    key="custom_ticker_input"
)

# Add a button to clear the custom ticker input
#if st.sidebar.button("Clear Ticker"):
   # clear_custom_ticker()  # Call the function to clear the ticker

# Use the custom ticker if provided
if custom_ticker.strip():  # If the user entered something
    selected_ticker = custom_ticker.strip().upper()
    # Clear the custom ticker after using it
    st.session_state.custom_ticker = ""  # Clear the input
else:
    selected_ticker = st.session_state.ticker  # Use the toggled ticker

# Display the selected ticker
st.write(f"Selected Ticker: {selected_ticker}")

#################
# Fetch the latest closing price for the selected ticker
def get_latest_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")  # Fetch the latest day's data
        return hist["Close"].iloc[-1]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Initialize variables
latest_price = None
price_difference = None

# Fetch data based on the selected time period
st.sidebar.subheader("Historical Data Settings")
time_period = st.sidebar.selectbox(
    "Select time period:",
    options=["3mo", "6mo", "1y", "2y"],  # Removed "5y"
    index=1  # Default: 6 months
)

# Download historical data for the selected ticker
try:
    stock = yf.Ticker(selected_ticker)
    hist = stock.history(period=time_period)
    hist0 = stock.history(period="5y")
except Exception as e:
    st.error(f"Error fetching historical data for {selected_ticker}: {e}")
    st.stop()

# Ensure the index is a DatetimeIndex
if not isinstance(hist.index, pd.DatetimeIndex):
    hist.index = pd.to_datetime(hist.index)

if not isinstance(hist0.index, pd.DatetimeIndex):
    hist0.index = pd.to_datetime(hist0.index)

# Extract closing prices and round to two decimal places
closing_prices = hist["Close"].round(2)
closing_prices0 = hist0["Close"].round(2)

# Check if the dataset is empty
if len(closing_prices) == 0:
    st.error(f"No data found for {selected_ticker} for the selected time period. Please try again.")
    st.stop()  # Stop execution if the dataset is empty

# Use the closing prices as the dataset
data = closing_prices.tolist()
data0 = closing_prices0.tolist()

# Fetch the latest closing price
latest_price = get_latest_price(selected_ticker)

# Calculate the difference between the last two closing prices
if len(data) >= 2:
    last_price = data[-1]
    second_last_price = data[-2]
    price_difference = last_price - second_last_price
else:
    price_difference = None

# Divider
st.sidebar.divider()

# Add a selection box for the prediction period
st.sidebar.subheader("Prediction Settings")
prediction_period = st.sidebar.selectbox(
    "Select prediction period:",
    options=["15 days", "30 days", "60 days", "90 days"],
    index=0  # Default: 30 days
)

# Divider
st.sidebar.divider()

# Check if the dataset has enough data points for moving averages
window_size = 5
if data is None or len(data) < window_size:
    st.error(f"Not enough data points to compute {window_size}-day moving averages. Please provide at least {window_size} data points.")
    st.stop()  # Stop execution if the dataset is too small

# Function to calculate 5-point moving average
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

moving_avg = moving_average(data)

ma9 =  np.convolve(data0, np.ones(9)/9, mode='valid')[-(len(data) - 4):]
ma20 = np.convolve(data0, np.ones(20)/20, mode='valid')[-(len(data) - 4):]
ma50 = np.convolve(data0, np.ones(50)/50, mode='valid')[-(len(data) - 4):]
ma100 =  np.convolve(data0, np.ones(100)/100, mode='valid')[-(len(data) - 4):]
ma200 = np.convolve(data0, np.ones(200)/200, mode='valid')[-(len(data) - 4):]

# Create feature (X) and target (y) variables for moving averages
X = np.arange(len(moving_avg)).reshape(-1, 1)  # Time steps as features
y = moving_avg  # Moving averages as target

# Fit Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
r_squared_linear = r2_score(y, y_pred_linear)

# Fit Polynomial Regression model (degree=2 for quadratic)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X, y)
y_pred_poly = poly_model.predict(X)
r_squared_poly = r2_score(y, y_pred_poly)

# Determine which model has the higher R-squared
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

# Calculate residuals (errors) for the better model
residuals = y - better_y_pred
std_dev = np.std(residuals)  # Standard deviation of residuals

# Ensure latest_closing_price is defined
latest_closing_price = data[-1]  # Latest closing price

# Get the last modeled value from the better model
last_model_value = better_y_pred[-1]

# Calculate the differences for each standard deviation level
std_dev_levels = [1.0, 1.5, 2.0]
differences = []

for level in std_dev_levels:
    lower_bound = last_model_value - level * std_dev
    upper_bound = last_model_value + level * std_dev
    differences.append(abs(latest_closing_price - lower_bound))
    differences.append(abs(latest_closing_price - upper_bound))

# Find the smallest difference and corresponding standard deviation
min_diff = min(differences)
min_index = differences.index(min_diff)
std_dev_index = min_index // 2  # Determine which standard deviation level
std_dev_level = std_dev_levels[std_dev_index]

# Determine if the smallest difference is from the lower or upper bound
if min_index % 2 == 0:
    bound_type = "lower"
    bound_price = last_model_value - std_dev_level * std_dev
else:
    bound_type = "upper"
    bound_price = last_model_value + std_dev_level * std_dev

# Display the message indicating how many standard deviations the selected ticker is approaching
st.markdown(
    f'<p style="color: orange; font-size: 18px; font-weight: bold;">{selected_ticker} ({latest_price:.2f}) approaching {std_dev_level} Std_Dev ({bound_type} bound: {bound_price:.2f}) based on {model_type} Regression (R² = {better_r_squared:.3f})</p>',
    unsafe_allow_html=True
)

# Plot the data and regression lines
st.subheader("Plot")
fig, ax = plt.subplots(figsize=(12, 8))  # Wider plot to accommodate predictions

# Convert the selected prediction period to an integer
prediction_days = int(prediction_period.split()[0])  # Extract the number of days

# Predict future days
future_X = np.arange(len(moving_avg), len(moving_avg) + prediction_days).reshape(-1, 1)  # Future time steps
future_y_better = better_model.predict(future_X)  # Better model predictions

# Display current price and price difference above the plot
st.write(f"### {selected_ticker} Price Information")
if latest_price is not None and price_difference is not None:
    # Determine if the price increased or decreased
    if price_difference > 0:
        color = "green"
        change_symbol = "+"
    elif price_difference < 0:
        color = "red"
        change_symbol = "-"
    else:
        color = "black"  # No change
        change_symbol = ""
    
    # Get yesterday's closing price
    yesterday_close = latest_price - price_difference
    
    # Calculate the percentage change
    percentage_change = (price_difference / yesterday_close) * 100
    
    # Display the combined message with color applied to the entire message
    st.markdown(
        f'<span style="color: {color};">**{selected_ticker} Current Price:** ${latest_price:.2f}, {change_symbol}${abs(price_difference):.2f} ({percentage_change:.2f}%) from last close (${yesterday_close:.2f})</span>',
        unsafe_allow_html=True
    )
elif latest_price is not None:
    st.write(f"**{selected_ticker} Current Price:** ${latest_price:.2f} (Price difference data not available)")
else:
    st.write(f"**{selected_ticker} Current Price:** Not available")

# Add the comparison logic and display the trend message
latest_closing_price = data[-1]  # Latest closing price
latest_moving_avg = moving_avg[-1]  # Latest 5-day moving average
latest_ma9 = ma9[-1]  # Latest 9-day moving average
latest_ma20 = ma20[-1]  # Latest 20-day moving average
latest_ma50 = ma50[-1]  # Latest 50-day moving average
latest_ma100 = ma100[-1]  # Latest 50-day moving average

# Calculate the absolute differences between the selected ticker and moving averages
diff_moving_avg = abs(latest_closing_price - latest_moving_avg)
diff_ma9 = abs(latest_closing_price - latest_ma9)
diff_ma20 = abs(latest_closing_price - latest_ma20)
diff_ma50 = abs(latest_closing_price - latest_ma50)
diff_ma100 = abs(latest_closing_price - latest_ma100)

# Check for up_trend condition
if latest_closing_price  > latest_ma9 and latest_ma9 > latest_ma20:
    st.markdown(f'<p style="color: green; font-size: 20px; font-weight: bold;">{selected_ticker} Up_trend</p>',
        unsafe_allow_html=True)

# Check for down_trend condition
elif latest_closing_price <  latest_ma9 and latest_ma9 < latest_ma20:
    st.markdown(f'<p style="color: red; font-size: 20px; font-weight: bold;">{selected_ticker} Down_trend</p>',
        unsafe_allow_html=True)

else:
    st.markdown(f'<p style="color: gray; font-size: 20px; font-weight: bold;">{selected_ticker} trend unclear</p>',
        unsafe_allow_html=True)

# Find the smallest difference
smallest_diff = min(diff_moving_avg, diff_ma9, diff_ma20, diff_ma50, diff_ma100)

# Determine which moving average is closest
if smallest_diff == diff_moving_avg:
    closest_ma = "5-day Moving Average"
    closest_ma_value = latest_moving_avg
elif smallest_diff == diff_ma9:
    closest_ma = "9-day Moving Average"
    closest_ma_value = latest_ma9
elif smallest_diff == diff_ma20:
    closest_ma = "20-day Moving Average"
    closest_ma_value = latest_ma20
elif smallest_diff == diff_ma50:
    closest_ma = "50-day Moving Average"
    closest_ma_value = latest_ma50
else:
    closest_ma = "100-day Moving Average"
    closest_ma_value = latest_ma100

# Display the message indicating which moving average the selected ticker is approaching, along with its value
st.markdown(
    f'<p style="color: orange; font-size: 18px; font-weight: bold;">{selected_ticker} is approaching {closest_ma} (Value: {closest_ma_value:.2f})</p>',
    unsafe_allow_html=True
)

# Plot closing prices
ax.plot(np.arange(len(data) - 4), data[4:], color='black', label=f'{selected_ticker} Closing Prices', alpha=0.5)

# plot ma9
ax.plot(np.arange(len(data) - 4 ), ma9, color='red', label=f'{selected_ticker} ma9', alpha=0.5)

# plot ma20
ax.plot(np.arange(len(data) - 4), ma20, color='blue', label=f'{selected_ticker} ma20', alpha=0.5)

# plot ma50
ax.plot(np.arange(len(data) - 4), ma50, color='orange', label=f'{selected_ticker} ma50', alpha=0.5)

# plot ma100
ax.plot(np.arange(len(data) - 4), ma100, color='black', label=f'{selected_ticker} ma100', alpha=0.5)

# plot ma200
ax.plot(np.arange(len(data) - 4), ma200, color='purple', label=f'{selected_ticker} ma200', alpha=0.5)

# Plot 5-day moving averages
ax.scatter(X, y, color='blue', label='5-Day Moving Averages')
ax.plot(X, y_pred_linear, color='red', label=f'Linear Regression (R² = {r_squared_linear:.3f})')
ax.plot(X, y_pred_poly, color='green', label=f'Polynomial Regression (R² = {r_squared_poly:.3f})')

# Add shaded regions for multiple standard deviations only for the better model
std_dev_levels = [1.0, 1.5, 2.0]  # Removed 1.25
colors = ['gray', 'orange', 'red']  # Removed 'blue'
alphas = [0.2, 0.1, 0.05]  # Removed 0.15

for level, color, alpha in zip(std_dev_levels, colors, alphas):
    ax.fill_between(
        X.flatten(),  # X values
        better_y_pred - level * std_dev,  # Lower bound
        better_y_pred + level * std_dev,  # Upper bound
        color=color, alpha=alpha, label=f'±{level} Standard Deviations'
    )

# Plot future predictions
ax.scatter(future_X, future_y_better, color='purple', label=f'{model_type} Regression Predictions')

# Annotate future predictions with their values
for i, (x, y_pred) in enumerate(zip(future_X, future_y_better)):
    ax.text(x, y_pred, f'{y_pred:.2f}', color='purple', fontsize=8, ha='left', va='top')

# Add labels, title, and legend
ax.set_title(f"5-Day Moving Averages: {model_type} Regression with {prediction_period} Predictions (R² = {better_r_squared:.3f})")
ax.set_xlabel("Time Step")
ax.set_ylabel("Value")
ax.legend()
ax.grid()

# Display the plot in Streamlit
st.pyplot(fig)

# Display predictions below the plot
st.subheader(f"Predictions for the Next {prediction_period}")
st.write(f"**{model_type} Regression Predictions:**", future_y_better)
