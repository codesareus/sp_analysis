####spy.py final


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

# Fetch the latest SPY closing price
def get_latest_spy_price():
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1d")  # Fetch the latest day's data
    return hist["Close"].iloc[-1]

# Initialize variables
latest_spy_price = None
price_difference = None

# Fetch SPY data based on the selected time period

st.sidebar.subheader("Historical Data Settings")
time_period = st.sidebar.selectbox(
        "Select time period:",
        options=["3mo", "6mo", "1y", "2y", "5y"],  # Available time periods
        index=0  # Default: 3 months
)

# Download historical data for SPY
spy = yf.Ticker("SPY")
hist = spy.history(period=time_period)
hist0 = spy.history(period="5y")

# Ensure the index is a DatetimeIndex
if not isinstance(hist.index, pd.DatetimeIndex):
        hist.index = pd.to_datetime(hist.index)

# Ensure the index is a DatetimeIndex
if not isinstance(hist0.index, pd.DatetimeIndex):
        hist0.index = pd.to_datetime(hist0.index)

# Extract closing prices and round to two decimal places
closing_prices = hist["Close"].round(2)
closing_prices0 = hist0["Close"].round(2)

# Check if the dataset is empty
if len(closing_prices) == 0:
        st.error("No data found for the selected time period. Please try again.")
        st.stop()  # Stop execution if the dataset is empty

# Convert the closing prices to a comma-separated string
closing_prices_str = ",".join(map(str, closing_prices))
closing_prices0_str = ",".join(map(str, closing_prices0))

# Display the closing prices in a fixed-height scrollable container
st.sidebar.write(f"**SPY Closing Prices (Last {time_period}):**")
    
# Custom CSS to create a fixed-height scrollable container
st.markdown(
        """
        <style>
        .scrollable-container {
            height: 200px;  /* Fixed height */
            overflow-y: auto;  /* Enable vertical scrolling */
            border: 1px solid #ccc;  /* Optional: Add a border */
            padding: 10px;  /* Optional: Add padding */
        }
        </style>
        """,
        unsafe_allow_html=True,
)

# Use the custom CSS class for the container
st.sidebar.markdown(
        f'<div class="scrollable-container">{closing_prices_str}</div>',
        unsafe_allow_html=True,
)

# Use the closing prices as the dataset
data = closing_prices.tolist()
data0 = closing_prices0.tolist()

# Save the SPY data as a CSV file
spy_data_df = pd.DataFrame({
        "Date": hist.index.strftime('%Y-%m-%d'),  # Format dates as strings
        "Closing Price": closing_prices
})
spy_data_df.to_csv("spy.csv", index=False)
st.sidebar.success("SPY data saved as spy.csv")

# Fetch the latest SPY closing price
latest_spy_price = get_latest_spy_price()

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

# Calculate residuals (errors) for the polynomial model
residuals = y - y_pred_poly
std_dev = np.std(residuals)  # Standard deviation of residuals

# Convert the selected prediction period to an integer
prediction_days = int(prediction_period.split()[0])  # Extract the number of days

# Predict future days
future_X = np.arange(len(moving_avg), len(moving_avg) + prediction_days).reshape(-1, 1)  # Future time steps
future_y_linear = linear_model.predict(future_X)  # Linear regression predictions
future_y_poly = poly_model.predict(future_X)  # Polynomial regression predictions

# Display current SPY price and price difference above the plot
st.write("### SPY Price Information")
if latest_spy_price is not None:
    st.write(f"**Current SPY Closing Price:** ${latest_spy_price:.2f}")
else:
    st.write("**Current SPY Closing Price:** Not available")

if price_difference is not None:
    st.write(f"**Difference between last two SPY closing prices:** ${price_difference:.2f}")
else:
    st.write("**Difference between last two SPY closing prices:** Not enough data")

# Add the comparison logic and display the trend message
latest_closing_price = data[-1]  # Latest closing price
latest_moving_avg = moving_avg[-1]  # Latest 5-day moving average
latest_ma9 = ma9[-1]  # Latest 9-day moving average
latest_ma20 = ma20[-1]  # Latest 20-day moving average
latest_ma50 = ma50[-1]  # Latest 50-day moving average
latest_ma100 = ma100[-1]  # Latest 50-day moving average

# Calculate the absolute differences between SPY and moving averages
diff_moving_avg = abs(latest_closing_price - latest_moving_avg)
diff_ma9 = abs(latest_closing_price - latest_ma9)
diff_ma20 = abs(latest_closing_price - latest_ma20)
diff_ma50 = abs(latest_closing_price - latest_ma50)
diff_ma100 = abs(latest_closing_price - latest_ma100)

# Check for up_trend condition
if latest_closing_price  > latest_ma9 and latest_ma9 > latest_ma20:
    st.markdown('<p style="color: green; font-size: 20px; font-weight: bold;">SPY Up_trend</p>',
        unsafe_allow_html=True)

# Check for down_trend condition
elif latest_closing_price <  latest_ma9 and latest_ma9 < latest_ma20:
    st.markdown('<p style="color: red; font-size: 20px; font-weight: bold;">SPY Down_trend</p>',
        unsafe_allow_html=True)

else:
    st.markdown('<p style="color: gray; font-size: 20px; font-weight: bold;">SPY trend unclear</p>',
        unsafe_allow_html=True)

# Find the smallest difference
smallest_diff = min(diff_moving_avg, diff_ma9, diff_ma20, diff_ma50, diff_ma100)

#########################
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

# Display the message indicating which moving average SPY is approaching, along with its value
st.markdown(
    f'<p style="color: orange; font-size: 18px; font-weight: bold;">SPY is approaching {closest_ma} (Value: {closest_ma_value:.2f})</p>',
    unsafe_allow_html=True
)

# Plot the data and regression lines
st.subheader("Plot")
fig, ax = plt.subplots(figsize=(12, 8))  # Wider plot to accommodate predictions

# Plot SPY closing prices
ax.plot(np.arange(len(data) - 4), data[4:], color='black', label='SPY Closing Prices', alpha=0.5)

# plot ma9
ax.plot(np.arange(len(data) - 4 ), ma9, color='red', label='SPY ma9', alpha=0.5)

# plot ma20
ax.plot(np.arange(len(data) - 4), ma20, color='blue', label='SPY ma20', alpha=0.5)

# plot ma50
ax.plot(np.arange(len(data) - 4), ma50, color='orange', label='SPY ma50', alpha=0.5)

# plot ma100
ax.plot(np.arange(len(data) - 4), ma100, color='black', label='SPY ma100', alpha=0.5)

# plot ma200
ax.plot(np.arange(len(data) - 4), ma200, color='purple', label='SPY ma200', alpha=0.5)

# Plot 5-day moving averages
ax.scatter(X, y, color='blue', label='5-Day Moving Averages')
ax.plot(X, y_pred_linear, color='red', label=f'Linear Regression (R² = {r_squared_linear:.3f})')
ax.plot(X, y_pred_poly, color='green', label=f'Polynomial Regression (R² = {r_squared_poly:.3f})')

# Add shaded regions for multiple standard deviations
std_dev_levels = [1.0, 1.5, 2.0]  # Removed 1.25
colors = ['gray', 'orange', 'red']  # Removed 'blue'
alphas = [0.2, 0.1, 0.05]  # Removed 0.15

for level, color, alpha in zip(std_dev_levels, colors, alphas):
    ax.fill_between(
        X.flatten(),  # X values
        y_pred_poly - level * std_dev,  # Lower bound
        y_pred_poly + level * std_dev,  # Upper bound
        color=color, alpha=alpha, label=f'±{level} Standard Deviations'
    )

# Plot future predictions
ax.scatter(future_X, future_y_linear, color='orange', label='Linear Regression Predictions')
ax.scatter(future_X, future_y_poly, color='purple', label='Polynomial Regression Predictions')

# Annotate future predictions with their values
for i, (x, y_lin, y_poly) in enumerate(zip(future_X, future_y_linear, future_y_poly)):
    ax.text(x, y_lin, f'{y_lin:.2f}', color='orange', fontsize=8, ha='right', va='bottom')
    ax.text(x, y_poly, f'{y_poly:.2f}', color='purple', fontsize=8, ha='left', va='top')

# Add labels, title, and legend
ax.set_title(f"5-Day Moving Averages: Linear vs Polynomial Regression with {prediction_period} Predictions")
ax.set_xlabel("Time Step")
ax.set_ylabel("Value")
ax.legend()
ax.grid()

# Display the plot in Streamlit
st.pyplot(fig)

# Display predictions below the plot
st.subheader(f"Predictions for the Next {prediction_period}")
st.write("**Linear Regression Predictions:**", future_y_linear)
st.write("**Polynomial Regression Predictions:**", future_y_poly)
