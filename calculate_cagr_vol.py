import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Verify CSV file exists
csv_file = 'bitcoin_prices.csv'
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"The file {csv_file} was not found in the current directory.")

# Load dataset from CSV file
try:
    df = pd.read_csv(csv_file)
except Exception as e:
    raise Exception(f"Error reading {csv_file}: {str(e)}")

# Data preparation
# Convert 'Date' to datetime
try:
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
except ValueError as e:
    raise ValueError(f"Date format error in {csv_file}. Expected MM/DD/YYYY. Error: {str(e)}")

# Convert 'Price' to numeric, removing commas and handling errors
df['Price'] = pd.to_numeric(df['Price'].replace('[,]', '', regex=True), errors='coerce')

# Check for NaN values in 'Price' after conversion
if df['Price'].isna().any():
    raise ValueError("Some 'Price' values could not be converted to numeric. Check the data format in bitcoin_prices.csv.")

# Sort by date ascending and set index
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Calculate daily returns
df['Daily_Return'] = df['Price'].pct_change()

# Calculate historical Sharpe Ratio
risk_free_rate_annual = 0.02
trading_days = 252
risk_free_rate_daily = (1 + risk_free_rate_annual) ** (1 / trading_days) - 1
mean_daily_return_hist = df['Daily_Return'].mean()
std_daily_return_hist = df['Daily_Return'].std()
sharpe_ratio_hist = ((mean_daily_return_hist - risk_free_rate_daily) / std_daily_return_hist) * np.sqrt(trading_days)
print(f"Historical Sharpe Ratio (Annualized): {sharpe_ratio_hist:.4f}")

# Calculate rolling CAGR (1-year window)
window = 365
df['CAGR'] = (df['Price'] / df['Price'].shift(window)) ** (1 / 1) - 1

# Calculate rolling volatility (30-day window, annualized)
vol_window = 30
df['Volatility'] = df['Daily_Return'].rolling(window=vol_window).std() * np.sqrt(252)

# Calculate daily change in CAGR
df['CAGR_Change'] = df['CAGR'].diff()

# Drop NaN values for regression
reg_data = df[['Volatility', 'CAGR_Change']].dropna()

# Logarithmic transformation for CAGR_Change
shift_constant = abs(reg_data['CAGR_Change'].min()) + 1e-6
reg_data['CAGR_Change_Shifted'] = reg_data['CAGR_Change'] + shift_constant
reg_data['Log_CAGR_Change'] = np.log(reg_data['CAGR_Change_Shifted'])

# Random Forest regression with enhanced parameters
X = reg_data['Volatility'].values.reshape(-1, 1)
y = reg_data['Log_CAGR_Change'].values
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)
model.fit(X, y)

# Evaluate model
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Random Forest Regression Results:")
print(f"Mean Squared Error: {mse:.6f}")
print(f"Feature Importance (Volatility): {model.feature_importances_[0]:.6f}")

# Future projections
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end='2030-12-31', freq='D')
future_df = pd.DataFrame(index=future_dates)

# Simulate volatility with reflected Geometric Brownian Motion
mean_vol = reg_data['Volatility'].mean()
vol_std = reg_data['Volatility'].std()
mu = 0
sigma = 0.8
dt = 1 / 252
lower_bound = 0.2
upper_bound = 1.0

vol = np.zeros(len(future_dates))
vol[0] = df['Volatility'].iloc[-1] if df['Volatility'].iloc[-1] >= lower_bound and df['Volatility'].iloc[-1] <= upper_bound else mean_vol

for t in range(1, len(future_dates)):
    dW = np.random.normal(0, np.sqrt(dt))
    dlogV = mu * dt + sigma * dW
    vol[t] = vol[t-1] * np.exp(dlogV)
    if vol[t] < lower_bound:
        vol[t] = 2 * lower_bound - vol[t]
    elif vol[t] > upper_bound:
        vol[t] = 2 * upper_bound - vol[t]

future_df['Volatility'] = vol

# Predict log(CAGR_Change) and revert to original scale
log_cagr_change_pred = model.predict(future_df['Volatility'].values.reshape(-1, 1))
future_df['CAGR_Change'] = np.exp(log_cagr_change_pred) - shift_constant

# Add positive bias to keep CAGR above 0
positive_bias = 0.0005
future_df['CAGR_Change'] += positive_bias

# Add periodic spikes (every 6 months, ~183 days)
spike_interval = 45
spike_magnitude = 0.1
spike_days = np.arange(0, len(future_dates), spike_interval)
future_df['CAGR_Change'].iloc[spike_days] += spike_magnitude

# Calculate future CAGR starting from last historical CAGR
last_cagr = df['CAGR'].iloc[-1]
future_df['CAGR'] = last_cagr + future_df['CAGR_Change'].cumsum()

# Apply decay if CAGR exceeds thresholds for more than 90 days
decay_threshold_1 = 2
decay_threshold_2 = 5
decay_window = 15
decay_rate_1 = -0.005
decay_rate_2 = -0.02

cagr_above_1 = future_df['CAGR'] > decay_threshold_1
cagr_above_2 = future_df['CAGR'] > decay_threshold_2
sustained_above_1 = cagr_above_1.rolling(window=decay_window, min_periods=1).sum()
sustained_above_2 = cagr_above_2.rolling(window=decay_window, min_periods=1).sum()

decay_mask_1 = (sustained_above_1 >= decay_window) & (future_df['CAGR'] <= decay_threshold_2)
decay_mask_2 = sustained_above_2 >= decay_window

future_df.loc[decay_mask_1, 'CAGR_Change'] += decay_rate_1
future_df.loc[decay_mask_2, 'CAGR_Change'] += decay_rate_2

# Apply gain if CAGR falls below thresholds for more than 90 days
gain_threshold_1 = 0.0
gain_threshold_2 = -0.5
gain_window = 15
gain_rate_1 = 0.005
gain_rate_2 = 0.01

cagr_below_0 = future_df['CAGR'] < gain_threshold_1
cagr_below_neg_0_5 = future_df['CAGR'] < gain_threshold_2
sustained_below_0 = cagr_below_0.rolling(window=gain_window, min_periods=1).sum()
sustained_below_neg_0_5 = cagr_below_neg_0_5.rolling(window=gain_window, min_periods=1).sum()

gain_mask_1 = (sustained_below_0 >= gain_window) & (future_df['CAGR'] >= gain_threshold_2)
gain_mask_2 = sustained_below_neg_0_5 >= gain_window

future_df.loc[gain_mask_1, 'CAGR_Change'] += gain_rate_1
future_df.loc[gain_mask_2, 'CAGR_Change'] += gain_rate_2

# Recalculate CAGR with decay and gain adjustments
future_df['CAGR'] = last_cagr + future_df['CAGR_Change'].cumsum()

# Ensure CAGR stays above -1 and below historical max
max_cagr = 10.8659552424368
future_df['CAGR'] = future_df['CAGR'].clip(lower=-1, upper=max_cagr)

# Simulate future prices using CAGR
last_price = df['Price'].iloc[-1]
future_df['Price'] = last_price
for i in range(1, len(future_dates)):
    daily_rate = (1 + future_df['CAGR'].iloc[i]) ** (1/365) - 1
    future_df['Price'].iloc[i] = future_df['Price'].iloc[i-1] * (1 + daily_rate)

# Adjust future prices to maintain Sharpe Ratio between 0.5 and 1.0
future_df['Daily_Return'] = future_df['Price'].pct_change()
mean_daily_return_future = future_df['Daily_Return'].mean()
std_daily_return_future = future_df['Daily_Return'].std()
sharpe_ratio_future = ((mean_daily_return_future - risk_free_rate_daily) / std_daily_return_future) * np.sqrt(trading_days)
print(f"Initial Future Sharpe Ratio (Annualized): {sharpe_ratio_future:.4f}")

# Target Sharpe Ratio within 0.5 to 1.0
target_sharpe = min(max(sharpe_ratio_hist, 0.5), 1.0)
target_mean_return = target_sharpe * std_daily_return_future / np.sqrt(trading_days) + risk_free_rate_daily
adjustment_factor = (target_mean_return - risk_free_rate_daily) / (mean_daily_return_future - risk_free_rate_daily)

# Adjust the daily returns
future_df['Adjusted_Daily_Return'] = (future_df['Daily_Return'] - mean_daily_return_future) * adjustment_factor + target_mean_return

# Recalculate future prices using adjusted daily returns
future_df['Adjusted_Price'] = last_price
for i in range(1, len(future_df)):
    future_df['Adjusted_Price'].iloc[i] = future_df['Adjusted_Price'].iloc[i-1] * (1 + future_df['Adjusted_Daily_Return'].iloc[i])

# Verify the adjusted Sharpe Ratio
mean_daily_return_adj = future_df['Adjusted_Daily_Return'].mean()
std_daily_return_adj = future_df['Adjusted_Daily_Return'].std()
sharpe_ratio_adj = ((mean_daily_return_adj - risk_free_rate_daily) / std_daily_return_adj) * np.sqrt(trading_days)
print(f"Adjusted Future Sharpe Ratio (Annualized): {sharpe_ratio_adj:.4f}")

# Update future_df with adjusted prices
future_df['Price'] = future_df['Adjusted_Price']
future_df.drop(columns=['Daily_Return', 'Adjusted_Daily_Return', 'Adjusted_Price'], inplace=True)

# Combine historical and future data
combined_df = pd.concat([df[['Price', 'CAGR', 'Volatility']], future_df[['Price', 'CAGR', 'Volatility']]])

# Save results to CSV
combined_df.to_csv('bitcoin_cagr_volatility_projection_adjusted.csv')

# Select relevant columns and drop NaN values
result = df[['Price', 'CAGR', 'Volatility']].dropna()

# Save results to CSV
result.to_csv('bitcoin_cagr_volatility.csv')

# Display first and last few rows
print("First few rows:")
print(result.head())
print("\nLast few rows:")
print(result.tail())

# Plotting
plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax1.plot(combined_df.index, combined_df['CAGR'], label='CAGR (Historical + Projected)', color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('CAGR', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')
ax1.axvline(x=last_date, color='gray', linestyle='--', label='Projection Start')
plt.title('Bitcoin Rolling CAGR with Projections to 2030')

# Secondary axis for volatility
ax2 = ax1.twinx()
ax2.plot(combined_df.index, combined_df['Volatility'], label='Volatility', color='orange', alpha=0.5)
ax2.set_ylabel('Volatility (Annualized)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.set_ylim(0, 1.2)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('bitcoin_cagr_projection.png')
plt.close()

# Display last few historical and first few projected rows
print("\nLast few historical rows:")
print(df[['Price', 'CAGR', 'Volatility']].tail())
print("\nFirst few projected rows (adjusted):")
print(future_df[['Price', 'CAGR', 'Volatility']].head())
print("\nLast few projected rows (adjusted):")
print(future_df[['Price', 'CAGR', 'Volatility']].tail())

# Create a summary table with Sharpe Ratios
summary_data = {
    'Dataset': ['Historical', 'Projected (Adjusted)'],
    'Sharpe Ratio (Annualized)': [sharpe_ratio_hist, sharpe_ratio_adj],
    'Mean Daily Return': [mean_daily_return_hist, mean_daily_return_adj],
    'Std Dev of Daily Returns': [std_daily_return_hist, std_daily_return_adj]
}
summary_df = pd.DataFrame(summary_data)
print("\nSummary Table with Sharpe Ratios:")
print(summary_df.to_string(index=False))

