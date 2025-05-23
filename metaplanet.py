import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.gridspec import GridSpec
from bitcoin_prediction import predict_bitcoin_prices
from mnav_prediction import calculate_mnav_with_volatility
from data_sources import (
    get_metaplanet_3350_data,
    get_bitcoin_historical_data
)
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import CustomBusinessDay

def get_bitcoin_holdings():
    """
    Gets Metaplanet's Bitcoin holdings since April 1st 2024
    Returns: Tuple of (holdings Series, purchase DataFrame)
    """
    # Read the CSV file
    df = pd.read_csv('mp_btc.csv')
    
    # Clean up BTC Holding column and convert to float
    df['BTC Holding'] = df['BTC Holding'].astype(float)
    df['Reported'] = pd.to_datetime(df['Reported'])
    df = df.sort_values('Reported')
    
    # Calculate purchases with proper indexing
    df['BTC_Purchased'] = df['BTC Holding'].diff()
    purchases = pd.DataFrame(index=df['Reported'])
    purchases['BTC_Purchased'] = df['BTC_Purchased']
    purchases = purchases[purchases['BTC_Purchased'] > 0]  # Filter after DataFrame is created
    
    # Create holdings series
    date_range = pd.date_range(start=df['Reported'].min(), end=datetime.now().date(), freq='D')
    btc_holdings = pd.Series(index=date_range, dtype=float)
    btc_holdings[df['Reported']] = df['BTC Holding']
    btc_holdings = btc_holdings.ffill()  # Use ffill() instead of fillna(method='ffill')
    
    return btc_holdings, purchases

def calculate_mnav(market_cap, btc_holdings, btc_price):
    """
    Calculates mNAV based on market cap and BTC value
    Returns: Series with mNAV values
    """
    btc_value = btc_holdings * btc_price
    return market_cap / btc_value

def predict_future_mnav(days_from_start, btc_value):
    """
    Predicts future mNAV using the Weierstrass volatility model
    Returns: Float with predicted mNAV value
    """
    return calculate_mnav_with_volatility(btc_value, days_from_start)

def calculate_btc_per_share(btc_holdings, outstanding_shares):
    """
    Calculates BTC per share
    Returns: Series with BTC per share values
    """
    return btc_holdings / outstanding_shares

def predict_future_mnav(historical_mnav, btc_price_prediction):
    """
    Predicts future mNAV based on BTC correlation
    Returns: Series with predicted mNAV values
    """
    correlation = historical_mnav.corr(btc_price_prediction)
    return historical_mnav * correlation * (btc_price_prediction / btc_price_prediction.iloc[0])

def predict_mnav_with_swings(historical_mnav, btc_price):
    """
    Predicts mNAV considering historical swings
    Returns: Series with predicted mNAV values
    """
    volatility = historical_mnav.std()
    base_prediction = predict_future_mnav(historical_mnav, btc_price)
    return base_prediction * (1 + np.random.normal(0, volatility, len(base_prediction)))

def calculate_stock_price(predicted_mnav, btc_holdings, outstanding_shares):
    """
    Calculates stock price based on predicted mNAV
    Returns: Series with predicted stock prices
    """
    return predicted_mnav * btc_holdings / outstanding_shares

def calculate_daily_dilution(price_data, volume_data):
    """
    Calculates daily dilution and funds raised
    Returns: DataFrame with dilution amount and funds raised
    """
    daily_returns = price_data.pct_change()
    dilution = pd.DataFrame(index=price_data.index)
    dilution['dilution_shares'] = 0.0
    dilution['funds_raised'] = 0.0
    
    mask = daily_returns > 0.03
    dilution.loc[mask, 'dilution_shares'] = volume_data[mask] * 0.1
    dilution.loc[mask, 'funds_raised'] = dilution['dilution_shares'] * price_data[mask]
    
    return dilution

def calculate_bitcoin_purchases(funds_raised, btc_price):
    """
    Calculates Bitcoin purchases from dilution proceeds
    Returns: Series with BTC amounts purchased
    """
    return funds_raised / btc_price

def predict_bitcoin_price(historical_data, end_date="2030-12-31"):
    """
    Predicts Bitcoin price using combined models
    Args:
        historical_data: Series or DataFrame with bitcoin prices
        end_date: String date format YYYY-MM-DD
    Returns: Series of predicted prices
    """
    if historical_data.empty:
        last_price = 65000  # Default price if no historical data
    else:
        if isinstance(historical_data, pd.DataFrame):
            last_price = historical_data['Close'].iloc[-1]
        else:
            last_price = historical_data.iloc[-1]
    
    start_date = datetime.now().date() if historical_data.empty else historical_data.index[-1] + timedelta(days=1)
    
    future_prices = predict_bitcoin_prices(
        start_date=start_date,
        end_date=end_date,
        last_price=last_price
    )
    return future_prices['Price']

def is_tse_trading_day(date):
    """Check if date is a Tokyo Stock Exchange trading day"""
    # TSE is closed on weekends
    if date.weekday() in [5, 6]:  # Saturday = 5, Sunday = 6
        return False
        
    # Japanese holidays (simplified list - add more as needed)
    holidays = [
        "2024-01-01", "2024-01-02", "2024-01-03",  # New Year
        "2024-01-08", "2024-02-12", "2024-02-23",  # Coming of Age, Foundation, Emperor's Birthday
        "2024-03-20", "2024-04-29", "2024-05-03",  # Spring Equinox, Showa Day, Constitution
        "2024-05-04", "2024-05-05", "2024-05-06",  # Greenery, Children's, Holiday
        "2024-07-15", "2024-08-12", "2024-09-16",  # Marine, Mountain, Respect for Aged
        "2024-09-23", "2024-10-14", "2024-11-03",  # Autumn Equinox, Sports, Culture
        "2024-11-23", "2024-12-31"                 # Labor, Market Holiday
    ]
    return str(date.date()) not in holidays

def simulate_through_2030(btc_data, meta_3350_data, initial_shares, btc_holdings, start_date=None, end_date="2030-12-31"):
    """Simulates Metaplanet metrics through 2030"""
    # Use current date if no start date provided
    if start_date is None:
        start_date = datetime.now().date()
    
    sim_start = pd.Timestamp(start_date)
    sim_end = pd.Timestamp(end_date)
    
    # Load historical BTC holdings from CSV and get final value
    df = pd.read_csv('mp_btc.csv')
    df['Reported'] = pd.to_datetime(df['Reported'])
    df = df.sort_values('Reported')
    initial_btc = float(df['BTC Holding'].iloc[-1])  # Use final historical value (7800)
    
    # Initialize simulation DataFrame
    future_dates = pd.date_range(start=sim_start, end=sim_end, freq='D')
    simulation = pd.DataFrame(index=future_dates)
    simulation['is_trading_day'] = simulation.index.map(is_tse_trading_day)
    simulation['btc_holdings'] = initial_btc
    
    # Track cumulative values
    current_btc = initial_btc  # Start with historical final value
    current_shares = float(initial_shares)
    cumulative_btc_purchased = 0.0  # Track only new purchases
    min_dilution_interval = 3  # Days between dilution events

    # Calculate S-curve dilution parameters
    peak_annual_rate = 1.0  # 100% dilution at peak
    end_annual_rate = 0.33  # 33% dilution by 2030 (changed from 0.10)
    total_days = (sim_end - sim_start).days
    steepness = 6.0 / total_days  # Controls S-curve steepness
    
    def get_dilution_rate(days_from_start):
        """Calculate dilution rate using logistic decay function"""
        # Start at peak and decay to end rate
        return end_annual_rate + (peak_annual_rate - end_annual_rate) / (1 + np.exp(steepness * days_from_start))

    # Add historical BTC prices where available
    historical_data = btc_data.reindex(future_dates)
    simulation['btc_price'] = historical_data['Close']
    
    # Get future Bitcoin price predictions for missing dates
    missing_dates = simulation[simulation['btc_price'].isna()].index
    if len(missing_dates) > 0:
        last_known_date = simulation[simulation['btc_price'].notna()].index[-1]
        last_known_price = simulation.loc[last_known_date, 'btc_price']
        future_prices = predict_bitcoin_price(btc_data, end_date)
        simulation.loc[missing_dates, 'btc_price'] = future_prices
    
    # Fill any remaining gaps
    simulation['btc_price'] = simulation['btc_price'].ffill()
    
    # Initialize starting values
    prev_stock_price = float(meta_3350_data['Close'].iloc[-1] if not meta_3350_data.empty else 5.0)
    
    # Initialize mNAV based on both models
    initial_btc_nav = current_btc * simulation['btc_price'].iloc[0]  # Use iloc instead of direct indexing
    theoretical_mcap = 35.1221 * (initial_btc_nav ** 0.89)  # Power law model
    initial_mnav = theoretical_mcap / initial_btc_nav
    prev_mnav = initial_mnav
    
    # Initialize volume decay parameters
    total_days = (sim_end - sim_start).days
    end_volume_target = 0.01  # 3% of shares as final target
    decay_rate = -np.log(end_volume_target / 0.05) / total_days  # Decay from 25% to 3%
    
    # Get historical volume trend components
    base_vol_volatility = meta_3350_data['Volume'].std() / meta_3350_data['Volume'].mean()
    base_volume = meta_3350_data['Volume_Trend'].iloc[-1]
    volume_growth = meta_3350_data['Volume_Growth'].iloc[-1]
    
    # Add dilution cycle counter
    days_since_dilution = 0
    
    for date in simulation.index:
        btc_price = simulation.loc[date, 'btc_price']
        btc_value = current_btc * btc_price
        days_from_start = (date - sim_start).days
        
        # Calculate mNAV using imported function
        current_mnav = calculate_mnav_with_volatility(btc_value, days_from_start)
        
        # Calculate volume with decay and mNAV influence
        decay_factor = np.exp(-decay_rate * days_from_start)
        base_daily_pct = 0.25 * decay_factor  # Decaying from 25% to 3%
        
        # Combine base volume with mNAV influence
        mnav_factor = 1 + (current_mnav - 1) if current_mnav > 1 else 1 / current_mnav
        volatility_factor = 1 + np.random.normal(0, base_vol_volatility)
        daily_volume = current_shares * base_daily_pct * mnav_factor * max(0.1, volatility_factor)
        
        # Only update stock price and apply dilution on trading days
        if simulation.loc[date, 'is_trading_day']:
            stock_price = (btc_value * current_mnav) / current_shares
            
            if date >= sim_start:
                days_since_dilution += 1
                price_increase = (stock_price / prev_stock_price - 1) if prev_stock_price > 0 else 0
                btc_purchased = 0.0
                
                if days_since_dilution >= min_dilution_interval and stock_price >= prev_stock_price:
                    # Get current target annual dilution rate from S-curve
                    current_annual_rate = get_dilution_rate(days_from_start)
                    daily_dilution_target = (1 + current_annual_rate) ** (1/365) - 1
                    target_new_shares = current_shares * daily_dilution_target * min_dilution_interval
                    
                    # Use volume as a limiter - max 10% of daily volume
                    volume_limit = daily_volume * 0.10
                    new_shares = min(target_new_shares, volume_limit)
                    
                    # Calculate BTC purchases based on raised funds
                    funds_raised = new_shares * stock_price
                    btc_purchased = funds_raised / btc_price  # No artificial limit
                    
                    # Apply dilution
                    current_shares += new_shares
                    current_btc += btc_purchased
                    cumulative_btc_purchased += btc_purchased
                    days_since_dilution = 0
                
                simulation.loc[date, 'shares_outstanding'] = current_shares
                simulation.loc[date, 'btc_purchased'] = btc_purchased
                prev_stock_price = stock_price
        else:
            stock_price = prev_stock_price if prev_stock_price is not None else 5.0
            simulation.loc[date, 'btc_purchased'] = 0.0
        
        # Store simulation results
        simulation.loc[date, 'stock_price'] = stock_price
        simulation.loc[date, 'volume'] = daily_volume if simulation.loc[date, 'is_trading_day'] else 0
        simulation.loc[date, 'shares_outstanding'] = current_shares
        simulation.loc[date, 'btc_holdings'] = current_btc
        simulation.loc[date, 'mnav'] = current_mnav

    # Forward fill any missing values in final results
    simulation = simulation.ffill()  # Use ffill() instead of fillna(method='ffill')
    
    return simulation

def plot_simulation_results(simulation):
    """Plot simulation results with aligned axes"""
    # Read historical BTC holdings
    historical_df = pd.read_csv('mp_btc.csv')
    historical_df['Reported'] = pd.to_datetime(historical_df['Reported'])
    historical_df = historical_df.sort_values('Reported')
    
    # Set common x-axis limits
    start_date = pd.Timestamp('2024-04-01')
    end_date = simulation.index[-1]
    date_formatter = plt.matplotlib.dates.DateFormatter('%Y-%m')

    # Create complete BTC holdings series
    complete_holdings = pd.Series(index=pd.date_range(start=start_date, end=end_date, freq='D'))
    
    # Add historical data up to last historical date
    historical_dates = historical_df['Reported']
    last_historical_date = historical_dates.max()
    complete_holdings[historical_dates] = historical_df['BTC Holding']
    complete_holdings = complete_holdings.ffill()
    
    # Add simulated data starting exactly from last historical value
    first_sim_date = simulation.index[0]
    complete_holdings[first_sim_date:] = simulation['btc_holdings']

    # Update simulation's BTC holdings to match complete series
    simulation['btc_holdings'] = complete_holdings[simulation.index]
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Price plots with dynamic y-axes
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(simulation.index, simulation['btc_price'], 'b-', label='BTC Price', linewidth=2)
    ax1.set_ylabel('BTC Price (USD)')
    ax1.set_title('Bitcoin Price')
    ax1.set_ylim(simulation['btc_price'].min() * 0.95, simulation['btc_price'].max() * 1.05)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(simulation.index, simulation['stock_price'], 'g-', label='Stock Price', linewidth=2)
    ax2.set_ylabel('Stock Price (USD)')
    ax2.set_title('Metaplanet Stock Price')
    ax2.set_ylim(simulation['stock_price'].min() * 0.95, simulation['stock_price'].max() * 1.05)
    
    # Holdings plot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(complete_holdings.index, complete_holdings.values, 'b-', label='BTC Holdings', linewidth=2)
    ax3.set_ylabel('BTC Holdings')
    ax3.set_title('Bitcoin Holdings')

    # Dynamic y-axis for BTC holdings
    ax3.set_ylim(0, complete_holdings.max() * 1.05)
    
    # Shares outstanding with dynamic y-axis
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(simulation.index, simulation['shares_outstanding'], 'g-', 
            label='Shares Outstanding', linewidth=2)
    ax4.set_ylim(simulation['shares_outstanding'].min() * 0.95, 
                 simulation['shares_outstanding'].max() * 1.05)
    
    # mNAV with dynamic y-axis
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(simulation.index, simulation['mnav'], 'r-', label='mNAV', linewidth=2)
    ax5.set_ylim(simulation['mnav'].min() * 0.95, simulation['mnav'].max() * 1.05)
    
    # Replace volume plot with dilution plot
    ax6 = fig.add_subplot(gs[2, 1])
    diluted_shares = simulation['shares_outstanding'].diff()
    diluted_shares.iloc[0] = 0  # Set first day's dilution to 0 using iloc
    ax6.plot(simulation.index, diluted_shares, 'g-', label='Daily Share Dilution', linewidth=2)
    ax6.set_ylabel('Shares Issued')
    ax6.set_title('Daily Share Dilution')
    ax6.set_ylim(0, diluted_shares.max() * 1.05)
    
    # Common settings for all plots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.grid(True)
        ax.xaxis.set_major_formatter(date_formatter)
        ax.set_xlim(start_date, end_date)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
    
    plt.tight_layout()
    return fig

def run_complete_simulation(start_date="2024-04-01", end_date="2030-12-31", initial_shares=593210000, initial_btc=7800):
    """
    Runs complete simulation and generates visualizations
    Default values:
    - 593.21M shares (current shares outstanding)
    - 7800 BTC (approximate current holdings)
    """
    print("Starting simulation...")
    
    # Get historical data
    btc_data = get_bitcoin_historical_data()
    meta_3350_data = get_metaplanet_3350_data()
    btc_holdings, _ = get_bitcoin_holdings()
    
    # Run simulation and get merged data
    simulation = simulate_through_2030(
        btc_data, meta_3350_data, initial_shares, btc_holdings, 
        start_date=start_date, end_date=end_date
    )
    
    # Create and save plots
    print("Generating plots...")
    fig = plot_simulation_results(simulation)
    fig.savefig('metaplanet_simulation_2030.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    simulation.to_csv('metaplanet_simulation_2030.csv')
    
    # Print summary statistics
    print("\nSimulation Results for 2030:")
    print(f"Final Bitcoin Price: ${simulation['btc_price'].iloc[-1]:,.2f}")
    print(f"Final Stock Price: ${simulation['stock_price'].iloc[-1]:,.2f}")
    print(f"Final BTC Holdings: {simulation['btc_holdings'].iloc[-1]:,.2f} BTC")
    print(f"Final Shares Outstanding: {simulation['shares_outstanding'].iloc[-1]:,.0f}")
    print(f"Final mNAV: {simulation['mnav'].iloc[-1]:,.4f}")
    
    return simulation

def main(start_date="2024-05-18", end_date="2030-12-31", initial_shares=593210000, initial_btc=0):
    """
    Main function to run the complete Metaplanet analysis and simulation
    Args:
        start_date (str): Simulation start date (YYYY-MM-DD)
        end_date (str): Simulation end date (YYYY-MM-DD)
        initial_shares (int): Initial number of outstanding shares
        initial_btc (float): Initial BTC holdings
    """
    print("Starting Metaplanet analysis...")
    
    # Run simulation
    print("Running simulation through 2030...")
    simulation = run_complete_simulation(
        start_date=start_date, end_date=end_date, 
        initial_shares=initial_shares, initial_btc=initial_btc
    )
    
    return simulation

if __name__ == "__main__":
    # Example usage: provide your own start_date, initial_shares, and initial_btc
    # simulation_results = run_complete_simulation("2024-04-01", "2030-12-31", 1234567, 100)
    simulation_results = run_complete_simulation()
