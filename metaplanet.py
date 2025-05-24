import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.gridspec import GridSpec
from bitcoin_prediction import predict_bitcoin_prices
from data_sources import (
    get_metaplanet_3350_data,
    get_bitcoin_historical_data
)

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
    Calculates mNAV based on natural log model of Bitcoin NAV in JPY
    Returns: Series with mNAV values
    """
    # Calculate total Bitcoin value in USD
    btc_value = btc_holdings * btc_price
        
    # mNAV is the ratio between market price and underlying BTC value per share
    mnav = market_cap / btc_value
    
    return mnav

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

def simulate_through_2030(btc_data, meta_3350_data, initial_shares, btc_holdings, start_date=None, end_date="2030-12-31"):
    """Simulates Metaplanet metrics through 2030"""
    # Setup simulation dates
    sim_start = pd.Timestamp(start_date)
    sim_end = pd.Timestamp(end_date)
    future_dates = pd.date_range(start=sim_start, end=sim_end, freq='D')
    
    # Initialize simulation DataFrame
    simulation = pd.DataFrame(index=future_dates)
    
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
    current_shares = float(initial_shares)
    current_btc = float(btc_holdings.iloc[-1] if not btc_holdings.empty else initial_shares)
    prev_stock_price = float(meta_3350_data['Close'].iloc[-1] if not meta_3350_data.empty else 5.0)
    
    # Initialize mNAV based on both models
    initial_btc_nav = current_btc * simulation.loc[simulation.index[0], 'btc_price']
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
    
    for date in simulation.index:
        btc_price = simulation.loc[date, 'btc_price']
        btc_value = current_btc * btc_price
        
        # Calculate power law mNAV component
        theoretical_mcap = 35.1221 * (btc_value ** 0.91)
        power_law_mnav = theoretical_mcap / btc_value
        
        # Calculate sensitivity-based mNAV component
        if date == simulation.index[0]:
            sensitivity_mnav = prev_mnav
        else:
            btc_change = (btc_price - simulation['btc_price'].shift(1).loc[date]) / simulation['btc_price'].shift(1).loc[date]
            sensitivity_mnav = prev_mnav * (1 + btc_change * 0.2)  # 5:1 sensitivity
        
        # Combine both mNAV models with 50/50 weight
        current_mnav = 0.5 * power_law_mnav + 0.5 * sensitivity_mnav
        
        # Calculate volume with decay and mNAV influence
        days_from_start = (date - sim_start).days
        decay_factor = np.exp(-decay_rate * days_from_start)
        base_daily_pct = 0.25 * decay_factor  # Decaying from 25% to 3%
        
        # Combine base volume with mNAV influence
        mnav_factor = 1 + (current_mnav - 1) if current_mnav > 1 else 1 / current_mnav
        volatility_factor = 1 + np.random.normal(0, base_vol_volatility)
        daily_volume = current_shares * base_daily_pct * mnav_factor * max(0.1, volatility_factor)
        
        # Stock price calculation using combined mNAV
        stock_price = (btc_value * current_mnav) / current_shares
        
        # Dilution check - only on 2% price increases with fixed 0.3% dilution
        price_increase = (stock_price / prev_stock_price - 1) if prev_stock_price > 0 else 0
        if price_increase > 0.02:
            new_shares = current_shares * 0.003  # Fixed 0.3% dilution
            funds_raised = new_shares * stock_price
            btc_purchased = funds_raised / btc_price
            
            current_shares += new_shares
            current_btc += btc_purchased
        
        # Store simulation results
        simulation.loc[date, 'stock_price'] = stock_price
        simulation.loc[date, 'volume'] = daily_volume
        simulation.loc[date, 'shares_outstanding'] = current_shares
        simulation.loc[date, 'btc_holdings'] = current_btc
        simulation.loc[date, 'mnav'] = current_mnav
        
        prev_mnav = current_mnav
        prev_stock_price = stock_price
    
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
    last_historical_date = historical_df['Reported'].max()
    ax3.plot(historical_df['Reported'], historical_df['BTC Holding'], 'b-', 
            label='Historical Holdings', linewidth=2)
    
    future_data = simulation[simulation.index > last_historical_date].copy()
    if not future_data.empty:
        transition_point = pd.DataFrame({
            'btc_holdings': historical_df['BTC Holding'].iloc[-1]
        }, index=[last_historical_date])
        future_data = pd.concat([transition_point, future_data])
        ax3.plot(future_data.index, future_data['btc_holdings'], 'r--', 
                label='Projected Holdings', linewidth=2)

    # Dynamic y-axis for BTC holdings
    max_btc = max(historical_df['BTC Holding'].max(), simulation['btc_holdings'].max())
    ax3.set_ylim(0, max_btc * 1.05)
    
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
    
    # Volume with dynamic y-axis
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(simulation.index, simulation['volume'], 'b-', label='Daily Volume', linewidth=2)
    ax6.set_ylim(simulation['volume'].min() * 0.95, simulation['volume'].max() * 1.05)
    
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

