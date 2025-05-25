import pandas as pd
import numpy as np
from datetime import datetime
from scipy import optimize

def btc_power_law_formula(index):
    """Bitcoin Power Law model"""
    genesis = pd.Timestamp('2009-01-03')
    days_since_genesis = (index - genesis).days.values.astype(float)
    days_since_genesis[days_since_genesis < 1] = 1
    price = 10**-17 * (days_since_genesis ** 5.8)
    support = 0.5 * price
    resistance = 2.0 * price
    return support, price, resistance

def weierstrass_function(t, a=0.5, b=3, n_terms=10):
    """Generate Weierstrass-like function"""
    w = np.zeros_like(t, dtype=float)
    for n in range(n_terms):
        w += a**n * np.cos(np.pi * b**n * t)
    w = w / np.max(np.abs(w))
    return w

def multi_weierstrass(t, configs):
    """Overlay multiple Weierstrass functions"""
    wsum = np.zeros_like(t, dtype=float)
    for cfg in configs:
        w = weierstrass_function(
            t * cfg.get('scale', 1.0),
            a=cfg.get('a', 0.5),
            b=cfg.get('b', 3),
            n_terms=cfg.get('n_terms', 10)
        )
        wsum += cfg.get('weight', 1.0) * w
    wsum = wsum / np.max(np.abs(wsum))
    return wsum

def predict_bitcoin_prices(start_date, end_date, last_price):
    """Predict Bitcoin prices using combined power law and Weierstrass models with smooth transition"""
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_df = pd.DataFrame(index=future_dates)
    
    _, future_center, _ = btc_power_law_formula(future_dates)
    
    t = np.arange(len(future_df))
    # Increased Weierstrass weights by another 100%
    configs = [
        {'a': 0.5, 'b': 3, 'n_terms': 10, 'weight': 1.816, 'scale': 1/730},    # Doubled from 0.908
        {'a': 0.8, 'b': 2.5, 'n_terms': 8, 'weight': 1.210, 'scale': 1/1825},  # Doubled from 0.605
        {'a': 0.3, 'b': 2.2, 'n_terms': 6, 'weight': 0.606, 'scale': 1/180}    # Doubled from 0.303
    ]
    w = multi_weierstrass(t, configs)
    
    # Initialize price array
    prices = np.zeros(len(future_df))
    initial_price = float(last_price.iloc[0]) if isinstance(last_price, pd.Series) else float(last_price)
    prices[0] = initial_price

    # Calculate initial trend using linear regression on last 30 days
    transition_days = 180  # Doubled transition period
    if isinstance(last_price, pd.Series) and len(last_price) >= 90:
        X = np.arange(90).reshape(-1, 1)
        y = last_price[-90:].values
        reg = optimize.minimize(
            lambda x: np.sum((y - (x[0] * X.flatten() + x[1]))**2),
            [0, initial_price],
            method='Nelder-Mead'
        ).x
        initial_trend = reg[0]  # Daily price change
    else:
        initial_trend = 0

    # Smooth transition period (180 days instead of 90)
    for i in range(1, len(future_df)):
        if i < transition_days:
            # Calculate base price trend
            trend_price = initial_price + (initial_trend * i)
            power_law_price = future_center[i]
            
            # Calculate price difference to bridge
            price_diff = power_law_price - trend_price
            
            # Use Weierstrass to create oscillating bridge between prices
            blend_factor = 0.5 * (1 - np.cos(np.pi * i / transition_days))
            weierstrass_component = w[i] * price_diff * blend_factor
            
            # Combine trend with Weierstrass oscillations
            prices[i] = trend_price + weierstrass_component
        else:
            # More power law influence but slightly more volatility
            base_price = future_center[i]
            osc = w[i] * 0.55  # Increased from 0.5
            amplitude = 0.44 * base_price  # Increased from 0.4
            prices[i] = base_price + osc * amplitude

        # Ensure no negative prices and limit daily changes
        max_daily_change = 0.22  # Increased from 0.20
        if i > 0:
            min_price = prices[i-1] * (1 - max_daily_change)
            max_price = prices[i-1] * (1 + max_daily_change)
            prices[i] = np.clip(prices[i], min_price, max_price)
    
    future_df['Price'] = prices
    future_df['CAGR'] = (future_df['Price'] / future_df['Price'].shift(365)) ** (1 / 1) - 1
    
    return future_df
