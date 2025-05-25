import numpy as np
from datetime import datetime

def weierstrass_function(t, a=0.5, b=3, n_terms=10):
    """Generate Weierstrass-like function for mNAV volatility"""
    w = np.zeros_like(t, dtype=float)
    for n in range(n_terms):
        w += a**n * np.cos(np.pi * b**n * t)
    w = w / np.max(np.abs(w))
    return w

def multi_weierstrass_mnav(t, days_scale=60):  # Doubled the days_scale to lower frequency
    """Generate combined Weierstrass functions for mNAV"""
    configs = [
        {'a': 0.4, 'b': 2.5, 'n_terms': 8, 'weight': 0.5, 'scale': 1/days_scale},
        {'a': 0.6, 'b': 2.0, 'n_terms': 6, 'weight': 0.3, 'scale': 1/(days_scale*5)},  # Increased from 3 to 5
        {'a': 0.3, 'b': 3.0, 'n_terms': 4, 'weight': 0.2, 'scale': 1/(days_scale/2)}   # Reduced from 3 to 2
    ]
    
    wsum = np.zeros_like(t, dtype=float)
    for cfg in configs:
        w = weierstrass_function(
            t * cfg['scale'],
            a=cfg['a'],
            b=cfg['b'],
            n_terms=cfg['n_terms']
        )
        wsum += cfg['weight'] * w
    
    # Add safety check for division by zero
    max_abs = np.max(np.abs(wsum))
    if max_abs < 1e-10:  # If sum is effectively zero
        return np.zeros_like(t)
    return wsum / max_abs

def calculate_mnav_with_volatility(btc_value, days_from_start, base_volatility=0.12):
    """
    Calculates mNAV with enhanced volatility and mean reversion
    Returns: Float with calculated mNAV value including volatility
    """
    # Calculate power law baseline (theoretical fair value)
    theoretical_mcap = 35.1221 * (btc_value ** 0.89)
    power_law_mnav = theoretical_mcap / btc_value

    # Add random overshooting for mean reversion targets
    if np.random.random() < 0.15:  # 15% chance of setting new overshoot target
        # Generate asymmetric overshoots
        if np.random.random() < 0.5:  # Upside overshoot
            overshoot = np.random.uniform(1.33, 4.0)  # 33% to 100% upside (increased from 20%)
        else:  # Downside overshoot
            overshoot = np.random.uniform(0.5, 0.8)  # 20% to 50% downside (unchanged)
        target_mnav = power_law_mnav * overshoot
    else:
        target_mnav = power_law_mnav

    # Calculate current mNAV with enhanced volatility
    volatility = base_volatility * (1 + 0.5 * np.sin(days_from_start / 30))  # Cyclical volatility
    noise = np.random.normal(0, volatility)
    
    # Mean reversion strength varies randomly
    reversion_speed = np.random.uniform(0.05, 0.15)
    
    # Get previous mNAV (or use power law if first calculation)
    current_mnav = getattr(calculate_mnav_with_volatility, 'last_mnav', power_law_mnav)
    
    # Apply mean reversion with noise
    new_mnav = current_mnav + (target_mnav - current_mnav) * reversion_speed + noise
    
    # Ensure minimum mNAV floor
    min_mnav = power_law_mnav * 0.4  # Allow deeper downside
    new_mnav = max(min_mnav, new_mnav)
    
    # Store for next calculation
    calculate_mnav_with_volatility.last_mnav = new_mnav
    
    return new_mnav
