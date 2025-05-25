import numpy as np
from datetime import datetime

def weierstrass_function(t, a=0.5, b=3, n_terms=10):
    """Generate Weierstrass-like function for mNAV volatility"""
    w = np.zeros_like(t, dtype=float)
    for n in range(n_terms):
        w += a**n * np.cos(np.pi * b**n * t)
    w = w / np.max(np.abs(w))
    return w

def multi_weierstrass_mnav(t, days_scale=30):
    """Generate combined Weierstrass functions for mNAV"""
    configs = [
        {'a': 0.4, 'b': 2.5, 'n_terms': 8, 'weight': 0.5, 'scale': 1/days_scale},
        {'a': 0.6, 'b': 2.0, 'n_terms': 6, 'weight': 0.3, 'scale': 1/(days_scale*3)},
        {'a': 0.3, 'b': 3.0, 'n_terms': 4, 'weight': 0.2, 'scale': 1/(days_scale/3)}
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
    return wsum / np.max(np.abs(wsum))

def calculate_mnav_with_volatility(btc_value, days_from_start):
    """Calculate mNAV with Weierstrass volatility and mean reversion"""
    # Base power law mNAV calculation
    theoretical_mcap = 35.1221 * (btc_value ** 0.91)
    power_law_mnav = theoretical_mcap / btc_value
    
    # Add faster exponential decay to volatility
    decay_factor = np.exp(-0.0003 * days_from_start)  # Slightly faster decay
    
    # Add Weierstrass volatility with mean reversion
    t = np.array([days_from_start])
    base_volatility = multi_weierstrass_mnav(t)
    
    # Target mNAV of 3.0 with overshooting
    target_mnav = 3.0
    current_mnav = power_law_mnav
    
    # Mean reversion factor (0.1 = 10% reversion per step)
    reversion_speed = 0.1
    
    # Calculate mean reversion with overshooting
    if current_mnav > target_mnav:
        # Overshooting to the downside
        reversion = (target_mnav - current_mnav) * reversion_speed * (1.2 + base_volatility[0])
    else:
        # Overshooting to the upside
        reversion = (target_mnav - current_mnav) * reversion_speed * (1.3 + base_volatility[0])
    
    # Apply volatility and mean reversion
    final_mnav = current_mnav + (reversion * decay_factor)
    
    # Ensure mNAV never goes below 1.0
    return max(1.0, final_mnav)
