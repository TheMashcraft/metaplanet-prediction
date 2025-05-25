import numpy as np
from datetime import datetime

def weierstrass_function(t, a=0.5, b=3, n_terms=10):
    """Generate Weierstrass-like function for mNAV volatility"""
    w = np.zeros_like(t, dtype=float)
    for n in range(n_terms):
        w += a**n * np.cos(np.pi * b**n * t)
    w = w / np.max(np.abs(w))
    return w

def multi_weierstrass_mnav(t, days_scale=365):
    """Generate combined Weierstrass functions for mNAV"""
    configs = [
        {'a': 0.4, 'b': 2.5, 'n_terms': 8, 'weight': 0.5, 'scale': 1/days_scale},
        {'a': 0.6, 'b': 2.0, 'n_terms': 6, 'weight': 0.3, 'scale': 1/(days_scale*2)},
        {'a': 0.3, 'b': 3.0, 'n_terms': 4, 'weight': 0.2, 'scale': 1/(days_scale/2)}
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
    """Calculate mNAV with Weierstrass volatility"""
    # Base power law mNAV calculation
    theoretical_mcap = 35.1221 * (btc_value ** 0.91)
    power_law_mnav = theoretical_mcap / btc_value
    
    # Add Weierstrass volatility
    t = np.array([days_from_start])
    volatility = multi_weierstrass_mnav(t)
    
    # Combine with 33% weight for Weierstrass
    final_mnav = (0.67 * power_law_mnav) + (0.33 * power_law_mnav * (1 + volatility[0]))
    
    return final_mnav
