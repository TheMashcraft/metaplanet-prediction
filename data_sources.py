import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pandas_datareader import data as pdr

def get_metaplanet_3350_data():
    """
    Pulls price and volume data for Metaplanet 3350 since April 1st 2024
    Returns: DataFrame with date, price, and volume in USD
    """
    start_date = "2024-04-01"
    ticker = "3350.T"

    # Get stock data in JPY
    data = yf.download(ticker, start=start_date)
    
    dates_as_int = (data.index - data.index[0]).days.values.reshape(-1, 1)
    volume_regression = np.polyfit(dates_as_int.flatten(), np.log(data['Volume'].values), 1)
    
    # Calculate trend for each date
    data_usd = pd.DataFrame(index=data.index)
    data_usd['Volume'] = data['Volume']
    data_usd['Volume_Trend'] = np.exp(volume_regression[1] + volume_regression[0] * dates_as_int.flatten())
    data_usd['Close'] = data['Close'].div(150)
    # Store growth rate as column with same length as data
    data_usd['Volume_Growth'] = pd.Series([volume_regression[0]] * len(data), index=data.index)
    
    return data_usd

def get_bitcoin_historical_data():
    """Pulls Bitcoin price and volume data since 2012"""
    ticker = "BTC-USD"
    start_date = "2024-04-01"
    data = yf.download(ticker, start=start_date)
    return data[['Close', 'Volume']]  # Return simple subset instead of creating new DataFrame
