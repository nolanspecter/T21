import numpy as np
import pandas as pd
import ta

def get_sma(close, period):
    ma = close.rolling(period).mean()
    return pd.Series(ma)

def get_ema(close, period):
    """ This function takes closing price and period and returns exponential moving average"""
    close = close.squeeze()
    #calculate Multiplier (smoothing factor)
    multiplier = 2 / (period + 1)
    #calculate Simple Moving Average
    ma = get_sma(close, period)
    #initialize the Exponential Moving Average list
    ema = [np.NaN] * len(close)

    #iterate over the closing prices
    for index in range(len(close)):
        #check index equals the period calculate EMA
        if index == period:
            ema[index] = close.iloc[index] * multiplier + ma[index - 1] * (1 - multiplier)
        #check index is greater than the period calculate follwing EMA
        if index > period:
            ema[index] = close.iloc[index] * multiplier + ema[index - 1] * (1 - multiplier)

    return pd.Series(ema)

def get_macd(close, fast_period = 12, slow_period = 26, smoothing = 9):
    """ This function takes in closing price, a period of 12 and 26 and a smoothing value and returns"""

    #calculate MACD using the difference of EMA periods 12 and 26
    macd = get_ema(close, 12) - get_ema(close, 26)
    #create series for signal calculation
    empty_values = pd.Series([np.NaN]*slow_period)

    #calculate signal line using smoothing value
    signal_calc_values = macd.iloc[slow_period::].reset_index()
    signal_calc_values.drop(['index'], axis = 1, inplace = True)
    signal = pd.concat([empty_values, get_ema(signal_calc_values[0], smoothing)])
    signal = pd.Series(signal.reset_index().drop(['index'], axis = 1)[0])

    #calculate histogram
    histogram = macd - signal

    return macd, signal, histogram

def get_ichimoku_cloud(high, low, close, conversion_period = 9, base_period = 26, leading_period = 52):
    """ This function takes in price data for high, low, close and also conversion, base and leading periods 
    returning conversion and base line also leading and lagging spans"""
    #calculate conversion line using high and low prices of last 9 periods
    conversion_line = (high.rolling(conversion_period).max() + low.rolling(conversion_period).min()) / 2
    #calculate base line using high and low prices of last 26 periods
    base_line = (high.rolling(base_period).max() + low.rolling(base_period).min()) / 2

    #calculate leading span using high and low prices using period of 52
    leading_span_a = ((conversion_line + base_line) / 2).shift(base_period)
    leading_span_b = (high.rolling(leading_period).max() + low.rolling(leading_period).min()) / 2
    leading_span_b = leading_span_b.shift(base_period)

    #calculate lagging span by shifting using base period
    lagging_span = close.shift(-base_period)
    
    return conversion_line, base_line, leading_span_a, leading_span_b, lagging_span

def get_atr(data, period = 14):
    """calculate ATR using high, low, close data and period"""
    
    #separate price data
    high = data['High']
    low = data['Low']
    close = data['Close']

    #create empty series
    atr = pd.Series(0.0, index=data.index)
    #set ATR to closing value
    atr[0] = close[0]

    #for each remaining close data calculate ATR
    for i in range(1, len(close)):
        
        range1 = high[i] - low[i]
        range2 = abs(high[i] - close[i-1])
        range3 = abs(low[i] - close[i-1])
        #using the difference of prices true range takes max value of ranges
        true_range = max(range1, range2, range3)
        atr[i] = ((period - 1) * atr[i-1] + true_range) / period

    return atr

def get_donchian_channels(data, window_size = 20):
    """calculate donchain using closing data and n value of 20 parameters
    returns upper and lower channels and midline"""

    #separate price data
    close = data['Close']

    #calculate upperchannel using max value in window
    upper_channel = close.rolling(window=window_size).max()
    #calculate lowerchannel using min value in window
    lower_channel = close.rolling(window=window_size).min()
    #calculate midline using upper and lower channel /2
    mid_line = (upper_channel + lower_channel) / 2
    
    return upper_channel, lower_channel, mid_line

def get_bollinger_bands(close, window_size=20):
    """calculate bollinger bands using closing price parameters closing price, window size
    returns upper, lower and rolling mean for bollinger bands"""
    #calculate rolling mean of closing price
    rolling_mean = close.rolling(window_size).mean()
    #calculate standard deviation of closing price
    rolling_std = close.rolling(window_size).std()

    #calculate upper band using rolling mean and rolling standard deviation
    upper_band = rolling_mean + (rolling_std * 2)
    #calculate lower band using rolling mean and rolling standard deviation
    lower_band = rolling_mean - (rolling_std * 2)

    return upper_band, rolling_mean, lower_band

def get_rsi(close, period = 14):
    """
    Calculate the Relative Strength Index (RSI) for a given set of price data.
    """
    diff = close.diff(1)
    up_direction = diff.where(diff > 0, 0.0)
    down_direction = -diff.where(diff < 0, 0.0)
    emaup = up_direction.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    emadn = down_direction.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    relative_strength = emaup / emadn
    rsi = pd.Series(
        np.where(emadn == 0, 100, 100 - (100 / (1 + relative_strength))),
        index=close.index,
    )
    return rsi


