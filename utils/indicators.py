# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:59:48 2026

@author: 0125i
"""

import pandas as pd

def compute_MA(close,fast=5,slow=20):
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    return ma_fast - ma_slow

def compute_KD(close, n=9):
    low_n = close.rolling(window=n, min_periods=n).min()
    high_n = close.rolling(window=n, min_periods=n).max()

    rsv = (close - low_n) / (high_n - low_n) * 100

    K = rsv.ewm(alpha=1/3, adjust=False).mean()
    D = K.ewm(alpha=1/3, adjust=False).mean()

    return K, D
    
def compute_RSI(close, n=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_MACD(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal

    return macd, macd_signal, macd_hist


