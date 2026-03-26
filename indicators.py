"""
indicators.py
-------------
Computes technical indicators that are used as features for the ML model.

Indicators implemented:
  - SMA  : Simple Moving Average
  - EMA  : Exponential Moving Average
  - RSI  : Relative Strength Index  (momentum)
  - MACD : Moving Average Convergence Divergence (trend)
  - Bollinger Bands                              (volatility)
  - OBV  : On-Balance Volume                    (volume)
  - ATR  : Average True Range                   (volatility)
  - ROC  : Rate of Change                       (momentum)
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# Moving Averages
# ──────────────────────────────────────────────

def add_sma(df: pd.DataFrame, windows: list[int] = [10, 20, 50]) -> pd.DataFrame:
    """Simple Moving Average over multiple window sizes."""
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_ema(df: pd.DataFrame, windows: list[int] = [12, 26]) -> pd.DataFrame:
    """Exponential Moving Average — gives more weight to recent prices."""
    for w in windows:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df


# ──────────────────────────────────────────────
# Momentum Indicators
# ──────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index (0–100).
    > 70  → overbought (potential SELL signal)
    < 30  → oversold   (potential BUY  signal)
    """
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_roc(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Rate of Change — % price change over 'period' bars."""
    df["ROC"] = df["Close"].pct_change(periods=period) * 100
    return df


# ──────────────────────────────────────────────
# Trend Indicators
# ──────────────────────────────────────────────

def add_macd(df: pd.DataFrame,
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD = EMA(fast) − EMA(slow)
    Signal line = EMA(MACD, signal)
    Histogram  = MACD − Signal
    """
    ema_fast   = df["Close"].ewm(span=fast,   adjust=False).mean()
    ema_slow   = df["Close"].ewm(span=slow,   adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    return df


# ──────────────────────────────────────────────
# Volatility Indicators
# ──────────────────────────────────────────────

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands around a 20-day SMA ± 2 standard deviations.
    BB_Width = (Upper − Lower) / Middle  — measures volatility.
    BB_Pct   = (Close − Lower) / (Upper − Lower)  — where price sits in the band.
    """
    middle = df["Close"].rolling(window=window).mean()
    std    = df["Close"].rolling(window=window).std()

    df["BB_Upper"]  = middle + num_std * std
    df["BB_Middle"] = middle
    df["BB_Lower"]  = middle - num_std * std
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Pct"]    = (df["Close"] - df["BB_Lower"]) / (
                       df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average True Range — measures market volatility.
    TR = max(High−Low, |High−PrevClose|, |Low−PrevClose|)
    """
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=period, adjust=False).mean()
    return df


# ──────────────────────────────────────────────
# Volume Indicators
# ──────────────────────────────────────────────

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    On-Balance Volume — running total of volume with sign based on price direction.
    Rising OBV with rising price confirms an uptrend.
    """
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (direction * df["Volume"]).cumsum()
    return df


def add_volume_ma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume relative to its own moving average (ratio > 1 = above-average volume)."""
    df["Vol_MA20"]    = df["Volume"].rolling(window=window).mean()
    df["Vol_Ratio"]   = df["Volume"] / df["Vol_MA20"].replace(0, np.nan)
    return df


# ──────────────────────────────────────────────
# Price-derived features
# ──────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Daily return, candle body size, upper/lower wick, gap from open."""
    df["Daily_Return"]   = df["Close"].pct_change()
    df["HL_Range"]       = (df["High"] - df["Low"]) / df["Close"]   # candle height
    df["Body_Size"]      = (df["Close"] - df["Open"]).abs() / df["Close"]
    df["Upper_Wick"]     = (df["High"]  - df[["Close", "Open"]].max(axis=1)) / df["Close"]
    df["Lower_Wick"]     = (df[["Close", "Open"]].min(axis=1) - df["Low"]) / df["Close"]
    return df


# ──────────────────────────────────────────────
# Master function
# ──────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator and drop rows with NaN (warm-up period)."""
    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_roc(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_volume_ma(df)
    df = add_price_features(df)
    df.dropna(inplace=True)
    print(f"[Indicators] Feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    return df
