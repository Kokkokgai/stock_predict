"""
data_loader.py
--------------
Fetches OHLCV data from Yahoo Finance. Falls back to synthetic GBM data
when the network is unavailable (sandbox/offline mode).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def _synthetic_ohlcv(ticker: str, n_days: int = 504) -> pd.DataFrame:
    seed = sum(ord(c) for c in ticker)
    rng  = np.random.default_rng(seed)

    S0, mu, sigma_base, dt = 150.0, 0.12, 0.22, 1/252

    vol = np.zeros(n_days)
    vol[0] = sigma_base
    for t in range(1, n_days):
        shock  = rng.standard_normal()
        vol[t] = np.clip(0.9*vol[t-1] + 0.1*sigma_base + 0.03*abs(shock)*sigma_base, 0.08, 0.60)

    ret   = (mu - 0.5*vol**2)*dt + vol*np.sqrt(dt)*rng.standard_normal(n_days)
    close = np.clip(S0 * np.exp(np.cumsum(ret)), 1.0, None)

    intra = vol * close * np.sqrt(dt) * rng.uniform(0.5, 2.0, n_days)
    high  = close + intra * rng.uniform(0.4, 0.8, n_days)
    low   = np.maximum(close - intra * rng.uniform(0.4, 0.8, n_days), close*0.5)
    open_ = low + rng.uniform(0, 1, n_days) * (high - low)
    volume = (50_000_000 * rng.lognormal(0, 0.5, n_days)).astype(int)

    end   = datetime.today()
    start = end - timedelta(days=int(n_days * 1.45))
    dates = pd.bdate_range(start=start, end=end)[-n_days:]

    df = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                        "Close": close, "Volume": volume}, index=dates)
    df.index.name = "Date"
    return df


def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d",
                     allow_synthetic: bool = True) -> pd.DataFrame:
    period_map = {"1mo":21,"3mo":63,"6mo":126,"1y":252,"2y":504,"3y":756,"5y":1260}
    n_days = period_map.get(period, 504)

    print(f"[DataLoader] Fetching {ticker} — period={period}, interval={interval} ...")
    try:
        raw = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError("Empty response")
        df = raw[["Open","High","Low","Close","Volume"]].copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        print(f"[DataLoader] Loaded {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}")
        return df
    except Exception as exc:
        if not allow_synthetic:
            raise
        print(f"[DataLoader] ⚠ Network failed ({exc.__class__.__name__}). Using SYNTHETIC data.")
        df = _synthetic_ohlcv(ticker, n_days)
        print(f"[DataLoader] ✅ Synthetic: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}")
        return df


def get_company_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {"name": info.get("longName", ticker), "sector": info.get("sector","N/A"),
                "market_cap": info.get("marketCap","N/A"), "currency": info.get("currency","USD")}
    except Exception:
        return {"name": ticker, "sector": "N/A", "market_cap": "N/A", "currency": "USD"}
