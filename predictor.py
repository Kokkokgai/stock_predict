"""
predictor.py
------------
Converts model output (probability) into actionable trading signals
(BUY / SELL / HOLD) and runs a simple backtesting simulation.

Signal logic:
  probability > BUY_THRESHOLD  → BUY
  probability < SELL_THRESHOLD → SELL
  otherwise                    → HOLD

The confidence score = how far the probability is from 0.5:
  confidence = |probability − 0.5| × 2   (scaled 0→1)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

# Default thresholds — tune these for your risk tolerance
BUY_THRESHOLD  = 0.58   # > 58% probability of going up  → BUY
SELL_THRESHOLD = 0.42   # < 42% probability of going up  → SELL


@dataclass
class PredictionResult:
    """Structured output for a single prediction."""
    ticker     : str
    date       : str
    signal     : str        # "BUY" | "SELL" | "HOLD"
    probability: float      # P(price goes UP next day)
    confidence : float      # how confident the model is (0–1)
    last_close : float      # most recent closing price


def probability_to_signal(prob: float,
                           buy_threshold : float = BUY_THRESHOLD,
                           sell_threshold: float = SELL_THRESHOLD) -> tuple[str, float]:
    """
    Convert a raw UP-probability into a trading signal + confidence score.

    Args:
        prob          : P(UP) in range [0, 1]
        buy_threshold : Minimum prob to trigger BUY
        sell_threshold: Maximum prob to trigger SELL

    Returns:
        (signal, confidence)  where confidence ∈ [0, 1]
    """
    # Confidence: distance from the 0.5 decision boundary, scaled to [0,1]
    confidence = min(abs(prob - 0.5) * 2, 1.0)

    if prob >= buy_threshold:
        return "BUY", confidence
    elif prob <= sell_threshold:
        return "SELL", confidence
    else:
        return "HOLD", confidence


def predict_latest(ticker: str, df_with_indicators: pd.DataFrame,
                   trained_model, forward_days: int = 1) -> PredictionResult:
    """
    Generate a signal for the MOST RECENT bar in the data.

    Args:
        ticker               : Stock symbol string.
        df_with_indicators   : DataFrame with OHLCV + all indicator columns.
        trained_model        : Fitted EnsembleModel instance.
        forward_days         : Not used for inference but kept for metadata.

    Returns:
        PredictionResult dataclass.
    """
    feat_cols = trained_model.feat_cols
    last_row  = df_with_indicators[feat_cols].dropna().iloc[[-1]]  # shape (1, n_features)

    prob       = float(trained_model.predict_proba(last_row.values)[0])
    signal, conf = probability_to_signal(prob)

    return PredictionResult(
        ticker      = ticker,
        date        = str(df_with_indicators.index[-1].date()),
        signal      = signal,
        probability = round(prob, 4),
        confidence  = round(conf, 4),
        last_close  = round(float(df_with_indicators["Close"].iloc[-1]), 2),
    )


# ──────────────────────────────────────────────────────────────────────────
# Backtesting
# ──────────────────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, trained_model,
             buy_threshold : float = BUY_THRESHOLD,
             sell_threshold: float = SELL_THRESHOLD,
             initial_capital: float = 10_000.0) -> pd.DataFrame:
    """
    Simple vectorised backtest: enter LONG on BUY, exit on SELL/HOLD.

    Rules:
    - Start fully in cash.
    - On BUY signal: invest all capital in the stock.
    - On SELL signal: liquidate position back to cash.
    - HOLD: maintain current position.
    - No short selling, no leverage, no transaction costs.

    Returns:
        results_df : DataFrame with daily equity curve + signals.
    """
    feat_cols = trained_model.feat_cols
    sub = df[feat_cols].dropna()

    # Align Close prices to feature rows
    close_prices = df.loc[sub.index, "Close"]

    # Get probabilities for the full history
    probs   = trained_model.predict_proba(sub.values)
    signals = [probability_to_signal(p, buy_threshold, sell_threshold)[0]
               for p in probs]

    # Simulate portfolio
    cash       = initial_capital
    shares     = 0.0
    equity     = []
    in_market  = False

    for i, (date, price) in enumerate(close_prices.items()):
        sig = signals[i]

        if sig == "BUY" and not in_market:
            # Buy as many whole shares as we can afford
            shares    = cash / price
            cash      = 0.0
            in_market = True

        elif sig == "SELL" and in_market:
            # Sell all shares
            cash      = shares * price
            shares    = 0.0
            in_market = False

        current_equity = cash + shares * price
        equity.append(current_equity)

    results = pd.DataFrame({
        "Date"    : close_prices.index,
        "Close"   : close_prices.values,
        "Signal"  : signals,
        "Prob_UP" : probs,
        "Equity"  : equity,
    }).set_index("Date")

    # ── Summary statistics ──────────────────────────────────────────────
    total_return  = (results["Equity"].iloc[-1] / initial_capital - 1) * 100
    buy_and_hold  = (results["Close"].iloc[-1]  / results["Close"].iloc[0]  - 1) * 100

    # Daily returns of strategy
    daily_ret     = results["Equity"].pct_change().dropna()
    sharpe        = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                     if daily_ret.std() > 0 else 0.0)

    # Max drawdown
    rolling_max   = results["Equity"].cummax()
    drawdown      = (results["Equity"] - rolling_max) / rolling_max
    max_drawdown  = drawdown.min() * 100

    n_trades = (results["Signal"] == "BUY").sum()

    print("\n[Backtest] ── Strategy Performance ────────────────────────────")
    print(f"           Period        : {results.index[0].date()} → {results.index[-1].date()}")
    print(f"           Strategy Ret  : {total_return:+.2f}%")
    print(f"           Buy & Hold    : {buy_and_hold:+.2f}%")
    print(f"           Sharpe Ratio  : {sharpe:.2f}")
    print(f"           Max Drawdown  : {max_drawdown:.2f}%")
    print(f"           # BUY entries : {n_trades}")

    return results
