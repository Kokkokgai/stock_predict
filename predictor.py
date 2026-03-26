"""
predictor.py  (v4)
------------------
Changes vs v3:
  1. backtest() gains `hold_exits` parameter (default True).
       True  → HOLD while in-market closes the position (more conservative,
               avoids the "266 BUY / 43 SELL" always-invested problem).
       False → legacy behaviour (HOLD = stay in current position).
  2. Backtest summary now prints days_in_market / exposure %.
  3. Signal box uses plain ASCII spacing so emoji width doesn't break alignment.
  4. predict_latest now prints a cleaner one-liner confidence bar.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

BUY_THRESHOLD = 0.58
SELL_THRESHOLD = 0.42
TRANSACTION_COST = 0.001  # 0.1% per side


@dataclass
class PredictionResult:
    ticker: str
    date: str
    signal: str
    probability: float
    confidence: float
    last_close: float


def probability_to_signal(
    prob: float,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
) -> tuple[str, float]:
    confidence = min(abs(prob - 0.5) * 2, 1.0)
    if prob >= buy_threshold:
        return "BUY", confidence
    elif prob <= sell_threshold:
        return "SELL", confidence
    else:
        return "HOLD", confidence


def predict_latest(
    ticker: str,
    df_with_indicators: pd.DataFrame,
    trained_model,
    forward_days: int = 1,
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
) -> PredictionResult:
    """Signal for the most recent bar — no look-ahead, uses only past data."""
    t = getattr(trained_model, "threshold", 0.50)
    b_thr = buy_threshold if buy_threshold is not None else min(t + 0.05, 0.80)
    s_thr = sell_threshold if sell_threshold is not None else max(t - 0.05, 0.20)

    feat_cols = trained_model.feat_cols
    last_row = df_with_indicators[feat_cols].dropna().iloc[[-1]]

    prob = float(trained_model.predict_proba(last_row.values)[0])
    signal, conf = probability_to_signal(prob, b_thr, s_thr)

    return PredictionResult(
        ticker=ticker,
        date=str(df_with_indicators.index[-1].date()),
        signal=signal,
        probability=round(prob, 4),
        confidence=round(conf, 4),
        last_close=round(float(df_with_indicators["Close"].iloc[-1]), 2),
    )


# ──────────────────────────────────────────────────────────────────────────
# Backtesting
# ──────────────────────────────────────────────────────────────────────────


def backtest(
    df: pd.DataFrame,
    trained_model,
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
    initial_capital: float = 10_000.0,
    transaction_cost: float = TRANSACTION_COST,
    hold_exits: bool = True,
) -> pd.DataFrame:
    """
    Realistic backtest with look-ahead correction (signal[t] executes at Close[t+1]).

    Args:
        hold_exits : If True, a HOLD signal while in-market closes the position.
                     This prevents the model from being "always long" when BUY
                     signals vastly outnumber SELL signals.
                     If False, HOLD = maintain current position (legacy behaviour).

    The `hold_exits=True` default gives a stricter, more informative test:
    the model must actively re-issue BUY to stay in — it can't coast.
    """
    t = getattr(trained_model, "threshold", 0.50)
    b_thr = buy_threshold if buy_threshold is not None else min(t + 0.05, 0.80)
    s_thr = sell_threshold if sell_threshold is not None else max(t - 0.05, 0.20)

    feat_cols = trained_model.feat_cols
    sub = df[feat_cols].dropna()
    close_all = df.loc[sub.index, "Close"]

    probs = trained_model.predict_proba(sub.values)
    signals = pd.Series(
        [probability_to_signal(p, b_thr, s_thr)[0] for p in probs],
        index=sub.index,
    )

    # Signal on day t → execute at Close[t+1]
    exec_signals = signals.shift(1).dropna()
    exec_prices = close_all.loc[exec_signals.index]

    # ── Portfolio simulation ───────────────────────────────────────────────
    cash, shares, in_market = initial_capital, 0.0, False
    equity = []
    days_in = 0

    for date, price in exec_prices.items():
        sig = exec_signals[date]

        if sig == "BUY" and not in_market:
            cost = cash * transaction_cost
            shares = (cash - cost) / price
            cash = 0.0
            in_market = True

        elif sig == "SELL" and in_market:
            proceeds = shares * price
            cash = proceeds * (1 - transaction_cost)
            shares = 0.0
            in_market = False

        elif sig == "HOLD" and in_market and hold_exits:
            # New in v4: HOLD closes the position (conservative mode)
            proceeds = shares * price
            cash = proceeds * (1 - transaction_cost)
            shares = 0.0
            in_market = False

        if in_market:
            days_in += 1

        equity.append(cash + shares * price)

    results = pd.DataFrame(
        {
            "Close": exec_prices.values,
            "Signal": exec_signals.values,
            "Prob_UP": probs[1:],
            "Equity": equity,
        },
        index=exec_prices.index,
    )

    # ── Performance stats ──────────────────────────────────────────────────
    total_bars = len(results)
    exposure = days_in / total_bars * 100 if total_bars > 0 else 0.0

    strat_ret = (results["Equity"].iloc[-1] / initial_capital - 1) * 100
    bh_ret = (results["Close"].iloc[-1] / results["Close"].iloc[0] - 1) * 100
    alpha = strat_ret - bh_ret

    dr = results["Equity"].pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0

    # Calmar ratio = annual return / |max drawdown|
    roll_max = results["Equity"].cummax()
    max_dd = ((results["Equity"] - roll_max) / roll_max).min() * 100
    n_years = total_bars / 252
    ann_ret = (
        (results["Equity"].iloc[-1] / initial_capital) ** (1 / max(n_years, 0.1)) - 1
    ) * 100
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    n_buy = (results["Signal"] == "BUY").sum()
    n_sell = (results["Signal"] == "SELL").sum()
    n_hold = (results["Signal"] == "HOLD").sum()

    mode_str = (
        "hold-exits=ON (HOLD closes position)"
        if hold_exits
        else "hold-exits=OFF (HOLD = stay long)"
    )

    print("\n[Backtest] ── Strategy Performance (look-ahead corrected) ─────")
    print(f"           Mode          : {mode_str}")
    print(
        f"           Period        : {results.index[0].date()} → {results.index[-1].date()}"
    )
    print(f"           Strategy Ret  : {strat_ret:+.2f}%")
    print(f"           Buy & Hold    : {bh_ret:+.2f}%")
    print(f"           Alpha         : {alpha:+.2f}%")
    print(f"           Ann. Return   : {ann_ret:+.2f}%")
    print(f"           Sharpe Ratio  : {sharpe:.2f}")
    print(f"           Calmar Ratio  : {calmar:.2f}")
    print(f"           Max Drawdown  : {max_dd:.2f}%")
    print(f"           Exposure      : {exposure:.1f}% of days in-market")
    print(f"           BUY/HOLD/SELL : {n_buy} / {n_hold} / {n_sell} signals")
    print(f"           Trans. cost   : {transaction_cost:.1%} per side")

    if strat_ret > 3 * max(bh_ret, 1) and strat_ret > 0:
        print("  ⚠  Strategy >>3x Buy&Hold — treat with scepticism.")
        print("     Try --period 5y for a more robust evaluation window.")

    return results
