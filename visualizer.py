"""
visualizer.py
-------------
Generates three publication-quality charts saved as PNG files:

  1. price_indicators.png  — Price + Bollinger Bands + Volume + RSI + MACD
  2. backtest_equity.png   — Strategy equity curve vs. Buy-and-Hold
  3. feature_importance.png — Top-15 XGBoost feature importances
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path

CHART_DIR = Path("./charts")
CHART_DIR.mkdir(exist_ok=True)

# Colour palette
C_UP = "#26a69a"  # teal
C_DOWN = "#ef5350"  # red
C_HOLD = "#ffa726"  # amber
C_LINE = "#1565c0"  # deep blue
C_BAND = "#90caf9"  # light blue
C_VOL = "#78909c"  # slate
BG = "#0d1117"  # dark background
GRID = "#1f2937"


def _apply_dark_style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.spines[:].set_color("#374151")
    ax.grid(color=GRID, linewidth=0.5, alpha=0.8)


def plot_price_indicators(
    df: pd.DataFrame, ticker: str, backtest_df: pd.DataFrame = None
) -> str:
    """
    4-panel chart:
      Panel 1: Close price + Bollinger Bands + SMA_20 + BUY/SELL markers
      Panel 2: Volume bars
      Panel 3: RSI with overbought/oversold zones
      Panel 4: MACD histogram + signal line
    """
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 12),
        gridspec_kw={"height_ratios": [4, 1.2, 1.5, 1.5]},
        facecolor=BG,
    )
    fig.suptitle(
        f"{ticker} — Price, Indicators & Signals",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    ax_price, ax_vol, ax_rsi, ax_macd = axes
    dates = df.index

    # ── Panel 1: Price + BB + SMA ───────────────────────────────────────
    _apply_dark_style(ax_price)
    ax_price.plot(
        dates, df["Close"], color=C_LINE, linewidth=1.2, label="Close", zorder=3
    )
    if "BB_Upper" in df.columns:
        ax_price.plot(
            dates,
            df["BB_Upper"],
            color=C_BAND,
            linewidth=0.8,
            alpha=0.7,
            label="BB Upper",
        )
        ax_price.plot(
            dates,
            df["BB_Lower"],
            color=C_BAND,
            linewidth=0.8,
            alpha=0.7,
            label="BB Lower",
        )
        ax_price.fill_between(
            dates, df["BB_Upper"], df["BB_Lower"], alpha=0.08, color=C_BAND
        )
    if "SMA_20" in df.columns:
        ax_price.plot(
            dates,
            df["SMA_20"],
            color="#f59e0b",
            linewidth=0.9,
            linestyle="--",
            label="SMA 20",
            alpha=0.85,
        )
    if "EMA_12" in df.columns:
        ax_price.plot(
            dates,
            df["EMA_12"],
            color="#a78bfa",
            linewidth=0.9,
            linestyle=":",
            label="EMA 12",
            alpha=0.85,
        )

    # Overlay BUY/SELL signals from backtest
    if backtest_df is not None:
        shared_idx = df.index.intersection(backtest_df.index)
        bt = backtest_df.loc[shared_idx]
        buy_mask = bt["Signal"] == "BUY"
        sell_mask = bt["Signal"] == "SELL"

        if buy_mask.any():
            ax_price.scatter(
                bt.index[buy_mask],
                df.loc[bt.index[buy_mask], "Close"],
                marker="^",
                color=C_UP,
                s=50,
                zorder=5,
                label="BUY",
            )
        if sell_mask.any():
            ax_price.scatter(
                bt.index[sell_mask],
                df.loc[bt.index[sell_mask], "Close"],
                marker="v",
                color=C_DOWN,
                s=50,
                zorder=5,
                label="SELL",
            )

    ax_price.set_ylabel("Price (USD)", color="#9ca3af", fontsize=9)
    ax_price.legend(
        loc="upper left",
        fontsize=7.5,
        facecolor="#1f2937",
        labelcolor="white",
        framealpha=0.8,
    )
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # ── Panel 2: Volume ─────────────────────────────────────────────────
    _apply_dark_style(ax_vol)
    colors = [C_UP if c >= o else C_DOWN for c, o in zip(df["Close"], df["Open"])]
    ax_vol.bar(dates, df["Volume"] / 1e6, color=colors, alpha=0.7, width=1)
    if "Vol_MA20" in df.columns:
        ax_vol.plot(
            dates,
            df["Vol_MA20"] / 1e6,
            color="#fbbf24",
            linewidth=0.9,
            label="Vol MA20",
        )
    ax_vol.set_ylabel("Volume (M)", color="#9ca3af", fontsize=9)

    # ── Panel 3: RSI ─────────────────────────────────────────────────────
    _apply_dark_style(ax_rsi)
    ax_rsi.plot(dates, df["RSI"], color="#f472b6", linewidth=1.0)
    ax_rsi.axhline(70, color=C_DOWN, linewidth=0.7, linestyle="--", alpha=0.8)
    ax_rsi.axhline(30, color=C_UP, linewidth=0.7, linestyle="--", alpha=0.8)
    ax_rsi.fill_between(
        dates, df["RSI"], 70, where=df["RSI"] >= 70, alpha=0.2, color=C_DOWN
    )
    ax_rsi.fill_between(
        dates, df["RSI"], 30, where=df["RSI"] <= 30, alpha=0.2, color=C_UP
    )
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI", color="#9ca3af", fontsize=9)
    ax_rsi.text(dates[-1], 72, "Overbought", color=C_DOWN, fontsize=7, ha="right")
    ax_rsi.text(dates[-1], 25, "Oversold", color=C_UP, fontsize=7, ha="right")

    # ── Panel 4: MACD ───────────────────────────────────────────────────
    _apply_dark_style(ax_macd)
    if "MACD" in df.columns:
        hist_colors = [C_UP if v >= 0 else C_DOWN for v in df["MACD_Hist"]]
        ax_macd.bar(
            dates,
            df["MACD_Hist"],
            color=hist_colors,
            alpha=0.7,
            width=1,
            label="Histogram",
        )
        ax_macd.plot(dates, df["MACD"], color="#38bdf8", linewidth=0.9, label="MACD")
        ax_macd.plot(
            dates, df["MACD_Signal"], color="#fb923c", linewidth=0.9, label="Signal"
        )
        ax_macd.axhline(0, color="#6b7280", linewidth=0.5)
    ax_macd.set_ylabel("MACD", color="#9ca3af", fontsize=9)
    ax_macd.legend(
        loc="upper left",
        fontsize=7,
        facecolor="#1f2937",
        labelcolor="white",
        framealpha=0.8,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = str(CHART_DIR / f"{ticker}_price_indicators.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[Viz] Saved → {out}")
    return out


def plot_backtest(
    backtest_df: pd.DataFrame, ticker: str, initial_capital: float = 10_000.0
) -> str:
    """
    Equity curve of the strategy vs. Buy-and-Hold benchmark.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1]}, facecolor=BG
    )
    fig.suptitle(
        f"{ticker} — Backtest: Strategy vs. Buy & Hold",
        color="white",
        fontsize=13,
        fontweight="bold",
    )

    # Normalise both curves to initial_capital
    bh = initial_capital * backtest_df["Close"] / backtest_df["Close"].iloc[0]

    _apply_dark_style(ax1)
    ax1.plot(
        backtest_df.index,
        backtest_df["Equity"],
        color=C_UP,
        linewidth=1.4,
        label="ML Strategy",
    )
    ax1.plot(
        backtest_df.index,
        bh,
        color=C_BAND,
        linewidth=1.2,
        linestyle="--",
        alpha=0.9,
        label="Buy & Hold",
    )
    ax1.axhline(initial_capital, color="#6b7280", linewidth=0.6, linestyle=":")
    ax1.fill_between(
        backtest_df.index,
        backtest_df["Equity"],
        bh,
        where=backtest_df["Equity"] >= bh,
        alpha=0.12,
        color=C_UP,
        label="Outperform",
    )
    ax1.fill_between(
        backtest_df.index,
        backtest_df["Equity"],
        bh,
        where=backtest_df["Equity"] < bh,
        alpha=0.12,
        color=C_DOWN,
        label="Underperform",
    )
    ax1.set_ylabel("Portfolio Value (USD)", color="#9ca3af", fontsize=9)
    ax1.legend(fontsize=8, facecolor="#1f2937", labelcolor="white")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Drawdown panel
    _apply_dark_style(ax2)
    rolling_max = backtest_df["Equity"].cummax()
    drawdown = (backtest_df["Equity"] - rolling_max) / rolling_max * 100
    ax2.fill_between(backtest_df.index, drawdown, 0, color=C_DOWN, alpha=0.6)
    ax2.set_ylabel("Drawdown %", color="#9ca3af", fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = str(CHART_DIR / f"{ticker}_backtest.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[Viz] Saved → {out}")
    return out


def plot_feature_importance(model, ticker: str, top_n: int = 15) -> str:
    """Horizontal bar chart of XGBoost feature importances."""
    imp = model.get_feature_importance().head(top_n)[::-1]  # ascending for barh

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    _apply_dark_style(ax)

    bars = ax.barh(imp.index, imp.values, color=C_LINE, alpha=0.85)
    # Colour top bar
    bars[-1].set_color(C_UP)

    ax.set_xlabel("Feature Importance (gain)", color="#9ca3af", fontsize=9)
    ax.set_title(
        f"{ticker} — Top {top_n} Features",
        color="white",
        fontsize=11,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=8)

    for bar, val in zip(bars, imp.values):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            color="#d1d5db",
            fontsize=7,
        )

    plt.tight_layout()
    out = str(CHART_DIR / f"{ticker}_feature_importance.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[Viz] Saved → {out}")
    return out
