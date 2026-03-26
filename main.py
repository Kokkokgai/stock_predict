"""
main.py
-------
Command-line entry point for the Stock Prediction System.

Usage examples:
    python main.py                    # interactive prompt
    python main.py --ticker AAPL
    python main.py --ticker TSLA --period 3y --forward 3
    python main.py --ticker MSFT --no-backtest

Pipeline:
    1. Fetch historical OHLCV data       (data_loader)
    2. Compute technical indicators      (indicators)
    3. Train ensemble ML model           (model)
    4. Generate latest prediction signal (predictor)
    5. Run backtest                      (predictor)
    6. Generate charts                   (visualizer)
"""

import argparse
import sys
from pathlib import Path

# ── Local modules ──────────────────────────────────────────────────────────
from data_loader import fetch_stock_data, get_company_info
from indicators import add_all_indicators
from model import EnsembleModel
from predictor import predict_latest, backtest
from visualizer import plot_price_indicators, plot_backtest, plot_feature_importance


# ══════════════════════════════════════════════════════════════════════════
# Signal display helpers
# ══════════════════════════════════════════════════════════════════════════

SIGNAL_EMOJI = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}


def print_banner():
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║         📈  STOCK PREDICTION SYSTEM  (ML-powered)           ║
║             XGBoost + Random Forest Ensemble                 ║
╚══════════════════════════════════════════════════════════════╝
"""
    )


def print_prediction(result, info: dict):
    """Pretty-print the final prediction result."""
    bar_len = int(result.confidence * 20)
    conf_bar = "█" * bar_len + "░" * (20 - bar_len)
    signal_lbl = {"BUY": "BUY  ▲", "SELL": "SELL ▼", "HOLD": "HOLD ~"}[result.signal]
    signal_pfx = {"BUY": "[+]", "SELL": "[-]", "HOLD": "[~]"}[result.signal]

    W = 63  # total inner width (between │ and │)

    def row(label, value, width=W):
        line = f"  {label:<12}: {value}"
        return f"│{line:<{width}}│"

    sep = "├" + "─" * W + "┤"
    top = "┌" + "─" * W + "┐"
    bot = "└" + "─" * W + "┘"

    print()
    print(top)
    print(row("Company", info["name"][:48]))
    print(row("Ticker", f"{result.ticker}   Sector: {info['sector'][:30]}"))
    print(row("As of", result.date))
    print(row("Last Close", f"${result.last_close:,.2f}"))
    print(sep)
    print(row("SIGNAL", f"{signal_pfx} {signal_lbl}"))
    print(row("P(UP)", f"{result.probability:.1%}"))
    print(row("Confidence", f"[{conf_bar}] {result.confidence:.0%}"))
    print(sep)
    print(f"│  {'WHAT THIS MEANS':<{W-2}}│")

    if result.signal == "BUY":
        msg1 = f"  Model sees {result.probability:.0%} probability of upward move."
        msg2 = "  Consider entering a long position."
    elif result.signal == "SELL":
        msg1 = f"  Model sees only {result.probability:.0%} probability of upward move."
        msg2 = "  Consider reducing exposure or exiting a position."
    else:
        msg1 = f"  Model is uncertain ({result.probability:.0%} P(UP))."
        msg2 = "  Wait for a clearer signal before entering."

    print(f"│{msg1:<{W}}│")
    print(f"│{msg2:<{W}}│")
    print(f"│{'':>{W}}│")
    print(f"│  {'WARNING: Not financial advice. Do your own research.':<{W-2}}│")
    print(bot)


# ══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════


def run(
    ticker: str,
    period: str = "2y",
    forward_days: int = 1,
    run_backtest: bool = True,
    threshold_pct: float = 0.3,
):
    """
    Execute the full pipeline for a given ticker.

    Args:
        ticker        : e.g. "AAPL"
        period        : historical data window, e.g. "2y"
        forward_days  : prediction horizon in trading days
        run_backtest  : whether to run and chart the backtest
        threshold_pct : minimum % daily move to label as UP/DOWN (noise filter)
    """
    print_banner()

    # ── Step 1: Load data ─────────────────────────────────────────────
    info = get_company_info(ticker)
    print(f"Company: {info['name']} | Sector: {info['sector']}")

    df_raw = fetch_stock_data(ticker, period=period)

    # ── Step 2: Feature engineering ───────────────────────────────────
    print("\n[Step 2] Computing technical indicators ...")
    df = add_all_indicators(df_raw)

    # ── Step 3: Train model ───────────────────────────────────────────
    print("\n[Step 3] Training ensemble model ...")
    model = EnsembleModel(xgb_weight=0.6)
    model.train(df, forward_days=forward_days, threshold_pct=threshold_pct)

    # ── Step 4: Predict latest signal ─────────────────────────────────
    print("\n[Step 4] Generating prediction for latest bar ...")
    result = predict_latest(ticker, df, model, forward_days)
    print_prediction(result, info)

    # ── Step 5: Backtest ──────────────────────────────────────────────
    backtest_df = None
    if run_backtest:
        print("\n[Step 5] Running backtest ...")
        backtest_df = backtest(df, model)

    # ── Step 6: Charts ────────────────────────────────────────────────
    print("\n[Step 6] Generating charts ...")
    chart1 = plot_price_indicators(df, ticker, backtest_df)
    chart2 = plot_feature_importance(model, ticker)
    charts = [chart1, chart2]

    if backtest_df is not None:
        chart3 = plot_backtest(backtest_df, ticker)
        charts.append(chart3)

    print(f"\n✅  All done!  Charts saved to: {Path(chart1).parent}")
    return result, model, df, backtest_df, charts


# ══════════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ══════════════════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML-powered stock prediction system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker", type=str, default=None, help="Stock ticker, e.g. AAPL"
    )
    parser.add_argument(
        "--period", type=str, default="2y", help="Data history: 1y, 2y, 5y ..."
    )
    parser.add_argument("--forward", type=int, default=1, help="Days ahead to predict")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Min %% move to label UP/DOWN (0.0 = label every day)",
    )
    parser.add_argument(
        "--no-backtest", action="store_true", help="Skip backtesting step"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Interactive ticker prompt if not supplied as argument
    ticker = args.ticker
    if not ticker:
        ticker = input("\nEnter stock ticker (e.g. AAPL, TSLA, MSFT): ").strip().upper()
        if not ticker:
            print("No ticker provided. Defaulting to AAPL.")
            ticker = "AAPL"

    run(
        ticker=ticker,
        period=args.period,
        forward_days=args.forward,
        run_backtest=not args.no_backtest,
        threshold_pct=args.threshold,
    )
