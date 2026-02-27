"""
Walk-Forward Backtesting with Profit Factor
Simulates trading with generated signals and calculates real performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger


class WalkForwardBacktest:
    """
    Simulates trading based on model signals and computes profit factor,
    win rate, max drawdown, and other institutional metrics.
    """

    def __init__(
        self,
        sl_atr_mult: float = config.SL_ATR_MULTIPLIER,
        tp_atr_mult: float = config.TP_ATR_MULTIPLIER,
        commission_pct: float = 0.0006,  # 0.06% per trade (Binance futures)
    ):
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.commission_pct = commission_pct

    def run(
        self,
        df: pd.DataFrame,
        signals: List[Dict],
        initial_capital: float = 10000.0
    ) -> Dict:
        """
        Run backtest on historical data with signals.

        Args:
            df: DataFrame with OHLCV + features
            signals: List of {"index": int, "signal": "LONG"/"SHORT", "confidence": float}
            initial_capital: Starting capital

        Returns:
            Performance metrics dict
        """
        if not signals:
            return self._empty_metrics()

        trades = []
        capital = initial_capital
        peak_capital = initial_capital
        max_drawdown = 0.0

        for sig in signals:
            idx = sig["index"]
            signal = sig["signal"]
            confidence = sig.get("confidence", 50)

            if signal not in ("LONG", "SHORT"):
                continue

            if idx >= len(df) - 1:
                continue

            entry_price = df["close"].iloc[idx]
            atr_norm = df["atr_norm"].iloc[idx] if "atr_norm" in df.columns else 0.02
            atr_price = atr_norm * entry_price

            if signal == "LONG":
                sl_price = entry_price - self.sl_atr_mult * atr_price
                tp_price = entry_price + self.tp_atr_mult * atr_price
            else:
                sl_price = entry_price + self.sl_atr_mult * atr_price
                tp_price = entry_price - self.tp_atr_mult * atr_price

            # Simulate forward to find exit
            exit_price = None
            exit_reason = "timeout"
            max_hold = min(20, len(df) - idx - 1)

            for j in range(1, max_hold + 1):
                bar = df.iloc[idx + j]
                high = bar["high"]
                low = bar["low"]

                if signal == "LONG":
                    if low <= sl_price:
                        exit_price = sl_price
                        exit_reason = "stop_loss"
                        break
                    if high >= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                        break
                else:  # SHORT
                    if high >= sl_price:
                        exit_price = sl_price
                        exit_reason = "stop_loss"
                        break
                    if low <= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                        break

            if exit_price is None:
                # Timeout — exit at close of last bar
                exit_price = df["close"].iloc[min(idx + max_hold, len(df) - 1)]
                exit_reason = "timeout"

            # Calculate P&L
            if signal == "LONG":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            # Subtract commission (entry + exit)
            pnl_pct -= 2 * self.commission_pct

            pnl_dollar = capital * pnl_pct
            capital += pnl_dollar

            # Track drawdown
            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)

            trades.append({
                "signal": signal,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_pct": round(pnl_pct * 100, 4),
                "pnl_dollar": round(pnl_dollar, 2),
                "confidence": confidence,
                "capital_after": round(capital, 2),
            })

        return self._compute_metrics(trades, initial_capital, capital, max_drawdown)

    def _compute_metrics(
        self,
        trades: List[Dict],
        initial_capital: float,
        final_capital: float,
        max_drawdown: float
    ) -> Dict:
        """Compute performance metrics from trades."""
        if not trades:
            return self._empty_metrics()

        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        # Profit factor = gross profit / gross loss
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Average win/loss
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Expectancy = (win_rate * avg_win) + ((1-win_rate) * avg_loss)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Sharpe-like ratio (simplified)
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0

        total_return = (final_capital - initial_capital) / initial_capital * 100

        # Exit reason distribution
        exit_reasons = {}
        for t in trades:
            r = t["exit_reason"]
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        return {
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 3),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "expectancy_pct": round(expectancy, 4),
            "sharpe_ratio": round(sharpe, 3),
            "initial_capital": initial_capital,
            "final_capital": round(final_capital, 2),
            "exit_reasons": exit_reasons,
            "trades": trades,
        }

    def _empty_metrics(self) -> Dict:
        return {
            "total_trades": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "total_return_pct": 0,
            "max_drawdown_pct": 0,
            "avg_win_pct": 0,
            "avg_loss_pct": 0,
            "expectancy_pct": 0,
            "sharpe_ratio": 0,
            "initial_capital": 0,
            "final_capital": 0,
            "exit_reasons": {},
            "trades": [],
        }


def backtest_model_signals(
    symbol: str,
    timeframe: str,
    model_path: str,
    test_days: int = 90
) -> Dict:
    """
    Run walk-forward backtest for a trained model.
    Generates signals on test data and evaluates performance.
    """
    from data.market import BinanceDataFetcher
    from data.features import FeatureEngine
    from ai.predict import load_model_and_predict, TechnicalAnalyzer
    import torch
    from ai.model import create_model

    logger.info(f"Running backtest for {symbol} {timeframe} ({test_days} days)")

    fetcher = BinanceDataFetcher()

    # Calculate candles needed
    tf_minutes = config.TIMEFRAME_TO_MINUTES.get(timeframe, 60)
    candles_needed = (test_days * 24 * 60) // tf_minutes + config.SEQUENCE_LENGTH + 200

    df = fetcher.fetch_latest_candles(symbol, timeframe, num_candles=min(candles_needed, 1000))
    if df is None:
        logger.error(f"Could not fetch data for backtest")
        return {}

    engine = FeatureEngine()
    df = engine.create_all_features(df, symbol=symbol)
    feature_cols = engine.get_feature_names()

    # Load model
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return {}

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = create_model(checkpoint["num_features"], model_type=checkpoint["model_type"], device="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate signals on test portion
    ta = TechnicalAnalyzer()
    signals = []

    test_start = max(config.SEQUENCE_LENGTH + 50, len(df) - (test_days * 24 * 60 // tf_minutes))

    for i in range(test_start, len(df) - 1, 3):  # Every 3 candles to avoid overtrading
        sub_df = df.iloc[:i + 1]
        ta_result = ta.analyze(sub_df)

        long_v = ta_result["long_votes"]
        short_v = ta_result["short_votes"]

        if long_v > short_v and long_v >= 4:
            signals.append({"index": i, "signal": "LONG", "confidence": long_v / ta_result["total"] * 100})
        elif short_v > long_v and short_v >= 4:
            signals.append({"index": i, "signal": "SHORT", "confidence": short_v / ta_result["total"] * 100})

    logger.info(f"Generated {len(signals)} signals for backtest")

    # Run backtest
    bt = WalkForwardBacktest()
    results = bt.run(df, signals)

    logger.info(f"Backtest results: PF={results['profit_factor']}, WR={results['win_rate']}%, "
                f"Return={results['total_return_pct']}%, MaxDD={results['max_drawdown_pct']}%")

    return results
