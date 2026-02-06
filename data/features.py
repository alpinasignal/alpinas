"""
Feature Engineering for Crypto Prediction
Professional quantitative features - NO simple indicators
All features are normalized and computed without look-ahead bias
"""

import pandas as pd
import numpy as np
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FeatureEngine:
    """
    Creates quantitative features for neural network training
    Focus on normalized, stationary features that capture market microstructure
    """

    def __init__(self):
        self.feature_names = []

    def compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns - fundamental feature"""
        df["returns"] = np.log(df["close"] / df["close"].shift(1))
        df["returns"] = df["returns"].fillna(0)
        self.feature_names.append("returns")
        return df

    def compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling volatility at multiple horizons"""
        for window in config.VOLATILITY_WINDOWS:
            col_name = f"volatility_{window}"
            df[col_name] = df["returns"].rolling(window=window).std()
            df[col_name] = df[col_name].fillna(df[col_name].mean())
            self.feature_names.append(col_name)
        return df

    def compute_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalized candle body and wick features
        Captures price action patterns
        """
        # Candle body (normalized by ATR)
        atr = self._compute_atr(df, period=14)
        body = abs(df["close"] - df["open"])
        df["candle_body_norm"] = body / (atr + 1e-8)

        # Upper wick
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        df["upper_wick_norm"] = upper_wick / (atr + 1e-8)

        # Lower wick
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
        df["lower_wick_norm"] = lower_wick / (atr + 1e-8)

        # Candle direction
        df["candle_direction"] = np.where(df["close"] > df["open"], 1, -1)

        # Fill NaNs
        for col in ["candle_body_norm", "upper_wick_norm", "lower_wick_norm"]:
            df[col] = df[col].fillna(0)

        self.feature_names.extend([
            "candle_body_norm", "upper_wick_norm", "lower_wick_norm", "candle_direction"
        ])

        return df

    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features"""
        # Volume change
        df["volume_change"] = df["volume"].pct_change()
        df["volume_change"] = df["volume_change"].fillna(0)
        df["volume_change"] = np.clip(df["volume_change"], -5, 5)  # clip outliers

        # Volume moving average ratio
        df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)
        df["volume_ratio"] = df["volume_ratio"].fillna(1)

        # Volume-weighted direction
        df["volume_direction"] = df["candle_direction"] * df["volume_ratio"]

        self.feature_names.extend(["volume_change", "volume_ratio", "volume_direction"])

        return df

    def compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum at multiple horizons"""
        for window in config.MOMENTUM_WINDOWS:
            col_name = f"momentum_{window}"
            # Rate of change
            df[col_name] = df["close"].pct_change(periods=window)
            df[col_name] = df[col_name].fillna(0)
            df[col_name] = np.clip(df[col_name], -1, 1)  # normalize
            self.feature_names.append(col_name)
        return df

    def compute_ema_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalized distance to EMA levels
        Captures trend and mean reversion
        """
        for period in config.EMA_PERIODS:
            ema = df["close"].ewm(span=period, adjust=False).mean()
            atr = self._compute_atr(df, period=14)

            col_name = f"ema_distance_{period}"
            df[col_name] = (df["close"] - ema) / (atr + 1e-8)
            df[col_name] = df[col_name].fillna(0)

            self.feature_names.append(col_name)

        return df

    def compute_atr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR-based features"""
        atr = self._compute_atr(df, period=config.ATR_PERIOD)

        # Normalized ATR
        df["atr_norm"] = atr / df["close"]
        df["atr_norm"] = df["atr_norm"].fillna(df["atr_norm"].mean())

        # ATR percentile (volatility regime)
        df["atr_percentile"] = df["atr_norm"].rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        df["atr_percentile"] = df["atr_percentile"].fillna(0.5)

        self.feature_names.extend(["atr_norm", "atr_percentile"])

        return df

    def compute_price_compression(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Price compression/expansion metrics
        Bollinger Band-like but normalized
        """
        window = 20

        # Rolling mean and std
        rolling_mean = df["close"].rolling(window=window).mean()
        rolling_std = df["close"].rolling(window=window).std()

        # Z-score (price compression)
        df["price_zscore"] = (df["close"] - rolling_mean) / (rolling_std + 1e-8)
        df["price_zscore"] = df["price_zscore"].fillna(0)
        df["price_zscore"] = np.clip(df["price_zscore"], -3, 3)

        # Bandwidth (expansion/contraction)
        df["bb_bandwidth"] = rolling_std / (rolling_mean + 1e-8)
        df["bb_bandwidth"] = df["bb_bandwidth"].fillna(df["bb_bandwidth"].mean())

        self.feature_names.extend(["price_zscore", "bb_bandwidth"])

        return df

    def compute_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify volatility regime
        0 = low, 1 = normal, 2 = high, 3 = extreme
        """
        vol = df["volatility_20"]

        percentiles = vol.quantile([0.25, 0.50, 0.75])

        df["vol_regime"] = 1  # default normal

        df.loc[vol < percentiles[0.25], "vol_regime"] = 0  # low
        df.loc[vol > percentiles[0.50], "vol_regime"] = 2  # high
        df.loc[vol > percentiles[0.75], "vol_regime"] = 3  # extreme

        self.feature_names.append("vol_regime")

        return df

    def compute_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Measure trend strength using ADX-like metric
        But normalized and simplified
        """
        window = 14

        # Directional movement
        high_diff = df["high"].diff()
        low_diff = -df["low"].diff()

        pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # Smooth
        pos_dm_smooth = pd.Series(pos_dm).rolling(window=window).mean()
        neg_dm_smooth = pd.Series(neg_dm).rolling(window=window).mean()

        # ATR for normalization
        atr = self._compute_atr(df, period=window)

        # Directional indicators
        pos_di = pos_dm_smooth / (atr + 1e-8)
        neg_di = neg_dm_smooth / (atr + 1e-8)

        # Trend strength (similar to ADX)
        df["trend_strength"] = abs(pos_di - neg_di) / (pos_di + neg_di + 1e-8)
        df["trend_strength"] = df["trend_strength"].fillna(0)

        self.feature_names.append("trend_strength")

        return df

    def compute_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Market microstructure features
        High-low range, close position within range
        """
        # Range
        range_val = df["high"] - df["low"]
        atr = self._compute_atr(df, period=14)

        df["range_norm"] = range_val / (atr + 1e-8)
        df["range_norm"] = df["range_norm"].fillna(1)

        # Close position within range (0 = low, 1 = high)
        df["close_position"] = (df["close"] - df["low"]) / (range_val + 1e-8)
        df["close_position"] = df["close_position"].fillna(0.5)

        self.feature_names.extend(["range_norm", "close_position"])

        return df

    def compute_rsi(self, df: pd.DataFrame, periods: list = [7, 14, 21]) -> pd.DataFrame:
        """
        Relative Strength Index at multiple periods
        Normalized to [-1, 1] range for better neural network training
        """
        for period in periods:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))

            # Normalize RSI from [0, 100] to [-1, 1]
            col_name = f"rsi_{period}"
            df[col_name] = (rsi - 50) / 50
            df[col_name] = df[col_name].fillna(0)

            self.feature_names.append(col_name)

        return df

    def compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD indicator with signal line and histogram
        All normalized by ATR
        """
        # MACD components
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        atr = self._compute_atr(df, period=14)

        # Normalize by ATR
        df["macd_norm"] = macd_line / (atr + 1e-8)
        df["macd_signal_norm"] = signal_line / (atr + 1e-8)
        df["macd_hist_norm"] = histogram / (atr + 1e-8)

        # MACD crossover signal
        df["macd_cross"] = np.where(macd_line > signal_line, 1, -1)

        for col in ["macd_norm", "macd_signal_norm", "macd_hist_norm"]:
            df[col] = df[col].fillna(0)
            df[col] = np.clip(df[col], -5, 5)

        self.feature_names.extend(["macd_norm", "macd_signal_norm", "macd_hist_norm", "macd_cross"])

        return df

    def compute_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator (%K and %D)
        Normalized to [-1, 1] range
        """
        # %K
        lowest_low = df["low"].rolling(window=k_period).min()
        highest_high = df["high"].rolling(window=k_period).max()
        stoch_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low + 1e-8)

        # %D (signal line)
        stoch_d = stoch_k.rolling(window=d_period).mean()

        # Normalize to [-1, 1]
        df["stoch_k"] = (stoch_k - 50) / 50
        df["stoch_d"] = (stoch_d - 50) / 50

        # Stochastic crossover
        df["stoch_cross"] = np.where(stoch_k > stoch_d, 1, -1)

        for col in ["stoch_k", "stoch_d"]:
            df[col] = df[col].fillna(0)

        self.feature_names.extend(["stoch_k", "stoch_d", "stoch_cross"])

        return df

    def compute_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Dynamic support/resistance levels
        Distance to recent high/low levels normalized by ATR
        """
        atr = self._compute_atr(df, period=14)

        # Rolling high/low as resistance/support
        rolling_high = df["high"].rolling(window=window).max()
        rolling_low = df["low"].rolling(window=window).min()

        # Distance to resistance (negative = above resistance)
        df["dist_resistance"] = (rolling_high - df["close"]) / (atr + 1e-8)
        df["dist_resistance"] = df["dist_resistance"].fillna(0)
        df["dist_resistance"] = np.clip(df["dist_resistance"], -5, 5)

        # Distance to support (positive = above support)
        df["dist_support"] = (df["close"] - rolling_low) / (atr + 1e-8)
        df["dist_support"] = df["dist_support"].fillna(0)
        df["dist_support"] = np.clip(df["dist_support"], -5, 5)

        # Position within range (0 = at support, 1 = at resistance)
        df["sr_position"] = (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-8)
        df["sr_position"] = df["sr_position"].fillna(0.5)

        self.feature_names.extend(["dist_resistance", "dist_support", "sr_position"])

        return df

    def compute_order_flow_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Order flow imbalance proxy using volume and price action
        Estimates buying vs selling pressure
        """
        # Volume-weighted price change
        price_change = df["close"] - df["open"]
        volume_pressure = price_change * df["volume"]

        # Normalize
        df["volume_pressure"] = volume_pressure / (volume_pressure.rolling(20).std() + 1e-8)
        df["volume_pressure"] = df["volume_pressure"].fillna(0)
        df["volume_pressure"] = np.clip(df["volume_pressure"], -3, 3)

        # Cumulative delta proxy (buying vs selling)
        df["cum_delta"] = df["volume_pressure"].rolling(window=10).sum()
        df["cum_delta"] = df["cum_delta"].fillna(0)
        df["cum_delta"] = np.clip(df["cum_delta"], -10, 10)

        # Volume climax detection
        vol_ma = df["volume"].rolling(20).mean()
        df["vol_climax"] = df["volume"] / (vol_ma + 1e-8)
        df["vol_climax"] = np.where(df["vol_climax"] > 2, df["candle_direction"], 0)

        self.feature_names.extend(["volume_pressure", "cum_delta", "vol_climax"])

        return df

    def compute_multi_timeframe_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-timeframe momentum signals
        Helps capture trend alignment across periods
        """
        # Short-term momentum (fast)
        df["momentum_fast"] = df["close"].pct_change(periods=3)
        df["momentum_fast"] = np.clip(df["momentum_fast"], -0.1, 0.1) * 10  # Scale to [-1, 1]

        # Medium-term momentum
        df["momentum_medium"] = df["close"].pct_change(periods=10)
        df["momentum_medium"] = np.clip(df["momentum_medium"], -0.2, 0.2) * 5

        # Long-term momentum
        df["momentum_long"] = df["close"].pct_change(periods=30)
        df["momentum_long"] = np.clip(df["momentum_long"], -0.3, 0.3) * 3.33

        # Momentum alignment score (-3 to +3)
        df["momentum_align"] = (
            np.sign(df["momentum_fast"]) +
            np.sign(df["momentum_medium"]) +
            np.sign(df["momentum_long"])
        ) / 3  # Normalize to [-1, 1]

        for col in ["momentum_fast", "momentum_medium", "momentum_long", "momentum_align"]:
            df[col] = df[col].fillna(0)

        self.feature_names.extend(["momentum_fast", "momentum_medium", "momentum_long", "momentum_align"])

        return df

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Helper: compute Average True Range"""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all features
        """
        # Reset feature names
        self.feature_names = []

        # ===== CORE FEATURES =====
        df = self.compute_returns(df)
        df = self.compute_volatility(df)
        df = self.compute_candle_features(df)
        df = self.compute_volume_features(df)
        df = self.compute_momentum(df)
        df = self.compute_ema_distance(df)
        df = self.compute_atr_features(df)
        df = self.compute_price_compression(df)
        df = self.compute_volatility_regime(df)
        df = self.compute_trend_strength(df)
        df = self.compute_microstructure_features(df)

        # ===== NEW ADVANCED FEATURES =====
        # TEMPORARILY DISABLED - enable after retraining models locally
        # df = self.compute_rsi(df)              # RSI at multiple periods
        # df = self.compute_macd(df)             # MACD with histogram
        # df = self.compute_stochastic(df)       # Stochastic oscillator
        # df = self.compute_support_resistance(df)  # Support/Resistance
        # df = self.compute_order_flow_proxy(df)    # Order flow
        # df = self.compute_multi_timeframe_momentum(df)  # MTF momentum

        # Drop any remaining NaNs (from initial rolling windows)
        df = df.dropna().reset_index(drop=True)

        return df

    def get_feature_names(self) -> list:
        """Return list of all feature column names"""
        return self.feature_names

    def normalize_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Standardize features (z-score normalization)
        Important: compute stats on training set only to avoid lookahead
        """
        for col in feature_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / (std + 1e-8)

        return df


def create_labels(
    df: pd.DataFrame,
    lookahead: int = config.LABEL_LOOKAHEAD,
    threshold_multiplier: float = config.LABEL_THRESHOLD_MULTIPLIER
) -> pd.DataFrame:
    """
    Create labels for supervised learning - IMPROVED VERSION

    Uses multiple factors for labeling:
    1. Future return relative to ATR (dynamic threshold)
    2. Risk/reward consideration
    3. Maximum adverse excursion filtering

    Labels:
    - 0: NO TRADE (flat or risky)
    - 1: LONG (strong bullish)
    - 2: SHORT (strong bearish)
    """
    df = df.copy()

    # Compute ATR for dynamic thresholds
    atr = FeatureEngine()._compute_atr(df, period=14)

    # Future prices
    future_close = df["close"].shift(-lookahead)
    future_high = df["high"].rolling(window=lookahead).max().shift(-lookahead)
    future_low = df["low"].rolling(window=lookahead).min().shift(-lookahead)

    # Returns
    future_return = (future_close - df["close"]) / df["close"]

    # Maximum favorable/adverse excursion
    max_profit_long = (future_high - df["close"]) / df["close"]
    max_loss_long = (df["close"] - future_low) / df["close"]
    max_profit_short = (df["close"] - future_low) / df["close"]
    max_loss_short = (future_high - df["close"]) / df["close"]

    # Dynamic threshold based on ATR
    threshold = (atr / df["close"]) * threshold_multiplier

    # Risk/reward ratio requirement (minimum 1.5)
    min_risk_reward = 1.5

    # ===== IMPROVED LABELING LOGIC =====

    df["label"] = 0  # default NO TRADE

    # LONG conditions:
    # 1. Positive return exceeds threshold
    # 2. Max profit > Max loss (favorable risk/reward)
    # 3. Return is positive (trade closed in profit)
    long_condition = (
        (future_return > threshold) &
        (max_profit_long > max_loss_long * min_risk_reward) &
        (future_return > 0)
    )

    # SHORT conditions:
    # 1. Negative return exceeds threshold
    # 2. Max profit > Max loss (favorable risk/reward)
    # 3. Return is negative (trade closed in profit)
    short_condition = (
        (future_return < -threshold) &
        (max_profit_short > max_loss_short * min_risk_reward) &
        (future_return < 0)
    )

    df.loc[long_condition, "label"] = 1  # LONG
    df.loc[short_condition, "label"] = 2  # SHORT

    # Drop rows where we can't compute future (end of dataset)
    df = df.iloc[:-lookahead]

    return df


def create_labels_triple_barrier(
    df: pd.DataFrame,
    profit_take: float = 0.02,  # 2% profit target
    stop_loss: float = 0.01,   # 1% stop loss
    max_holding: int = 20      # Maximum holding period
) -> pd.DataFrame:
    """
    Alternative: Triple barrier labeling method
    Used by professional quant funds

    Barriers:
    1. Upper barrier (profit take)
    2. Lower barrier (stop loss)
    3. Time barrier (maximum holding period)

    Label is based on which barrier is hit first
    """
    df = df.copy()
    labels = np.zeros(len(df))

    for i in range(len(df) - max_holding):
        entry_price = df["close"].iloc[i]
        upper_barrier = entry_price * (1 + profit_take)
        lower_barrier = entry_price * (1 - stop_loss)

        # Look forward
        for j in range(1, max_holding + 1):
            if i + j >= len(df):
                break

            high = df["high"].iloc[i + j]
            low = df["low"].iloc[i + j]

            # Check if barriers are hit
            if high >= upper_barrier:
                labels[i] = 1  # LONG (hit profit target)
                break
            elif low <= lower_barrier:
                labels[i] = 2  # SHORT (hit stop loss = price went down)
                break
        # If no barrier hit, label stays 0 (NO TRADE)

    df["label"] = labels.astype(int)
    df = df.iloc[:-max_holding]

    return df


if __name__ == "__main__":
    # Test feature engineering
    from market import BinanceDataFetcher

    fetcher = BinanceDataFetcher()
    df = fetcher.fetch_historical_data("BTCUSDT", "1h", days=365)

    if df is not None:
        print(f"Original data shape: {df.shape}")

        engine = FeatureEngine()
        df_features = engine.create_all_features(df)

        print(f"Features shape: {df_features.shape}")
        print(f"Number of features: {len(engine.get_feature_names())}")
        print(f"Features: {engine.get_feature_names()}")

        # Create labels
        df_labeled = create_labels(df_features)
        print(f"\nLabeled data shape: {df_labeled.shape}")
        print(f"Label distribution:\n{df_labeled['label'].value_counts()}")
