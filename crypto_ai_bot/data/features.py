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

        # Apply all feature computations
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
    Create labels for supervised learning

    Labels are based on future return relative to ATR (dynamic threshold)
    - 0: NO TRADE (flat)
    - 1: LONG (up)
    - 2: SHORT (down)

    Args:
        df: DataFrame with features
        lookahead: How many candles forward to look
        threshold_multiplier: Multiplier for ATR-based threshold

    Returns:
        DataFrame with 'label' column
    """
    # Compute ATR for dynamic thresholds
    atr = FeatureEngine()._compute_atr(df, period=14)

    # Future return
    future_return = (df["close"].shift(-lookahead) - df["close"]) / df["close"]

    # Dynamic threshold based on ATR
    threshold = (atr / df["close"]) * threshold_multiplier

    # Assign labels
    df["label"] = 0  # default NO TRADE

    df.loc[future_return > threshold, "label"] = 1  # LONG
    df.loc[future_return < -threshold, "label"] = 2  # SHORT

    # Drop rows where we can't compute future return (end of dataset)
    df = df.iloc[:-lookahead]

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
