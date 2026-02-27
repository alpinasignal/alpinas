"""
Market Regime Detection
Identifies market state (trending, ranging, high-vol) using K-Means clustering.
Used to adjust signal confidence based on market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed. Install with: pip install scikit-learn")


# Regime names
REGIMES = {
    0: "trending_up",
    1: "trending_down",
    2: "ranging",
    3: "high_volatility"
}


class RegimeDetector:
    """
    Detects market regime using K-Means clustering on market features.
    4 regimes: Trending-Up, Trending-Down, Ranging, High-Volatility.
    """

    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.regime_labels = {}  # Maps cluster index to regime name

    def _extract_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features used for regime detection."""
        features = []

        # Volatility
        if "volatility_20" in df.columns:
            features.append(df["volatility_20"].values)
        else:
            features.append(np.zeros(len(df)))

        # Trend strength
        if "trend_strength" in df.columns:
            features.append(df["trend_strength"].values)
        else:
            features.append(np.zeros(len(df)))

        # Bollinger bandwidth (range compression)
        if "bb_bandwidth" in df.columns:
            features.append(df["bb_bandwidth"].values)
        else:
            features.append(np.zeros(len(df)))

        # Volume ratio
        if "volume_ratio" in df.columns:
            features.append(df["volume_ratio"].values)
        else:
            features.append(np.ones(len(df)))

        # Momentum alignment
        if "momentum_align" in df.columns:
            features.append(df["momentum_align"].values)
        else:
            features.append(np.zeros(len(df)))

        # EMA distance (trend direction)
        if "ema_distance_50" in df.columns:
            features.append(df["ema_distance_50"].values)
        else:
            features.append(np.zeros(len(df)))

        X = np.column_stack(features)
        # Replace NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)
        return X

    def train(self, df: pd.DataFrame):
        """Train regime detector on historical data."""
        if not HAS_SKLEARN:
            logger.error("scikit-learn not installed, cannot train regime detector")
            return

        X = self._extract_regime_features(df)

        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # K-Means clustering
        self.model = KMeans(
            n_clusters=self.n_regimes,
            n_init=20,
            max_iter=300,
            random_state=42
        )
        labels = self.model.fit_predict(X_scaled)

        # Identify regimes by cluster characteristics
        self._label_clusters(df, labels)
        self.is_trained = True

        # Log regime distribution
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            regime_name = self.regime_labels.get(u, f"cluster_{u}")
            logger.info(f"Regime '{regime_name}': {c} samples ({c/len(labels)*100:.1f}%)")

    def _label_clusters(self, df: pd.DataFrame, labels: np.ndarray):
        """Assign regime names to cluster indices based on cluster characteristics."""
        for cluster_idx in range(self.n_regimes):
            mask = labels == cluster_idx
            if mask.sum() == 0:
                self.regime_labels[cluster_idx] = "unknown"
                continue

            cluster_df = df.iloc[mask]

            # Compute average characteristics
            avg_vol = cluster_df["volatility_20"].mean() if "volatility_20" in cluster_df.columns else 0
            avg_trend = cluster_df["trend_strength"].mean() if "trend_strength" in cluster_df.columns else 0
            avg_momentum = cluster_df["momentum_align"].mean() if "momentum_align" in cluster_df.columns else 0
            avg_ema = cluster_df["ema_distance_50"].mean() if "ema_distance_50" in cluster_df.columns else 0

            # Classify based on characteristics
            if avg_vol > df["volatility_20"].quantile(0.75) if "volatility_20" in df.columns else False:
                self.regime_labels[cluster_idx] = "high_volatility"
            elif avg_trend > 0.4 and avg_ema > 0.3:
                self.regime_labels[cluster_idx] = "trending_up"
            elif avg_trend > 0.4 and avg_ema < -0.3:
                self.regime_labels[cluster_idx] = "trending_down"
            else:
                self.regime_labels[cluster_idx] = "ranging"

    def detect(self, df: pd.DataFrame) -> Dict:
        """
        Detect current market regime from latest data.
        Returns: {"regime": str, "cluster": int, "confidence_adj": float}
        """
        if not self.is_trained or self.model is None or self.scaler is None:
            return self._fallback_detection(df)

        X = self._extract_regime_features(df)
        # Use last row only
        X_last = X[-1:] if len(X) > 0 else np.zeros((1, X.shape[1] if len(X.shape) > 1 else 6))
        X_scaled = self.scaler.transform(X_last)

        cluster = self.model.predict(X_scaled)[0]
        regime = self.regime_labels.get(cluster, "unknown")

        # Distance to cluster center = uncertainty
        center = self.model.cluster_centers_[cluster]
        dist = np.linalg.norm(X_scaled[0] - center)
        # Closer to center = more confidence in regime
        regime_confidence = max(0, 1 - dist / 3)

        # Confidence adjustment per regime
        conf_adj = self._get_confidence_adjustment(regime)

        return {
            "regime": regime,
            "cluster": int(cluster),
            "regime_confidence": round(float(regime_confidence), 3),
            "confidence_adj": conf_adj
        }

    def _fallback_detection(self, df: pd.DataFrame) -> Dict:
        """Simple rule-based regime detection when model not trained."""
        if len(df) < 20:
            return {"regime": "unknown", "cluster": -1, "regime_confidence": 0, "confidence_adj": 1.0}

        last = df.iloc[-1]
        vol = last.get("volatility_20", 0)
        trend = last.get("trend_strength", 0)
        ema = last.get("ema_distance_50", 0)

        if vol > 0.03:
            regime = "high_volatility"
        elif trend > 0.5 and ema > 0.3:
            regime = "trending_up"
        elif trend > 0.5 and ema < -0.3:
            regime = "trending_down"
        else:
            regime = "ranging"

        return {
            "regime": regime,
            "cluster": -1,
            "regime_confidence": 0.5,
            "confidence_adj": self._get_confidence_adjustment(regime)
        }

    def _get_confidence_adjustment(self, regime: str) -> float:
        """
        Regime-specific confidence multiplier.
        Trending = trust signals more, ranging = trust less.
        """
        adjustments = {
            "trending_up": 1.1,      # Boost confidence in strong trends
            "trending_down": 1.1,    # Trends are easier to predict
            "ranging": 0.85,         # Reduce confidence in choppy markets
            "high_volatility": 0.9,  # Slightly reduce in volatile markets
            "unknown": 1.0
        }
        return adjustments.get(regime, 1.0)

    def save(self, path: str):
        """Save regime detector."""
        if self.model is not None:
            data = {
                "model": self.model,
                "scaler": self.scaler,
                "regime_labels": self.regime_labels,
                "n_regimes": self.n_regimes
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)

    def load(self, path: str):
        """Load regime detector."""
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.regime_labels = data["regime_labels"]
            self.n_regimes = data["n_regimes"]
            self.is_trained = True
            logger.info(f"Loaded regime detector from {path}")
