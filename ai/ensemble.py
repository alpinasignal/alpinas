"""
Ensemble Models for Crypto Prediction
XGBoost + LightGBM + Statistical Momentum alongside Transformer
Each model outputs probabilities for [NO_TRADE, LONG, SHORT]
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("lightgbm not installed. Install with: pip install lightgbm")


class XGBoostModel:
    """
    XGBoost gradient boosted trees for tabular features.
    Uses flattened last-N-candle features as input.
    """

    def __init__(self, flatten_window: int = 5):
        self.flatten_window = flatten_window
        self.model = None
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """Flatten last N candles into a single feature vector."""
        features = df[feature_cols].values.astype(np.float32)
        n_features = len(feature_cols)

        if len(features) < self.flatten_window:
            return np.zeros((1, self.flatten_window * n_features), dtype=np.float32)

        # Take last flatten_window rows and flatten
        flat = features[-self.flatten_window:].flatten()
        return flat.reshape(1, -1)

    def prepare_training_data(self, df: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with flattened sliding windows."""
        features = df[feature_cols].values.astype(np.float32)
        labels = df["label"].values.astype(np.int64)
        n_features = len(feature_cols)

        X_list = []
        y_list = []

        for i in range(self.flatten_window, len(df)):
            flat = features[i - self.flatten_window:i].flatten()
            X_list.append(flat)
            y_list.append(labels[i])

        return np.array(X_list), np.array(y_list)

    def train(self, df: pd.DataFrame, feature_cols: list, val_df: pd.DataFrame = None):
        """Train XGBoost model."""
        if not HAS_XGB:
            logger.error("xgboost not installed")
            return

        X_train, y_train = self.prepare_training_data(df, feature_cols)

        # Class weights
        classes, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sample_weights = np.array([weight_map.get(y, 1.0) for y in y_train])

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)

        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "tree_method": "hist",
        }

        evals = [(dtrain, "train")]
        if val_df is not None:
            X_val, y_val = self.prepare_training_data(val_df, feature_cols)
            if len(X_val) > 0:
                dval = xgb.DMatrix(X_val, label=y_val)
                evals.append((dval, "val"))

        self.model = xgb.train(
            params, dtrain, num_boost_round=300,
            evals=evals, early_stopping_rounds=20, verbose_eval=False
        )
        self.is_trained = True
        logger.info(f"XGBoost trained: {self.model.best_iteration} rounds")

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """Predict probabilities [NO_TRADE, LONG, SHORT]."""
        if not self.is_trained or self.model is None:
            return np.array([0.34, 0.33, 0.33])

        flat = self.prepare_features(df, feature_cols)
        dtest = xgb.DMatrix(flat)
        probs = self.model.predict(dtest)
        return probs[0] if len(probs.shape) > 1 else probs

    def save(self, path: str):
        if self.model is not None:
            self.model.save_model(path)

    def load(self, path: str):
        if not HAS_XGB:
            return
        if os.path.exists(path):
            self.model = xgb.Booster()
            self.model.load_model(path)
            self.is_trained = True


class LightGBMModel:
    """
    LightGBM gradient boosted trees — fast and efficient.
    """

    def __init__(self, flatten_window: int = 5):
        self.flatten_window = flatten_window
        self.model = None
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """Flatten last N candles into single vector."""
        features = df[feature_cols].values.astype(np.float32)
        if len(features) < self.flatten_window:
            return np.zeros((1, self.flatten_window * len(feature_cols)), dtype=np.float32)
        flat = features[-self.flatten_window:].flatten()
        return flat.reshape(1, -1)

    def prepare_training_data(self, df: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with flattened sliding windows."""
        features = df[feature_cols].values.astype(np.float32)
        labels = df["label"].values.astype(np.int64)

        X_list = []
        y_list = []
        for i in range(self.flatten_window, len(df)):
            flat = features[i - self.flatten_window:i].flatten()
            X_list.append(flat)
            y_list.append(labels[i])

        return np.array(X_list), np.array(y_list)

    def train(self, df: pd.DataFrame, feature_cols: list, val_df: pd.DataFrame = None):
        """Train LightGBM model."""
        if not HAS_LGB:
            logger.error("lightgbm not installed")
            return

        X_train, y_train = self.prepare_training_data(df, feature_cols)

        # Class weights
        classes, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sample_weights = np.array([weight_map.get(y, 1.0) for y in y_train])

        dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights)

        dval_list = []
        if val_df is not None:
            X_val, y_val = self.prepare_training_data(val_df, feature_cols)
            if len(X_val) > 0:
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                dval_list = [dval]

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": -1,
        }

        callbacks = [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)]

        self.model = lgb.train(
            params, dtrain, num_boost_round=300,
            valid_sets=dval_list if dval_list else [dtrain],
            callbacks=callbacks,
        )
        self.is_trained = True
        logger.info(f"LightGBM trained: {self.model.best_iteration} rounds")

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """Predict probabilities [NO_TRADE, LONG, SHORT]."""
        if not self.is_trained or self.model is None:
            return np.array([0.34, 0.33, 0.33])

        flat = self.prepare_features(df, feature_cols)
        probs = self.model.predict(flat)
        return probs[0] if len(probs.shape) > 1 else probs

    def save(self, path: str):
        if self.model is not None:
            self.model.save_model(path)

    def load(self, path: str):
        if not HAS_LGB:
            return
        if os.path.exists(path):
            self.model = lgb.Booster(model_file=path)
            self.is_trained = True


class StatisticalMomentumModel:
    """
    Pure rules-based statistical momentum model — no ML.
    Uses weighted indicator scores to estimate signal direction.
    Serves as a stable baseline that doesn't overfit.
    """

    def __init__(self):
        self.is_trained = True  # No training needed

    def predict(self, df: pd.DataFrame, feature_cols: list = None) -> np.ndarray:
        """
        Generate probability distribution from indicator rules.
        Uses the last row of the DataFrame.
        """
        if len(df) < 5:
            return np.array([1.0, 0.0, 0.0])

        last = df.iloc[-1]
        score = 0.0  # Positive = bullish, negative = bearish
        total_weight = 0.0

        # EMA trend (weight: 2.0)
        ema50 = last.get("ema_distance_50", 0)
        ema200 = last.get("ema_distance_200", 0)
        ema_score = np.clip(ema50 * 0.5 + ema200 * 0.3, -1, 1)
        score += ema_score * 2.0
        total_weight += 2.0

        # RSI momentum (weight: 1.5)
        rsi = last.get("rsi_14", 0)
        score += np.clip(rsi, -1, 1) * 1.5
        total_weight += 1.5

        # MACD (weight: 1.5)
        macd_hist = last.get("macd_hist_norm", 0)
        macd_cross = last.get("macd_cross", 0)
        macd_score = np.clip(macd_hist * 0.5 + macd_cross * 0.3, -1, 1)
        score += macd_score * 1.5
        total_weight += 1.5

        # Momentum alignment (weight: 1.5)
        mom_align = last.get("momentum_align", 0)
        score += np.clip(mom_align, -1, 1) * 1.5
        total_weight += 1.5

        # Volume pressure (weight: 1.0)
        vol_pressure = last.get("volume_pressure", 0)
        cum_delta = last.get("cum_delta", 0)
        vol_score = np.clip(vol_pressure * 0.3 + cum_delta * 0.1, -1, 1)
        score += vol_score * 1.0
        total_weight += 1.0

        # Regression slope (weight: 1.5)
        reg_slope = last.get("regression_slope_20", 0)
        score += np.clip(reg_slope * 0.3, -1, 1) * 1.5
        total_weight += 1.5

        # HH/LL structure (weight: 1.0)
        hh_ll = last.get("hh_ll_score", 0)
        score += np.clip(hh_ll, -1, 1) * 1.0
        total_weight += 1.0

        # Normalize score to [-1, 1]
        normalized = score / (total_weight + 1e-8)
        normalized = np.clip(normalized, -1, 1)

        # Convert to probabilities using softmax-like transform
        if normalized > 0.15:
            # Bullish
            long_prob = 0.3 + normalized * 0.5  # 0.3 to 0.8
            short_prob = max(0.05, 0.2 - normalized * 0.15)
            no_trade_prob = 1.0 - long_prob - short_prob
        elif normalized < -0.15:
            # Bearish
            short_prob = 0.3 + abs(normalized) * 0.5
            long_prob = max(0.05, 0.2 - abs(normalized) * 0.15)
            no_trade_prob = 1.0 - long_prob - short_prob
        else:
            # Neutral
            no_trade_prob = 0.5 + (0.15 - abs(normalized)) * 2
            long_prob = (1.0 - no_trade_prob) / 2
            short_prob = long_prob

        probs = np.array([no_trade_prob, long_prob, short_prob])
        probs = np.clip(probs, 0, 1)
        probs = probs / probs.sum()  # Ensure sums to 1

        return probs

    def save(self, path: str):
        # No parameters to save
        pass

    def load(self, path: str):
        # No parameters to load
        pass


class EnsembleManager:
    """
    Manages all ensemble models: XGBoost + LightGBM + Statistical.
    Provides unified predict interface.
    """

    def __init__(self):
        self.xgb_model = XGBoostModel(flatten_window=5)
        self.lgb_model = LightGBMModel(flatten_window=5)
        self.stat_model = StatisticalMomentumModel()

    def train_all(self, train_df: pd.DataFrame, feature_cols: list, val_df: pd.DataFrame = None):
        """Train all ensemble models."""
        if HAS_XGB:
            logger.info("Training XGBoost...")
            self.xgb_model.train(train_df, feature_cols, val_df)

        if HAS_LGB:
            logger.info("Training LightGBM...")
            self.lgb_model.train(train_df, feature_cols, val_df)

        logger.info("Statistical model ready (no training needed)")

    def predict_all(self, df: pd.DataFrame, feature_cols: list) -> Dict[str, np.ndarray]:
        """Get predictions from all models."""
        results = {}

        if self.xgb_model.is_trained:
            results["xgboost"] = self.xgb_model.predict(df, feature_cols)

        if self.lgb_model.is_trained:
            results["lightgbm"] = self.lgb_model.predict(df, feature_cols)

        results["statistical"] = self.stat_model.predict(df, feature_cols)

        return results

    def get_ensemble_vote(self, df: pd.DataFrame, feature_cols: list) -> Tuple[int, str]:
        """
        Get ensemble consensus vote for the hybrid signal system.
        Returns: (vote, detail_string)
        vote: +1 (LONG), -1 (SHORT), 0 (NEUTRAL)
        """
        predictions = self.predict_all(df, feature_cols)

        if not predictions:
            return 0, "Ens:none"

        # Average probabilities across all ensemble models
        avg_probs = np.mean(list(predictions.values()), axis=0)
        # [NO_TRADE, LONG, SHORT]

        max_idx = int(np.argmax(avg_probs))

        if max_idx == 1 and avg_probs[1] > 0.40:
            return 1, f"Ens:LONG({avg_probs[1]:.0%})"
        elif max_idx == 2 and avg_probs[2] > 0.40:
            return -1, f"Ens:SHORT({avg_probs[2]:.0%})"
        elif max_idx == 1:
            return 1, f"Ens:mildL({avg_probs[1]:.0%})"
        elif max_idx == 2:
            return -1, f"Ens:mildS({avg_probs[2]:.0%})"
        return 0, f"Ens:neutral({avg_probs[0]:.0%})"

    def save_all(self, base_dir: str, symbol: str, timeframe: str):
        """Save all ensemble models."""
        prefix = f"{symbol}_{timeframe}"
        self.xgb_model.save(os.path.join(base_dir, f"{prefix}_xgb.json"))
        self.lgb_model.save(os.path.join(base_dir, f"{prefix}_lgb.txt"))

    def load_all(self, base_dir: str, symbol: str, timeframe: str):
        """Load all ensemble models."""
        prefix = f"{symbol}_{timeframe}"
        xgb_path = os.path.join(base_dir, f"{prefix}_xgb.json")
        lgb_path = os.path.join(base_dir, f"{prefix}_lgb.txt")

        if os.path.exists(xgb_path):
            self.xgb_model.load(xgb_path)
            logger.info(f"Loaded XGBoost from {xgb_path}")
        if os.path.exists(lgb_path):
            self.lgb_model.load(lgb_path)
            logger.info(f"Loaded LightGBM from {lgb_path}")
