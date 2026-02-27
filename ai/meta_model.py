"""
Meta-Model (Second-Layer Learner)
Combines predictions from Transformer + XGBoost + LightGBM + Statistical + Regime
into a final calibrated probability distribution.

This is a "stacking" approach — the meta-model learns which base models
are reliable in which conditions.
"""

import numpy as np
from typing import Dict, Optional, List
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loguru import logger

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MetaModel:
    """
    Second-layer meta-learner that combines base model predictions.

    Input features for meta-model:
    - Transformer probabilities (3 values)
    - XGBoost probabilities (3 values)
    - LightGBM probabilities (3 values)
    - Statistical probabilities (3 values)
    - Regime info (regime index, confidence adjustment)
    - TA consensus scores (long_votes, short_votes, neutral_votes)
    Total: ~17 meta-features

    Output: Final [NO_TRADE, LONG, SHORT] probabilities
    """

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.use_xgb = HAS_XGB  # Prefer XGBoost, fallback to LogReg

    def _build_meta_features(
        self,
        transformer_probs: np.ndarray,
        ensemble_preds: Dict[str, np.ndarray],
        regime_info: Dict,
        ta_result: Dict
    ) -> np.ndarray:
        """Build meta-feature vector from all model predictions."""
        features = []

        # Transformer probabilities (3)
        features.extend(transformer_probs.tolist())

        # XGBoost probabilities (3)
        xgb_probs = ensemble_preds.get("xgboost", np.array([0.34, 0.33, 0.33]))
        features.extend(xgb_probs.tolist())

        # LightGBM probabilities (3)
        lgb_probs = ensemble_preds.get("lightgbm", np.array([0.34, 0.33, 0.33]))
        features.extend(lgb_probs.tolist())

        # Statistical probabilities (3)
        stat_probs = ensemble_preds.get("statistical", np.array([0.34, 0.33, 0.33]))
        features.extend(stat_probs.tolist())

        # Regime info (2)
        features.append(regime_info.get("cluster", 0))
        features.append(regime_info.get("confidence_adj", 1.0))

        # TA consensus (3)
        features.append(ta_result.get("long_votes", 0))
        features.append(ta_result.get("short_votes", 0))
        features.append(ta_result.get("neutral_votes", 0))

        return np.array(features, dtype=np.float32)

    def train(
        self,
        meta_features_list: List[np.ndarray],
        true_labels: np.ndarray
    ):
        """
        Train meta-model on stacked predictions.

        Args:
            meta_features_list: List of meta-feature vectors
            true_labels: True class labels
        """
        X = np.array(meta_features_list)
        y = true_labels

        if len(X) < 100:
            logger.warning(f"Not enough meta-training samples: {len(X)} (need 100+)")
            return

        if self.use_xgb and HAS_XGB:
            dtrain = xgb.DMatrix(X, label=y)
            params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 10,
                "eval_metric": "mlogloss",
                "verbosity": 0,
            }
            self.model = xgb.train(
                params, dtrain, num_boost_round=100,
                evals=[(dtrain, "train")],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            self.is_trained = True
            logger.info(f"Meta-model (XGBoost) trained on {len(X)} samples")

        elif HAS_SKLEARN:
            self.model = LogisticRegression(
                C=1.0, max_iter=500, multi_class="multinomial", solver="lbfgs"
            )
            self.model.fit(X, y)
            self.is_trained = True
            logger.info(f"Meta-model (LogReg) trained on {len(X)} samples")

        else:
            logger.error("Neither XGBoost nor scikit-learn available for meta-model")

    def predict(
        self,
        transformer_probs: np.ndarray,
        ensemble_preds: Dict[str, np.ndarray],
        regime_info: Dict,
        ta_result: Dict
    ) -> np.ndarray:
        """
        Get meta-model prediction.

        Returns: [NO_TRADE, LONG, SHORT] probabilities
        """
        if not self.is_trained or self.model is None:
            # Fallback: simple average of all model probabilities
            return self._fallback_predict(transformer_probs, ensemble_preds)

        meta_features = self._build_meta_features(
            transformer_probs, ensemble_preds, regime_info, ta_result
        )
        meta_features = meta_features.reshape(1, -1)

        if self.use_xgb and HAS_XGB and isinstance(self.model, xgb.Booster):
            dtest = xgb.DMatrix(meta_features)
            probs = self.model.predict(dtest)
            return probs[0] if len(probs.shape) > 1 else probs
        elif HAS_SKLEARN and hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(meta_features)
            return probs[0]

        return self._fallback_predict(transformer_probs, ensemble_preds)

    def _fallback_predict(
        self,
        transformer_probs: np.ndarray,
        ensemble_preds: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Simple weighted average when meta-model not trained."""
        all_probs = [transformer_probs]
        weights = [1.0]  # Transformer weight

        if "xgboost" in ensemble_preds:
            all_probs.append(ensemble_preds["xgboost"])
            weights.append(1.5)  # XGBoost usually performs well on tabular

        if "lightgbm" in ensemble_preds:
            all_probs.append(ensemble_preds["lightgbm"])
            weights.append(1.5)

        if "statistical" in ensemble_preds:
            all_probs.append(ensemble_preds["statistical"])
            weights.append(1.0)

        # Weighted average
        total_weight = sum(weights)
        avg = np.zeros(3)
        for probs, w in zip(all_probs, weights):
            avg += probs * w
        avg /= total_weight

        return avg

    def get_vote(
        self,
        transformer_probs: np.ndarray,
        ensemble_preds: Dict[str, np.ndarray],
        regime_info: Dict,
        ta_result: Dict
    ) -> tuple:
        """
        Get meta-model consensus vote for the hybrid system.
        Returns: (vote, detail_string)
        """
        probs = self.predict(transformer_probs, ensemble_preds, regime_info, ta_result)
        max_idx = int(np.argmax(probs))

        if max_idx == 1 and probs[1] > 0.40:
            return 1, f"Meta:LONG({probs[1]:.0%})"
        elif max_idx == 2 and probs[2] > 0.40:
            return -1, f"Meta:SHORT({probs[2]:.0%})"
        elif max_idx == 1:
            return 1, f"Meta:mildL({probs[1]:.0%})"
        elif max_idx == 2:
            return -1, f"Meta:mildS({probs[2]:.0%})"
        return 0, f"Meta:neutral({probs[0]:.0%})"

    def save(self, path: str):
        """Save meta-model."""
        if self.model is not None:
            data = {
                "model": self.model,
                "use_xgb": self.use_xgb,
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)

    def load(self, path: str):
        """Load meta-model."""
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.use_xgb = data.get("use_xgb", True)
            self.is_trained = True
            logger.info(f"Loaded meta-model from {path}")
