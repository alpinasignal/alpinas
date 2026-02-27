"""
Probability Calibration using Platt Scaling
Maps raw model probabilities to calibrated (reliable) probabilities.
A model saying 70% should mean it's right 70% of the time.
"""

import numpy as np
from typing import Optional
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loguru import logger

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class PlattCalibrator:
    """
    Platt scaling calibration for probability outputs.
    Trains a logistic regression on validation set predictions
    to map raw probabilities to calibrated probabilities.
    """

    def __init__(self):
        self.calibrators = None  # One per class (one-vs-rest)
        self.is_trained = False

    def train(self, raw_probs: np.ndarray, true_labels: np.ndarray):
        """
        Train calibrator on validation set predictions.

        Args:
            raw_probs: Raw model probabilities, shape [n_samples, n_classes]
            true_labels: True class labels, shape [n_samples]
        """
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not installed, skipping calibration")
            return

        if len(raw_probs) < 50:
            logger.warning("Not enough samples for calibration, need at least 50")
            return

        n_classes = raw_probs.shape[1]
        self.calibrators = []

        for class_idx in range(n_classes):
            # Binary: this class vs rest
            binary_labels = (true_labels == class_idx).astype(int)
            class_probs = raw_probs[:, class_idx].reshape(-1, 1)

            # Platt scaling = logistic regression on raw probs
            lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)

            # Need both classes present
            if binary_labels.sum() < 5 or (1 - binary_labels).sum() < 5:
                self.calibrators.append(None)
                continue

            lr.fit(class_probs, binary_labels)
            self.calibrators.append(lr)

        self.is_trained = True
        logger.info(f"Platt calibration trained on {len(raw_probs)} samples")

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Calibrate raw probabilities.

        Args:
            raw_probs: Raw probabilities, shape [n_classes] or [n_samples, n_classes]

        Returns:
            Calibrated probabilities, same shape
        """
        if not self.is_trained or self.calibrators is None:
            return raw_probs

        single = raw_probs.ndim == 1
        if single:
            raw_probs = raw_probs.reshape(1, -1)

        calibrated = np.zeros_like(raw_probs)

        for class_idx, lr in enumerate(self.calibrators):
            if lr is None:
                calibrated[:, class_idx] = raw_probs[:, class_idx]
            else:
                class_probs = raw_probs[:, class_idx].reshape(-1, 1)
                # Predict probability of being this class
                cal_probs = lr.predict_proba(class_probs)[:, 1]
                calibrated[:, class_idx] = cal_probs

        # Normalize rows to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / (row_sums + 1e-8)

        return calibrated[0] if single else calibrated

    def save(self, path: str):
        """Save calibrator."""
        if self.calibrators is not None:
            with open(path, "wb") as f:
                pickle.dump({"calibrators": self.calibrators}, f)

    def load(self, path: str):
        """Load calibrator."""
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.calibrators = data["calibrators"]
            self.is_trained = True
            logger.info(f"Loaded calibrator from {path}")
