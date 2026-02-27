"""
Feature Importance Analysis & Selection
Uses permutation importance to identify and prune low-value features.
Prevents overfitting by removing noise features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger


class FeatureSelector:
    """
    Analyzes feature importance using permutation importance.
    Identifies features that contribute positively to model performance
    and flags those that add noise (negative importance).
    """

    def __init__(self):
        self.importance_scores = {}
        self.selected_features = []

    def compute_permutation_importance(
        self,
        model: torch.nn.Module,
        val_loader,
        feature_names: list,
        device: str = "cpu",
        n_repeats: int = 5
    ) -> Dict[str, float]:
        """
        Compute permutation importance for each feature.

        For each feature:
        1. Record baseline accuracy
        2. Shuffle that feature across all samples
        3. Measure accuracy drop
        4. Importance = accuracy_baseline - accuracy_shuffled

        Higher importance = more useful feature.
        Negative importance = feature adds noise (should be removed).
        """
        model.eval()

        # Compute baseline accuracy
        baseline_acc = self._compute_accuracy(model, val_loader, device)
        logger.info(f"Baseline validation accuracy: {baseline_acc:.2f}%")

        importance = {}

        for feat_idx, feat_name in enumerate(feature_names):
            drops = []

            for _ in range(n_repeats):
                shuffled_acc = self._compute_accuracy_with_shuffle(
                    model, val_loader, device, feat_idx
                )
                drop = baseline_acc - shuffled_acc
                drops.append(drop)

            avg_drop = np.mean(drops)
            importance[feat_name] = round(float(avg_drop), 4)

        # Sort by importance (descending)
        self.importance_scores = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        return self.importance_scores

    def _compute_accuracy(self, model, val_loader, device) -> float:
        """Compute accuracy on validation set."""
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                _, predicted = outputs.max(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return 100.0 * correct / total if total > 0 else 0

    def _compute_accuracy_with_shuffle(
        self, model, val_loader, device, feature_idx: int
    ) -> float:
        """Compute accuracy with one feature shuffled."""
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                # Shuffle one feature across the batch
                features_shuffled = features.clone()
                batch_size = features_shuffled.size(0)

                # Shuffle across batch dimension for the specific feature
                perm = torch.randperm(batch_size)
                features_shuffled[:, :, feature_idx] = features_shuffled[perm, :, feature_idx]

                features_shuffled = features_shuffled.to(device)
                labels = labels.to(device)

                outputs = model(features_shuffled)
                _, predicted = outputs.max(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return 100.0 * correct / total if total > 0 else 0

    def select_features(
        self,
        min_importance: float = 0.0,
        top_n: int = None
    ) -> List[str]:
        """
        Select features based on importance threshold or top N.

        Args:
            min_importance: Minimum importance score (features below this are pruned)
            top_n: If set, select only top N features

        Returns:
            List of selected feature names
        """
        if not self.importance_scores:
            logger.warning("No importance scores computed yet")
            return []

        if top_n:
            selected = list(self.importance_scores.keys())[:top_n]
        else:
            selected = [
                name for name, score in self.importance_scores.items()
                if score >= min_importance
            ]

        self.selected_features = selected

        # Log results
        total = len(self.importance_scores)
        removed = total - len(selected)
        logger.info(f"Feature selection: {len(selected)}/{total} kept, {removed} pruned")

        # Log worst features (negative importance = noise)
        noise_features = [
            (name, score) for name, score in self.importance_scores.items()
            if score < 0
        ]
        if noise_features:
            logger.info(f"Noise features (negative importance): {noise_features}")

        return selected

    def print_importance_report(self):
        """Print formatted feature importance report."""
        if not self.importance_scores:
            print("No importance scores computed yet")
            return

        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE REPORT")
        print("=" * 60)

        for i, (name, score) in enumerate(self.importance_scores.items()):
            bar_len = max(0, int(score * 10))
            bar = "+" * bar_len if score >= 0 else "-" * abs(int(score * 10))
            status = "KEEP" if score >= 0 else "PRUNE"
            print(f"{i+1:3d}. {name:30s} | {score:+.4f} | {bar:20s} | {status}")

        print("=" * 60)
        positive = sum(1 for s in self.importance_scores.values() if s > 0)
        negative = sum(1 for s in self.importance_scores.values() if s < 0)
        neutral = sum(1 for s in self.importance_scores.values() if s == 0)
        print(f"Positive: {positive} | Neutral: {neutral} | Negative (noise): {negative}")
        print("=" * 60)
