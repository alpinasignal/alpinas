"""
Online Learning — Weekly Retraining Pipeline
Fetches latest data, retrains models incrementally, and deploys only if improved.
Designed to run as a scheduled job (cron/Railway scheduler).
"""

import torch
import numpy as np
import os
import sys
from datetime import datetime
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger


class OnlineLearner:
    """
    Manages incremental model retraining with safety checks.
    Only deploys new model if it outperforms the old one on recent data.
    """

    def __init__(self, min_improvement: float = 0.5):
        """
        Args:
            min_improvement: Minimum accuracy improvement (%) to deploy new model
        """
        self.min_improvement = min_improvement

    def retrain_symbol(
        self,
        symbol: str,
        timeframe: str,
        warm_start: bool = True
    ) -> dict:
        """
        Retrain model for a symbol/timeframe.

        Args:
            symbol: e.g. "BTCUSDT"
            timeframe: e.g. "1h"
            warm_start: If True, initialize from existing weights

        Returns:
            Dict with training results
        """
        from ai.train import train_model, load_cached_data, Trainer
        from data.features import FeatureEngine, create_labels
        from data.datasets import create_dataloaders, get_class_weights
        from ai.model import create_model

        model_filename = f"{symbol}_{timeframe}_{config.MODEL_TYPE}.pt"
        model_path = os.path.join(config.MODELS_DIR, model_filename)
        backup_path = os.path.join(config.MODELS_DIR, f"{model_filename}.backup")

        logger.info(f"Online retraining: {symbol} {timeframe}")

        # Step 1: Load latest data
        logger.info("Fetching latest data...")
        from data.market import BinanceDataFetcher
        fetcher = BinanceDataFetcher()
        df = fetcher.fetch_or_load(symbol, timeframe, use_cache=False)

        if df is None or len(df) < 500:
            logger.error(f"Insufficient data for retraining: {len(df) if df is not None else 0}")
            return {"success": False, "error": "Insufficient data"}

        # BTC correlation data
        btc_df = None
        if symbol.upper() != "BTCUSDT":
            btc_df = fetcher.fetch_or_load("BTCUSDT", timeframe)

        # Step 2: Feature engineering
        engine = FeatureEngine()
        df = engine.create_all_features(df, btc_df=btc_df, symbol=symbol)
        df = create_labels(df)
        feature_cols = engine.get_feature_names()

        # Step 3: Create dataloaders
        train_loader, val_loader = create_dataloaders(df, feature_cols)
        class_weights = get_class_weights(df)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Step 4: Create model (with warm start if available)
        num_features = len(feature_cols)

        if warm_start and os.path.exists(model_path):
            logger.info("Warm start: loading existing model weights")
            try:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                old_num_features = checkpoint.get("num_features", num_features)

                if old_num_features == num_features:
                    model = create_model(num_features, device=device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    old_accuracy = self._evaluate(model, val_loader, device)
                    logger.info(f"Old model accuracy: {old_accuracy:.2f}%")
                else:
                    logger.warning(f"Feature count changed ({old_num_features} -> {num_features}), training from scratch")
                    model = create_model(num_features, device=device)
                    old_accuracy = 0
            except Exception as e:
                logger.warning(f"Could not load old model: {e}, training from scratch")
                model = create_model(num_features, device=device)
                old_accuracy = 0
        else:
            model = create_model(num_features, device=device)
            old_accuracy = 0

        # Step 5: Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            device=device,
            learning_rate=config.LEARNING_RATE * 0.5  # Lower LR for fine-tuning
        )

        history = trainer.train(max_epochs=config.MAX_EPOCHS)

        # Step 6: Evaluate new model
        new_accuracy = self._evaluate(model, val_loader, device)
        logger.info(f"New model accuracy: {new_accuracy:.2f}% (old: {old_accuracy:.2f}%)")

        improvement = new_accuracy - old_accuracy

        # Step 7: Deploy only if improved
        if improvement >= self.min_improvement or old_accuracy == 0:
            # Backup old model
            if os.path.exists(model_path):
                try:
                    import shutil
                    shutil.copy2(model_path, backup_path)
                    logger.info(f"Backed up old model to {backup_path}")
                except Exception as e:
                    logger.warning(f"Could not backup: {e}")

            # Save new model
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": config.MODEL_TYPE,
                "num_features": num_features,
                "feature_names": feature_cols,
                "symbol": symbol,
                "timeframe": timeframe,
                "history": history,
                "retrained_at": datetime.now().isoformat(),
                "config": {
                    "hidden_dim": config.HIDDEN_DIM,
                    "num_layers": config.NUM_LAYERS,
                    "num_heads": config.NUM_HEADS,
                    "dropout": config.DROPOUT,
                    "sequence_length": config.SEQUENCE_LENGTH,
                }
            }, model_path)

            logger.info(f"Deployed new model: +{improvement:.2f}% improvement")
            return {
                "success": True,
                "deployed": True,
                "old_accuracy": round(old_accuracy, 2),
                "new_accuracy": round(new_accuracy, 2),
                "improvement": round(improvement, 2),
            }
        else:
            logger.info(f"New model NOT deployed: {improvement:.2f}% < {self.min_improvement}% threshold")
            return {
                "success": True,
                "deployed": False,
                "old_accuracy": round(old_accuracy, 2),
                "new_accuracy": round(new_accuracy, 2),
                "improvement": round(improvement, 2),
                "reason": f"Improvement {improvement:.2f}% below threshold {self.min_improvement}%"
            }

    def _evaluate(self, model, val_loader, device) -> float:
        """Evaluate model accuracy."""
        model.eval()
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

    def retrain_all(self, warm_start: bool = True) -> dict:
        """Retrain all models for all supported pairs and timeframes."""
        results = {}

        for symbol in config.SUPPORTED_PAIRS:
            for timeframe in config.TIMEFRAMES:
                key = f"{symbol}_{timeframe}"
                try:
                    result = self.retrain_symbol(symbol, timeframe, warm_start=warm_start)
                    results[key] = result
                    logger.info(f"{key}: {result}")
                except Exception as e:
                    logger.error(f"{key}: Failed - {e}")
                    results[key] = {"success": False, "error": str(e)}

        # Summary
        deployed = sum(1 for r in results.values() if r.get("deployed", False))
        failed = sum(1 for r in results.values() if not r.get("success", False))
        total = len(results)

        logger.info(f"Retraining complete: {deployed}/{total} deployed, {failed} failed")
        return results


if __name__ == "__main__":
    logger.add(
        os.path.join(config.LOGS_DIR, "online_learning.log"),
        rotation="50 MB",
        level="INFO"
    )

    learner = OnlineLearner(min_improvement=0.5)
    results = learner.retrain_all(warm_start=True)

    print("\n=== RETRAINING SUMMARY ===")
    for key, result in results.items():
        status = "DEPLOYED" if result.get("deployed") else "SKIPPED" if result.get("success") else "FAILED"
        print(f"  {key}: {status}")
