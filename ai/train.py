"""
Training Pipeline for Crypto Prediction Model
Walk-forward validation, early stopping, class imbalance handling
Professional ML training - no shortcuts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger


def load_cached_data(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Load data from cached CSV files (downloaded via download_data.py)
    Much faster than downloading live!

    Args:
        symbol: e.g., "BTCUSDT"
        timeframe: e.g., "1h"

    Returns:
        DataFrame with OHLCV data or None
    """
    # Convert symbol format: BTCUSDT -> BTCUSDT (no change needed for our format)
    # But download_data.py saves as BTC/USDT -> BTCUSDT
    filename = f"{symbol}_{timeframe}.csv"
    filepath = os.path.join(config.DATA_DIR, filename)

    if os.path.exists(filepath):
        logger.info(f"Loading cached data from {filepath}")
        df = pd.read_csv(filepath)

        # Ensure correct column names
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Rename columns if needed (ccxt format)
        column_mapping = {
            'timestamp': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }

        logger.info(f"Loaded {len(df)} candles from cache")
        return df
    else:
        logger.warning(f"Cached data not found: {filepath}")
        return None


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard-to-classify examples
    Paper: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Class weights tensor
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [batch, num_classes]
            targets: Class indices [batch]
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    Monitors validation loss
    """

    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Args:
            val_loss: Current validation loss
            model: Model to save if improved

        Returns:
            True if should stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            return False

        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True

        return False


class Trainer:
    """
    Trainer class for neural network models
    Handles training loop, validation, metrics
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        learning_rate: float = config.LEARNING_RATE,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate

        # Loss function - Focal Loss for better class imbalance handling
        if class_weights is not None:
            class_weights = class_weights.to(device)

        # Use Focal Loss for better handling of hard examples
        self.criterion = FocalLoss(
            alpha=class_weights,
            gamma=2.0  # Focus on hard examples
        )

        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

        # History
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch

        Returns:
            average_loss, accuracy
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for features, labels in progress_bar:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "acc": 100.0 * correct / total
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float, Dict]:
        """
        Validate model

        Returns:
            average_loss, accuracy, metrics_dict
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                # Metrics
                total_loss += loss.item()
                _, predicted = outputs.max(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        # Per-class metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        per_class_acc = {}
        for class_idx in range(config.NUM_CLASSES):
            mask = all_labels == class_idx
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_labels[mask]).mean() * 100
                per_class_acc[f"class_{class_idx}"] = class_acc

        metrics = {
            "per_class_accuracy": per_class_acc
        }

        return avg_loss, accuracy, metrics

    def train(self, max_epochs: int = config.MAX_EPOCHS) -> Dict:
        """
        Full training loop

        Args:
            max_epochs: Maximum number of epochs

        Returns:
            Training history
        """
        logger.info(f"Starting training for {max_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Learning rate: {self.learning_rate}")

        best_val_acc = 0.0

        for epoch in range(max_epochs):
            logger.info(f"\nEpoch {epoch+1}/{max_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_metrics = self.validate()

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Save metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # Logging
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            logger.info(f"Per-class accuracy: {val_metrics['per_class_accuracy']}")

            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")

            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                logger.info("Early stopping triggered")
                # Restore best model
                self.model.load_state_dict(self.early_stopping.best_model_state)
                break

        logger.info(f"\nTraining complete!")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")

        return self.history


def train_model(
    symbol: str,
    timeframe: str,
    model_type: str = config.MODEL_TYPE,
    save_path: Optional[str] = None
) -> nn.Module:
    """
    Complete training pipeline for a single symbol/timeframe

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Timeframe ("15m", "1h", "4h")
        model_type: "transformer" or "lstm"
        save_path: Path to save trained model

    Returns:
        Trained model
    """
    from data.features import FeatureEngine, create_labels
    from data.datasets import create_dataloaders, get_class_weights
    from ai.model import create_model

    logger.info(f"Training model for {symbol} {timeframe}")

    # Try to load from cache first (MUCH FASTER!)
    logger.info("Loading market data from cache...")
    df = load_cached_data(symbol, timeframe)

    # Fallback to live download if cache not available
    if df is None:
        logger.info("Cache not found, trying live download...")
        from data.market import BinanceDataFetcher
        fetcher = BinanceDataFetcher()
        df = fetcher.fetch_or_load(symbol, timeframe)

    if df is None:
        logger.error(f"Failed to load data for {symbol} {timeframe}")
        return None

    # Load BTC data for correlation features (skip for BTCUSDT itself)
    btc_df = None
    if symbol.upper() != "BTCUSDT":
        logger.info("Loading BTC data for correlation features...")
        btc_df = load_cached_data("BTCUSDT", timeframe)
        if btc_df is None:
            try:
                from data.market import BinanceDataFetcher
                fetcher = BinanceDataFetcher()
                btc_df = fetcher.fetch_or_load("BTCUSDT", timeframe)
            except Exception as e:
                logger.warning(f"Could not load BTC data for correlation: {e}")

    # Cap data size for training speed (use most recent candles)
    max_candles = getattr(config, 'MAX_TRAINING_CANDLES', 25000)
    if len(df) > max_candles:
        logger.info(f"Capping data from {len(df)} to {max_candles} most recent candles")
        df = df.tail(max_candles).reset_index(drop=True)
        if btc_df is not None and len(btc_df) > max_candles:
            btc_df = btc_df.tail(max_candles).reset_index(drop=True)

    # Feature engineering
    logger.info("Creating features...")
    engine = FeatureEngine()
    df = engine.create_all_features(df, btc_df=btc_df, symbol=symbol)
    df = create_labels(df)

    feature_cols = engine.get_feature_names()
    num_features = len(feature_cols)

    logger.info(f"Number of features: {num_features}")
    logger.info(f"Dataset size: {len(df)} samples")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(df, feature_cols)

    # Class weights for imbalanced dataset
    class_weights = get_class_weights(df)
    logger.info(f"Class weights: {class_weights}")

    # Create model
    logger.info(f"Creating {model_type} model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(num_features, model_type=model_type, device=device)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device
    )

    # Train
    history = trainer.train()

    # Save model if path provided
    if save_path:
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "num_features": num_features,
            "feature_names": feature_cols,
            "symbol": symbol,
            "timeframe": timeframe,
            "history": history,
            "config": {
                "hidden_dim": config.HIDDEN_DIM,
                "num_layers": config.NUM_LAYERS,
                "num_heads": config.NUM_HEADS,
                "dropout": config.DROPOUT,
                "sequence_length": config.SEQUENCE_LENGTH
            }
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    return model


if __name__ == "__main__":
    # Setup logging
    logger.add(
        os.path.join(config.LOGS_DIR, "training.log"),
        rotation="100 MB",
        level="INFO"
    )

    # Train a model
    symbol = "BTCUSDT"
    timeframe = "1h"

    model_filename = f"{symbol}_{timeframe}_{config.MODEL_TYPE}.pt"
    model_path = os.path.join(config.MODELS_DIR, model_filename)

    trained_model = train_model(
        symbol=symbol,
        timeframe=timeframe,
        model_type=config.MODEL_TYPE,
        save_path=model_path
    )
