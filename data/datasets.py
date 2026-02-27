"""
PyTorch Dataset for Time Series Sequences
Prepares data for neural network training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CryptoSequenceDataset(Dataset):
    """
    Dataset that creates sequences of features for time series prediction
    Each sample is a sequence of SEQUENCE_LENGTH candles
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        sequence_length: int = config.SEQUENCE_LENGTH,
        label_col: str = "label"
    ):
        """
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            sequence_length: Length of input sequences
            label_col: Name of label column
        """
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        self.label_col = label_col

        # Extract feature matrix
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = df[label_col].values.astype(np.int64)

        # Valid indices (where we can create full sequences)
        self.valid_indices = list(range(sequence_length, len(df)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: tensor of shape [sequence_length, num_features]
            label: tensor of shape [1]
        """
        actual_idx = self.valid_indices[idx]

        # Get sequence of features
        start_idx = actual_idx - self.sequence_length
        end_idx = actual_idx

        feature_sequence = self.features[start_idx:end_idx]
        label = self.labels[actual_idx]

        # Convert to tensors
        feature_tensor = torch.from_numpy(feature_sequence)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feature_tensor, label_tensor


def create_dataloaders(
    df: pd.DataFrame,
    feature_cols: list,
    train_split: float = config.TRAIN_TEST_SPLIT,
    batch_size: int = config.BATCH_SIZE,
    sequence_length: int = config.SEQUENCE_LENGTH
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with walk-forward split

    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature columns
        train_split: Proportion of data for training
        batch_size: Batch size
        sequence_length: Sequence length

    Returns:
        train_loader, val_loader
    """
    # Walk-forward split (time-based, no shuffle)
    split_idx = int(len(df) * train_split)

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")

    # Create datasets
    train_dataset = CryptoSequenceDataset(
        train_df,
        feature_cols=feature_cols,
        sequence_length=sequence_length
    )

    val_dataset = CryptoSequenceDataset(
        val_df,
        feature_cols=feature_cols,
        sequence_length=sequence_length
    )

    # Create dataloaders
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # shuffle within training set
        num_workers=0,  # set to 0 for Windows compatibility
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader


def create_walk_forward_splits(
    df: pd.DataFrame,
    feature_cols: list,
    n_splits: int = config.WALK_FORWARD_SPLITS,
    min_train_pct: float = config.MIN_TRAIN_PCT,
    batch_size: int = config.BATCH_SIZE,
    sequence_length: int = config.SEQUENCE_LENGTH
) -> list:
    """
    Create expanding window walk-forward splits for proper time-series validation.

    Returns list of (train_loader, val_loader, fold_info) tuples.
    Each fold uses more training data: [50/10], [60/10], [70/10], [80/10], [90/10]
    """
    total_len = len(df)
    # Each validation fold is ~10% of data
    val_size = total_len // (n_splits + 1)
    splits = []

    for i in range(n_splits):
        # Training end = min_train_pct + i * step
        train_end_pct = min_train_pct + i * ((1.0 - min_train_pct - 0.1) / max(n_splits - 1, 1))
        train_end = int(total_len * train_end_pct)
        val_end = min(train_end + val_size, total_len)

        if val_end <= train_end or train_end < sequence_length * 2:
            continue

        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)

        if len(val_df) < sequence_length + 10:
            continue

        train_dataset = CryptoSequenceDataset(train_df, feature_cols, sequence_length)
        val_dataset = CryptoSequenceDataset(val_df, feature_cols, sequence_length)

        use_pin = torch.cuda.is_available()
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=use_pin
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=use_pin
        )

        fold_info = {
            "fold": i + 1,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "train_pct": f"{train_end_pct:.0%}",
        }

        splits.append((train_loader, val_loader, fold_info))

    return splits


def get_class_weights(df: pd.DataFrame, label_col: str = "label") -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset
    Used in loss function to handle class imbalance

    Args:
        df: DataFrame with labels
        label_col: Name of label column

    Returns:
        Tensor of class weights [num_classes]
    """
    label_counts = df[label_col].value_counts().sort_index()

    # Inverse frequency weighting
    total = len(df)
    weights = []

    for class_idx in range(config.NUM_CLASSES):
        if class_idx in label_counts.index:
            count = label_counts[class_idx]
            weight = total / (config.NUM_CLASSES * count)
        else:
            weight = 1.0
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Test dataset creation
    from market import BinanceDataFetcher
    from features import FeatureEngine, create_labels

    fetcher = BinanceDataFetcher()
    df = fetcher.fetch_historical_data("BTCUSDT", "1h", days=365)

    if df is not None:
        # Create features
        engine = FeatureEngine()
        df = engine.create_all_features(df)
        df = create_labels(df)

        feature_cols = engine.get_feature_names()

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            df,
            feature_cols=feature_cols,
            batch_size=32
        )

        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Test one batch
        features, labels = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"Features: {features.shape}")  # [batch, sequence, features]
        print(f"Labels: {labels.shape}")  # [batch]

        # Class weights
        weights = get_class_weights(df)
        print(f"\nClass weights: {weights}")
