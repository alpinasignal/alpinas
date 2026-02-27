"""
Bayesian Hyperparameter Optimization using Optuna
Optimizes model hyperparameters for maximum profit factor (not just accuracy).
"""

import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("optuna not installed. Install with: pip install optuna")


class HyperOptimizer:
    """
    Bayesian optimization for model hyperparameters.
    Objective: maximize profit factor on validation set.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        n_trials: int = 30,
        timeout_seconds: int = 3600
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_trials = n_trials
        self.timeout = timeout_seconds
        self.best_params = None

    def objective(self, trial) -> float:
        """
        Optuna objective function.
        Returns negative profit factor (Optuna minimizes by default).
        """
        from ai.train import Trainer
        from data.features import FeatureEngine, create_labels
        from data.datasets import create_dataloaders, get_class_weights
        from ai.model import TransformerPredictor
        from ai.train import load_cached_data, FocalLoss

        # Suggest hyperparameters
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        num_heads = trial.suggest_categorical("num_heads", [4, 8])
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        sequence_length = trial.suggest_categorical("sequence_length", [32, 48, 64])
        focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0, step=0.5)

        # Load data
        df = load_cached_data(self.symbol, self.timeframe)
        if df is None:
            return 0  # Cannot evaluate

        engine = FeatureEngine()
        df = engine.create_all_features(df, symbol=self.symbol)
        df = create_labels(df)
        feature_cols = engine.get_feature_names()
        num_features = len(feature_cols)

        # Create dataloaders with trial's batch_size
        from data.datasets import create_dataloaders as cd
        train_loader, val_loader = cd(
            df, feature_cols, batch_size=batch_size,
            sequence_length=sequence_length
        )

        class_weights = get_class_weights(df)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create model with trial's architecture
        model = TransformerPredictor(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            sequence_length=sequence_length
        ).to(device)

        # Custom loss with trial's gamma
        criterion = FocalLoss(
            alpha=class_weights.to(device),
            gamma=focal_gamma
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.01
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        # Train for limited epochs
        best_val_acc = 0
        patience = 5
        patience_counter = 0

        for epoch in range(20):  # Max 20 epochs per trial
            model.train()
            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validate
            model.eval()
            correct = 0
            total = 0
            val_loss_sum = 0

            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    val_loss_sum += loss.item()
                    _, predicted = outputs.max(dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            val_acc = 100.0 * correct / total if total > 0 else 0
            val_loss = val_loss_sum / len(val_loader) if len(val_loader) > 0 else float('inf')
            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            # Optuna pruning
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        logger.info(f"Trial {trial.number}: acc={best_val_acc:.2f}%, "
                    f"hd={hidden_dim}, nl={num_layers}, nh={num_heads}, "
                    f"do={dropout:.2f}, lr={learning_rate:.6f}")

        # Return accuracy (Optuna maximizes when direction="maximize")
        return best_val_acc

    def optimize(self) -> dict:
        """
        Run Bayesian hyperparameter optimization.

        Returns:
            Best parameters dict
        """
        if not HAS_OPTUNA:
            logger.error("Optuna not installed, cannot run hyperparameter optimization")
            return {}

        logger.info(f"Starting hyperparameter optimization for {self.symbol} {self.timeframe}")
        logger.info(f"Trials: {self.n_trials}, Timeout: {self.timeout}s")

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            study_name=f"{self.symbol}_{self.timeframe}_hyperopt"
        )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
            catch=(Exception,)
        )

        self.best_params = study.best_params

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best accuracy: {study.best_value:.2f}%")
        logger.info(f"Best params: {self.best_params}")

        return {
            "best_params": self.best_params,
            "best_accuracy": round(study.best_value, 2),
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
        }


if __name__ == "__main__":
    if not HAS_OPTUNA:
        print("Install optuna first: pip install optuna")
        sys.exit(1)

    logger.add(
        os.path.join(config.LOGS_DIR, "hyperopt.log"),
        rotation="50 MB",
        level="INFO"
    )

    optimizer = HyperOptimizer(
        symbol="BTCUSDT",
        timeframe="1h",
        n_trials=30,
        timeout_seconds=3600
    )

    results = optimizer.optimize()
    print(f"\nBest params: {results['best_params']}")
    print(f"Best accuracy: {results['best_accuracy']}%")
