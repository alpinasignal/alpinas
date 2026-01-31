"""
Model Storage and Management
Version control for trained models
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger


class ModelStore:
    """
    Manages trained models and metadata
    Tracks versions, performance metrics
    """

    def __init__(self, models_dir: str = config.MODELS_DIR):
        self.models_dir = models_dir
        self.metadata_file = os.path.join(models_dir, "models_metadata.json")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"models": []}

    def _save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, indent=2, fp=f)

    def register_model(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
        model_path: str,
        metrics: Dict,
        version: Optional[str] = None
    ):
        """
        Register a trained model

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            model_type: Model architecture
            model_path: Path to model file
            metrics: Training metrics
            version: Version string (auto-generated if None)
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_info = {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_type": model_type,
            "model_path": model_path,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics
        }

        self.metadata["models"].append(model_info)
        self._save_metadata()

        logger.info(f"Registered model: {symbol}_{timeframe}_{model_type} v{version}")

    def get_model_path(
        self,
        symbol: str,
        timeframe: str,
        version: Optional[str] = None
    ) -> Optional[str]:
        """
        Get path to a specific model

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            version: Specific version (latest if None)

        Returns:
            Path to model file or None
        """
        matching_models = [
            m for m in self.metadata["models"]
            if m["symbol"] == symbol and m["timeframe"] == timeframe
        ]

        if not matching_models:
            return None

        if version:
            # Find specific version
            for model in matching_models:
                if model["version"] == version:
                    return model["model_path"]
            return None
        else:
            # Return latest
            latest_model = max(matching_models, key=lambda x: x["created_at"])
            return latest_model["model_path"]

    def list_models(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Dict]:
        """
        List all models, optionally filtered

        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe

        Returns:
            List of model info dicts
        """
        models = self.metadata["models"]

        if symbol:
            models = [m for m in models if m["symbol"] == symbol]

        if timeframe:
            models = [m for m in models if m["timeframe"] == timeframe]

        return models

    def get_latest_models(self) -> Dict[str, Dict]:
        """
        Get latest model for each symbol/timeframe combination

        Returns:
            Dictionary: {(symbol, timeframe): model_info}
        """
        latest_models = {}

        for model in self.metadata["models"]:
            key = (model["symbol"], model["timeframe"])

            if key not in latest_models:
                latest_models[key] = model
            else:
                # Compare timestamps
                if model["created_at"] > latest_models[key]["created_at"]:
                    latest_models[key] = model

        return latest_models


def build_model_path(symbol: str, timeframe: str, model_type: str, version: Optional[str] = None) -> str:
    """
    Build standard model filename

    Args:
        symbol: Trading pair
        timeframe: Timeframe
        model_type: Model type
        version: Version string

    Returns:
        Full path to model file
    """
    if version:
        filename = f"{symbol}_{timeframe}_{model_type}_v{version}.pt"
    else:
        filename = f"{symbol}_{timeframe}_{model_type}.pt"

    return os.path.join(config.MODELS_DIR, filename)


if __name__ == "__main__":
    # Test model store
    store = ModelStore()

    # Register a dummy model
    store.register_model(
        symbol="BTCUSDT",
        timeframe="1h",
        model_type="transformer",
        model_path="/path/to/model.pt",
        metrics={
            "val_accuracy": 65.5,
            "val_loss": 0.82
        }
    )

    # List models
    models = store.list_models()
    print(f"Total models: {len(models)}")

    for model in models:
        print(f"{model['symbol']} {model['timeframe']} - v{model['version']}")

    # Get latest models
    latest = store.get_latest_models()
    print(f"\nLatest models: {len(latest)}")
    for key, model in latest.items():
        print(f"{key}: v{model['version']}")
