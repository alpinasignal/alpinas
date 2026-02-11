"""
Retrain All Models with Improved Architecture

This script retrains all 45 models (15 pairs x 3 timeframes)
with the new improved features and architecture.

Run this locally or on a machine with good CPU/GPU.
Training will take several hours.
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from ai.train import train_model
from loguru import logger

# Setup logging
logger.add(
    os.path.join(config.LOGS_DIR, f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    rotation="100 MB",
    level="INFO"
)


def retrain_all_models():
    """
    Retrain all models with improved architecture
    """
    logger.info("=" * 60)
    logger.info("RETRAINING ALL MODELS WITH IMPROVED ARCHITECTURE")
    logger.info("=" * 60)
    logger.info(f"Model type: {config.MODEL_TYPE}")
    logger.info(f"Hidden dim: {config.HIDDEN_DIM}")
    logger.info(f"Num layers: {config.NUM_LAYERS}")
    logger.info(f"Num heads: {config.NUM_HEADS}")
    logger.info(f"Max epochs: {config.MAX_EPOCHS}")
    logger.info("=" * 60)

    total_models = len(config.SUPPORTED_PAIRS) * len(config.TIMEFRAMES)
    completed = 0
    failed = []

    for symbol in config.SUPPORTED_PAIRS:
        for timeframe in config.TIMEFRAMES:
            completed += 1
            logger.info(f"\n[{completed}/{total_models}] Training {symbol} {timeframe}")

            try:
                model_filename = f"{symbol}_{timeframe}_{config.MODEL_TYPE}.pt"
                model_path = os.path.join(config.MODELS_DIR, model_filename)

                # Backup old model (delete old backup if exists)
                if os.path.exists(model_path):
                    backup_path = model_path.replace(".pt", "_backup.pt")
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(model_path, backup_path)
                    logger.info(f"Backed up old model to {backup_path}")

                # Train new model
                model = train_model(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=config.MODEL_TYPE,
                    save_path=model_path
                )

                if model is not None:
                    logger.info(f"Successfully trained {symbol} {timeframe}")
                else:
                    failed.append(f"{symbol}_{timeframe}")
                    logger.error(f"Failed to train {symbol} {timeframe}")

            except Exception as e:
                failed.append(f"{symbol}_{timeframe}")
                logger.error(f"Error training {symbol} {timeframe}: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total models: {total_models}")
    logger.info(f"Successful: {total_models - len(failed)}")
    logger.info(f"Failed: {len(failed)}")

    if failed:
        logger.warning(f"Failed models: {failed}")

    return failed


def retrain_single_model(symbol: str, timeframe: str):
    """
    Retrain a single model
    """
    logger.info(f"Retraining {symbol} {timeframe}")

    model_filename = f"{symbol}_{timeframe}_{config.MODEL_TYPE}.pt"
    model_path = os.path.join(config.MODELS_DIR, model_filename)

    model = train_model(
        symbol=symbol,
        timeframe=timeframe,
        model_type=config.MODEL_TYPE,
        save_path=model_path
    )

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrain crypto prediction models")
    parser.add_argument("--all", action="store_true", help="Retrain all models")
    parser.add_argument("--symbol", type=str, help="Symbol to retrain (e.g., BTCUSDT)")
    parser.add_argument("--timeframe", type=str, help="Timeframe to retrain (e.g., 1h)")

    args = parser.parse_args()

    if args.all:
        retrain_all_models()
    elif args.symbol and args.timeframe:
        retrain_single_model(args.symbol, args.timeframe)
    else:
        print("Usage:")
        print("  Retrain all models:    python retrain_models.py --all")
        print("  Retrain single model:  python retrain_models.py --symbol BTCUSDT --timeframe 1h")
