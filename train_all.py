"""
Train all AI models for all supported pairs and timeframes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.train import train_model
import config
from loguru import logger

def train_all_models():
    """Train models for all pairs and timeframes"""

    pairs = config.SUPPORTED_PAIRS
    timeframes = config.TIMEFRAMES

    total = len(pairs) * len(timeframes)
    current = 0

    logger.info(f"=== Starting training for ALL models ===")
    logger.info(f"Total models to train: {total}")
    logger.info(f"Pairs: {pairs}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info("=" * 60)

    for pair in pairs:
        for timeframe in timeframes:
            current += 1
            logger.info(f"\n[{current}/{total}] Training {pair} {timeframe}")
            logger.info("=" * 60)

            try:
                # Build save path for model file
                model_filename = f"{pair}_{timeframe}_{config.MODEL_TYPE}.pt"
                model_path = os.path.join(config.MODELS_DIR, model_filename)

                train_model(
                    symbol=pair,
                    timeframe=timeframe,
                    save_path=model_path
                )
                logger.success(f"Completed: {pair} {timeframe} -> {model_path}")

            except Exception as e:
                logger.error(f"❌ Failed: {pair} {timeframe} - {str(e)}")
                continue

    logger.info("=" * 60)
    logger.success(f"Training completed! {current} models processed")

if __name__ == "__main__":
    train_all_models()
