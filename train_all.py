"""
Train all AI models for all supported pairs and timeframes
Includes: Transformer + Ensemble (XGBoost + LightGBM) + Regime + Calibration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from ai.train import train_model, load_cached_data, Trainer, FocalLoss
from data.features import FeatureEngine, create_labels
from data.datasets import create_dataloaders, get_class_weights
from ai.model import create_model
import config
from loguru import logger


def train_ensemble_for_symbol(symbol: str, timeframe: str, df, feature_cols, val_df=None):
    """Train ensemble models (XGBoost + LightGBM + Regime) for a symbol."""
    try:
        from ai.ensemble import EnsembleManager
        ensemble = EnsembleManager()

        # Split data for ensemble training
        split_idx = int(len(df) * config.TRAIN_TEST_SPLIT)
        train_df = df.iloc[:split_idx]
        ens_val_df = df.iloc[split_idx:] if val_df is None else val_df

        ensemble.train_all(train_df, feature_cols, ens_val_df)
        ensemble.save_all(config.MODELS_DIR, symbol, timeframe)
        logger.info(f"Ensemble models saved for {symbol} {timeframe}")
    except Exception as e:
        logger.warning(f"Ensemble training failed for {symbol} {timeframe}: {e}")

    # Train regime detector
    try:
        from ai.regime import RegimeDetector
        regime = RegimeDetector()
        regime.train(df)
        regime_path = os.path.join(config.MODELS_DIR, f"{symbol}_{timeframe}_regime.pkl")
        regime.save(regime_path)
        logger.info(f"Regime detector saved for {symbol} {timeframe}")
    except Exception as e:
        logger.warning(f"Regime training failed for {symbol} {timeframe}: {e}")

    # Train calibrator on validation predictions
    try:
        from ai.calibration import PlattCalibrator

        model_filename = f"{symbol}_{timeframe}_{config.MODEL_TYPE}.pt"
        model_path = os.path.join(config.MODELS_DIR, model_filename)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            model = create_model(checkpoint["num_features"], model_type=checkpoint["model_type"], device="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            split_idx = int(len(df) * config.TRAIN_TEST_SPLIT)
            cal_df = df.iloc[split_idx:]

            if len(cal_df) > 100:
                feat_names = checkpoint["feature_names"]

                # Only use features that exist in df
                valid_feats = [f for f in feat_names if f in cal_df.columns]
                if len(valid_feats) == len(feat_names):
                    from data.datasets import CryptoSequenceDataset
                    from torch.utils.data import DataLoader

                    cal_dataset = CryptoSequenceDataset(cal_df, valid_feats, config.SEQUENCE_LENGTH)
                    cal_loader = DataLoader(cal_dataset, batch_size=64, shuffle=False)

                    all_probs = []
                    all_labels = []

                    with torch.no_grad():
                        for features, labels in cal_loader:
                            probs = model.predict_proba(features, temperature=0.7)
                            all_probs.append(probs.numpy())
                            all_labels.append(labels.numpy())

                    raw_probs = np.concatenate(all_probs, axis=0)
                    true_labels = np.concatenate(all_labels, axis=0)

                    calibrator = PlattCalibrator()
                    calibrator.train(raw_probs, true_labels)
                    cal_path = os.path.join(config.MODELS_DIR, f"{symbol}_{timeframe}_calibration.pkl")
                    calibrator.save(cal_path)
                    logger.info(f"Calibrator saved for {symbol} {timeframe}")
    except Exception as e:
        logger.warning(f"Calibration training failed for {symbol} {timeframe}: {e}")


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

                # Step 1: Train Transformer model
                train_model(
                    symbol=pair,
                    timeframe=timeframe,
                    save_path=model_path
                )
                logger.success(f"Transformer trained: {pair} {timeframe}")

                # Step 2: Train ensemble + regime + calibration
                logger.info(f"Training ensemble for {pair} {timeframe}...")
                df = load_cached_data(pair, timeframe)
                if df is None:
                    from data.market import BinanceDataFetcher
                    fetcher = BinanceDataFetcher()
                    df = fetcher.fetch_or_load(pair, timeframe)

                if df is not None:
                    # Cap data size for speed
                    max_candles = getattr(config, 'MAX_TRAINING_CANDLES', 25000)
                    if len(df) > max_candles:
                        df = df.tail(max_candles).reset_index(drop=True)

                    # Load BTC data for correlation
                    btc_df = None
                    if pair.upper() != "BTCUSDT":
                        btc_df = load_cached_data("BTCUSDT", timeframe)
                        if btc_df is not None and len(btc_df) > max_candles:
                            btc_df = btc_df.tail(max_candles).reset_index(drop=True)

                    engine = FeatureEngine()
                    df = engine.create_all_features(df, btc_df=btc_df, symbol=pair)
                    df = create_labels(df)
                    feature_cols = engine.get_feature_names()

                    train_ensemble_for_symbol(pair, timeframe, df, feature_cols)

                logger.success(f"Completed: {pair} {timeframe} -> {model_path}")

            except Exception as e:
                logger.error(f"Failed: {pair} {timeframe} - {str(e)}")
                continue

    logger.info("=" * 60)
    logger.success(f"Training completed! {current} models processed")

if __name__ == "__main__":
    logger.add(
        os.path.join(config.LOGS_DIR, "training.log"),
        rotation="100 MB",
        level="INFO"
    )
    train_all_models()
