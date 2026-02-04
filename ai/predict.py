"""
Inference and Signal Generation
Generates probabilistic predictions and trading signals
Professional approach: AI outputs probabilities, logic generates signals
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger


class SignalGenerator:
    """
    Generates trading signals from model predictions
    Applies volatility filter and confidence thresholds
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: list,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.model.eval()

    def predict_probabilities(self, features: torch.Tensor) -> np.ndarray:
        """
        Get probability distribution from model

        Args:
            features: [batch, seq_len, num_features] or [seq_len, num_features]

        Returns:
            probabilities: [batch, num_classes] or [num_classes]
        """
        with torch.no_grad():
            # Handle single sample
            if features.dim() == 2:
                features = features.unsqueeze(0)  # Add batch dimension

            features = features.to(self.device)
            probabilities = self.model.predict_proba(features)
            probabilities = probabilities.cpu().numpy()

            # Remove batch dimension if single sample
            if probabilities.shape[0] == 1:
                probabilities = probabilities[0]

            return probabilities

    def assess_volatility(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Assess current volatility regime

        Args:
            df: DataFrame with market data and features

        Returns:
            volatility_percentile, regime_label
        """
        # Use ATR percentile as volatility measure
        if "atr_percentile" in df.columns:
            current_vol = df["atr_percentile"].iloc[-1]
        else:
            # Fallback: compute from recent volatility
            recent_vol = df["returns"].iloc[-20:].std()
            historical_vol = df["returns"].std()
            current_vol = recent_vol / (historical_vol + 1e-8)

        # Determine regime
        if current_vol < 0.25:
            regime = "low"
        elif current_vol < 0.60:
            regime = "normal"
        elif current_vol < 0.90:
            regime = "high"
        else:
            regime = "extreme"

        return current_vol, regime

    def generate_signal(
        self,
        probabilities: np.ndarray,
        volatility_percentile: float,
        long_threshold: float = config.LONG_THRESHOLD,
        short_threshold: float = config.SHORT_THRESHOLD,
        max_vol_percentile: float = config.MAX_VOLATILITY_PERCENTILE / 100.0
    ) -> Dict:
        """
        Generate trading signal from probabilities

        NEW LOGIC: Always show the best signal based on highest probability.
        - If LONG has highest prob → LONG signal
        - If SHORT has highest prob → SHORT signal
        - If NO_TRADE has highest prob → NO TRADE

        Signal strength based on confidence:
        - Strong: >60%
        - Moderate: 45-60%
        - Weak: <45%
        """
        prob_no_trade = probabilities[0]
        prob_long = probabilities[1]
        prob_short = probabilities[2]

        # Volatility filter: if extreme, warn but still show signal
        vol_warning = ""
        if volatility_percentile > max_vol_percentile:
            vol_warning = " (High volatility - trade with caution)"

        # Find the highest probability class
        max_prob = max(probabilities)
        max_idx = int(np.argmax(probabilities))

        # Determine signal based on highest probability
        if max_idx == 1:  # LONG has highest probability
            signal = "LONG"
            signal_class = 1
            confidence = prob_long * 100

            # Determine strength
            if prob_long >= 0.60:
                reason = "Strong bullish signal" + vol_warning
            elif prob_long >= 0.45:
                reason = "Moderate bullish signal" + vol_warning
            else:
                reason = "Weak bullish tendency" + vol_warning

        elif max_idx == 2:  # SHORT has highest probability
            signal = "SHORT"
            signal_class = 2
            confidence = prob_short * 100

            if prob_short >= 0.60:
                reason = "Strong bearish signal" + vol_warning
            elif prob_short >= 0.45:
                reason = "Moderate bearish signal" + vol_warning
            else:
                reason = "Weak bearish tendency" + vol_warning

        else:  # NO_TRADE has highest probability (or equal)
            # Check if LONG or SHORT is close (within 5%)
            if prob_long > prob_short and (prob_no_trade - prob_long) < 0.10:
                signal = "LONG"
                signal_class = 1
                confidence = prob_long * 100
                reason = "Slight bullish bias - wait for confirmation" + vol_warning
            elif prob_short > prob_long and (prob_no_trade - prob_short) < 0.10:
                signal = "SHORT"
                signal_class = 2
                confidence = prob_short * 100
                reason = "Slight bearish bias - wait for confirmation" + vol_warning
            else:
                signal = "NO TRADE"
                signal_class = 0
                confidence = prob_no_trade * 100
                reason = "Market conditions unclear - wait for setup" + vol_warning

        return {
            "signal": signal,
            "signal_class": int(signal_class),
            "confidence": round(float(confidence), 2),
            "probabilities": {
                "no_trade": round(float(prob_no_trade * 100), 2),
                "long": round(float(prob_long * 100), 2),
                "short": round(float(prob_short * 100), 2)
            },
            "reason": reason
        }

    def predict_symbol(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Dict:
        """
        Generate prediction for a symbol

        Args:
            df: DataFrame with features (must include last SEQUENCE_LENGTH candles)
            symbol: Trading pair
            timeframe: Timeframe

        Returns:
            Prediction dictionary
        """
        # Check we have enough data
        if len(df) < config.SEQUENCE_LENGTH:
            logger.warning(f"Insufficient data for {symbol} {timeframe}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "signal": "NO TRADE",
                "confidence": 0,
                "error": "Insufficient data"
            }

        # Extract last sequence
        feature_df = df[self.feature_names].iloc[-config.SEQUENCE_LENGTH:]
        feature_array = feature_df.values.astype(np.float32)
        feature_tensor = torch.from_numpy(feature_array)

        # Get probabilities
        probabilities = self.predict_probabilities(feature_tensor)

        # Log probabilities for debugging
        logger.info(f"{symbol} {timeframe} raw probabilities: NO_TRADE={probabilities[0]:.3f}, LONG={probabilities[1]:.3f}, SHORT={probabilities[2]:.3f}")

        # Assess volatility
        volatility_percentile, volatility_regime = self.assess_volatility(df)
        logger.info(f"{symbol} {timeframe} volatility: {volatility_regime} ({volatility_percentile:.2%})")

        # Generate signal
        signal_info = self.generate_signal(probabilities, volatility_percentile)
        logger.info(f"{symbol} {timeframe} signal: {signal_info['signal']} ({signal_info['confidence']:.1f}%)")

        # Get current price
        current_price = df["close"].iloc[-1]
        timestamp = df["timestamp"].iloc[-1]

        # Compile full prediction
        prediction = {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": signal_info["signal"],
            "confidence": signal_info["confidence"],
            "probabilities": signal_info["probabilities"],
            "volatility": {
                "regime": volatility_regime,
                "percentile": round(float(volatility_percentile * 100), 2)
            },
            "price": float(current_price),
            "timestamp": timestamp.isoformat(),
            "model": config.MODEL_TYPE.upper(),
            "reason": signal_info["reason"]
        }

        return prediction


def load_model_and_predict(
    symbol: str,
    timeframe: str,
    model_path: str
) -> Optional[Dict]:
    """
    Load trained model and generate prediction

    Args:
        symbol: Trading pair
        timeframe: Timeframe
        model_path: Path to saved model

    Returns:
        Prediction dictionary
    """
    from data.market import BinanceDataFetcher
    from data.features import FeatureEngine
    from ai.model import create_model

    try:
        # Load model
        logger.info(f"Loading model from {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Create model architecture
        num_features = checkpoint["num_features"]
        model_type = checkpoint["model_type"]
        feature_names = checkpoint["feature_names"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_model(num_features, model_type=model_type, device=device)

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logger.info(f"Model loaded: {model_type} with {num_features} features")

        # Fetch latest data
        logger.info(f"Fetching latest data for {symbol} {timeframe}")
        fetcher = BinanceDataFetcher()
        df = fetcher.fetch_latest_candles(
            symbol,
            timeframe,
            num_candles=config.SEQUENCE_LENGTH + 200  # Extra for feature computation
        )

        if df is None:
            raise RuntimeError(f"Failed to fetch market data for {symbol} {timeframe} from exchange API")

        if len(df) < config.SEQUENCE_LENGTH:
            raise RuntimeError(f"Not enough candles: got {len(df)}, need {config.SEQUENCE_LENGTH}")

        # Create features
        logger.info("Computing features...")
        engine = FeatureEngine()
        df = engine.create_all_features(df)

        # Generate prediction
        signal_gen = SignalGenerator(model, feature_names, device=device)
        prediction = signal_gen.predict_symbol(df, symbol, timeframe)

        return prediction

    except Exception as e:
        logger.error(f"Prediction failed for {symbol} {timeframe}: {e}")
        raise


def format_signal_output(prediction: Dict) -> str:
    """
    Format prediction as human-readable text

    Args:
        prediction: Prediction dictionary

    Returns:
        Formatted string
    """
    output = f"""
{prediction['symbol']} | {prediction['timeframe'].upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal: {prediction['signal']}
Confidence: {prediction['confidence']}%
Volatility: {prediction['volatility']['regime'].capitalize()}
Model: Neural Network
Price: ${prediction['price']:.4f}
Time: {prediction['timestamp'][:19]} UTC

Probabilities:
  LONG: {prediction['probabilities']['long']}%
  SHORT: {prediction['probabilities']['short']}%
  NO TRADE: {prediction['probabilities']['no_trade']}%
"""
    return output


if __name__ == "__main__":
    # Setup logging
    logger.add(
        os.path.join(config.LOGS_DIR, "predictions.log"),
        rotation="10 MB",
        level="INFO"
    )

    # Generate prediction
    symbol = "BTCUSDT"
    timeframe = "1h"
    model_filename = f"{symbol}_{timeframe}_{config.MODEL_TYPE}.pt"
    model_path = os.path.join(config.MODELS_DIR, model_filename)

    prediction = load_model_and_predict(symbol, timeframe, model_path)

    if prediction:
        print(format_signal_output(prediction))
    else:
        print("Failed to generate prediction")
