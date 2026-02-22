"""
Inference and Signal Generation — Hybrid Technical Analysis System
Combines real technical indicators (EMA, RSI, MACD, Stochastic, Momentum, Volume)
with neural network prediction for accurate consensus-based signals.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger


class TechnicalAnalyzer:
    """
    Analyzes real technical indicators from computed features.
    Each indicator votes: +1 (LONG), -1 (SHORT), or 0 (NEUTRAL).
    """

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run all technical indicators and return consensus.
        Uses the LAST row of the DataFrame (current candle).
        """
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        votes = []
        details = []

        # 1. EMA Trend
        vote, detail = self._ema_trend(last)
        votes.append(vote)
        details.append(f"EMA: {detail}")

        # 2. RSI
        vote, detail = self._rsi_signal(last)
        votes.append(vote)
        details.append(f"RSI: {detail}")

        # 3. MACD
        vote, detail = self._macd_signal(last)
        votes.append(vote)
        details.append(f"MACD: {detail}")

        # 4. Stochastic
        vote, detail = self._stochastic_signal(last)
        votes.append(vote)
        details.append(f"Stoch: {detail}")

        # 5. Momentum Alignment
        vote, detail = self._momentum_signal(last)
        votes.append(vote)
        details.append(f"Mom: {detail}")

        # 6. Volume Confirmation
        vote, detail = self._volume_signal(last)
        votes.append(vote)
        details.append(f"Vol: {detail}")

        # Count votes
        long_votes = sum(1 for v in votes if v > 0)
        short_votes = sum(1 for v in votes if v < 0)
        neutral_votes = sum(1 for v in votes if v == 0)
        total = len(votes)

        logger.info(f"TA votes: LONG={long_votes}, SHORT={short_votes}, NEUTRAL={neutral_votes} | {', '.join(details)}")

        return {
            "votes": votes,
            "long_votes": long_votes,
            "short_votes": short_votes,
            "neutral_votes": neutral_votes,
            "total": total,
            "details": details
        }

    def _ema_trend(self, row) -> Tuple[int, str]:
        """EMA 50/200 trend direction"""
        ema50 = row.get("ema_dist_50", 0)
        ema200 = row.get("ema_dist_200", 0)

        # ema_dist is (price - EMA) / ATR, so positive = price above EMA
        if ema50 > 0.3 and ema200 > 0:
            return 1, "Bullish"
        elif ema50 < -0.3 and ema200 < 0:
            return -1, "Bearish"
        elif ema50 > 0.05:
            return 1, "Mild bull"
        elif ema50 < -0.05:
            return -1, "Mild bear"
        return 0, "Neutral"

    def _rsi_signal(self, row) -> Tuple[int, str]:
        """RSI as momentum/trend indicator (NOT contrarian)"""
        rsi14 = row.get("rsi_14", 0)
        rsi_val = rsi14 * 50 + 50  # Convert back to 0-100 scale

        # RSI as trend indicator: above 50 = bullish momentum, below 50 = bearish
        if rsi_val > 60:
            return 1, f"BullMom({rsi_val:.0f})"
        elif rsi_val < 40:
            return -1, f"BearMom({rsi_val:.0f})"
        elif rsi_val > 53:
            return 1, f"MildBull({rsi_val:.0f})"
        elif rsi_val < 47:
            return -1, f"MildBear({rsi_val:.0f})"
        return 0, f"Neutral({rsi_val:.0f})"

    def _macd_signal(self, row) -> Tuple[int, str]:
        """MACD crossover and histogram"""
        macd_cross = row.get("macd_cross", 0)
        macd_hist = row.get("macd_hist_norm", 0)

        if macd_cross > 0 and macd_hist > 0.1:
            return 1, "Bull cross"
        elif macd_cross < 0 and macd_hist < -0.1:
            return -1, "Bear cross"
        elif macd_cross > 0:
            return 1, "Above signal"
        elif macd_cross < 0:
            return -1, "Below signal"
        return 0, "Neutral"

    def _stochastic_signal(self, row) -> Tuple[int, str]:
        """Stochastic as trend indicator (position + crossover)"""
        stoch_k = row.get("stoch_k", 0)
        stoch_cross = row.get("stoch_cross", 0)
        stoch_val = stoch_k * 50 + 50  # Convert to 0-100 scale

        # Stochastic as trend: above 50 + cross up = bullish, below 50 + cross down = bearish
        if stoch_k > 0.1 and stoch_cross > 0:
            return 1, f"BullCross({stoch_val:.0f})"
        elif stoch_k < -0.1 and stoch_cross < 0:
            return -1, f"BearCross({stoch_val:.0f})"
        elif stoch_k > 0.2:
            return 1, f"HighZone({stoch_val:.0f})"
        elif stoch_k < -0.2:
            return -1, f"LowZone({stoch_val:.0f})"
        return 0, f"Mid({stoch_val:.0f})"

    def _momentum_signal(self, row) -> Tuple[int, str]:
        """Multi-timeframe momentum alignment"""
        align = row.get("momentum_align", 0)
        mom_fast = row.get("momentum_fast", 0)

        # momentum_align: [-1, 1], where 1 = all TFs bullish, -1 = all bearish
        if align > 0.5 and mom_fast > 0:
            return 1, "Strong up"
        elif align < -0.5 and mom_fast < 0:
            return -1, "Strong down"
        elif align > 0.2:
            return 1, "Mild up"
        elif align < -0.2:
            return -1, "Mild down"
        return 0, "Mixed"

    def _volume_signal(self, row) -> Tuple[int, str]:
        """Volume pressure and cumulative delta"""
        pressure = row.get("volume_pressure", 0)
        cum_delta = row.get("cum_delta", 0)

        if pressure > 0.5 and cum_delta > 1:
            return 1, "Buy pressure"
        elif pressure < -0.5 and cum_delta < -1:
            return -1, "Sell pressure"
        elif cum_delta > 0.5:
            return 1, "Mild buying"
        elif cum_delta < -0.5:
            return -1, "Mild selling"
        return 0, "Balanced"


class SignalGenerator:
    """
    Hybrid Signal Generator — combines Technical Analysis consensus
    with neural network predictions for accurate trading signals.
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
        self.ta = TechnicalAnalyzer()

    def predict_probabilities(self, features: torch.Tensor) -> np.ndarray:
        """Get probability distribution from model"""
        with torch.no_grad():
            if features.dim() == 2:
                features = features.unsqueeze(0)
            features = features.to(self.device)
            temperature = 0.7
            probabilities = self.model.predict_proba(features, temperature=temperature)
            probabilities = probabilities.cpu().numpy()
            if probabilities.shape[0] == 1:
                probabilities = probabilities[0]
            return probabilities

    def get_model_vote(self, probabilities: np.ndarray) -> Tuple[int, str]:
        """Convert model output to a vote (+1 LONG, -1 SHORT, 0 NEUTRAL)"""
        prob_no_trade = probabilities[0]
        prob_long = probabilities[1]
        prob_short = probabilities[2]

        max_idx = int(np.argmax(probabilities))

        if max_idx == 1 and prob_long > 0.40:
            return 1, f"NN:LONG({prob_long:.0%})"
        elif max_idx == 2 and prob_short > 0.40:
            return -1, f"NN:SHORT({prob_short:.0%})"
        elif max_idx == 1:
            return 1, f"NN:mildL({prob_long:.0%})"
        elif max_idx == 2:
            return -1, f"NN:mildS({prob_short:.0%})"
        return 0, f"NN:neutral({prob_no_trade:.0%})"

    def assess_volatility(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Assess current volatility regime"""
        if "atr_percentile" in df.columns:
            current_vol = df["atr_percentile"].iloc[-1]
        else:
            recent_vol = df["returns"].iloc[-20:].std()
            historical_vol = df["returns"].std()
            current_vol = recent_vol / (historical_vol + 1e-8)

        if current_vol < 0.25:
            regime = "low"
        elif current_vol < 0.60:
            regime = "normal"
        elif current_vol < 0.90:
            regime = "high"
        else:
            regime = "extreme"

        return current_vol, regime

    def generate_hybrid_signal(
        self,
        ta_result: Dict,
        model_vote: int,
        model_detail: str,
        volatility_percentile: float
    ) -> Dict:
        """
        Generate signal from TA consensus + model vote.

        7 voters total: 6 TA indicators + 1 neural network.
        Minimum 4/7 agreement for a signal.
        """
        # Combine all votes
        all_votes = ta_result["votes"] + [model_vote]
        all_details = ta_result["details"] + [model_detail]
        total = len(all_votes)  # 7

        long_votes = sum(1 for v in all_votes if v > 0)
        short_votes = sum(1 for v in all_votes if v < 0)
        neutral_votes = sum(1 for v in all_votes if v == 0)

        # Volatility warning
        vol_warning = ""
        if volatility_percentile > 0.90:
            vol_warning = " (Extreme volatility - be careful)"
        elif volatility_percentile > 0.75:
            vol_warning = " (High volatility)"

        # Determine signal based on consensus
        # Signal given when one side has clear majority over the other
        # Confidence = winning / (winning + losing) * 100

        if long_votes > short_votes and long_votes >= 3:
            signal = "LONG"
            signal_class = 1
            directional = long_votes + short_votes
            confidence = (long_votes / directional) * 100 if directional > 0 else 50

            if long_votes >= 6:
                reason = f"Strong bullish consensus ({long_votes}/{total} indicators){vol_warning}"
            elif long_votes >= 5:
                reason = f"Bullish signal ({long_votes}/{total} indicators agree){vol_warning}"
            elif long_votes >= 4:
                reason = f"Moderate bullish signal ({long_votes}/{total} indicators){vol_warning}"
            else:
                reason = f"Weak bullish tendency ({long_votes}/{total} indicators){vol_warning}"

        elif short_votes > long_votes and short_votes >= 3:
            signal = "SHORT"
            signal_class = 2
            directional = long_votes + short_votes
            confidence = (short_votes / directional) * 100 if directional > 0 else 50

            if short_votes >= 6:
                reason = f"Strong bearish consensus ({short_votes}/{total} indicators){vol_warning}"
            elif short_votes >= 5:
                reason = f"Bearish signal ({short_votes}/{total} indicators agree){vol_warning}"
            elif short_votes >= 4:
                reason = f"Moderate bearish signal ({short_votes}/{total} indicators){vol_warning}"
            else:
                reason = f"Weak bearish tendency ({short_votes}/{total} indicators){vol_warning}"

        else:
            # Tied or no clear direction
            signal = "NO TRADE"
            signal_class = 0
            confidence = 0
            reason = f"No consensus (L:{long_votes} S:{short_votes} N:{neutral_votes}) - wait for setup{vol_warning}"

        # Build probabilities from votes for API compatibility
        prob_long = long_votes / total
        prob_short = short_votes / total
        prob_no_trade = neutral_votes / total

        logger.info(f"Hybrid signal: {signal} {confidence:.1f}% | Votes: L={long_votes} S={short_votes} N={neutral_votes} | {', '.join(all_details)}")

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
        Generate prediction using hybrid TA + model consensus.
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

        # Step 1: Technical Analysis (real indicators)
        ta_result = self.ta.analyze(df)

        # Step 2: Neural Network prediction (1 vote)
        feature_df = df[self.feature_names].iloc[-config.SEQUENCE_LENGTH:]
        feature_array = feature_df.values.astype(np.float32)
        feature_tensor = torch.from_numpy(feature_array)
        probabilities = self.predict_probabilities(feature_tensor)
        model_vote, model_detail = self.get_model_vote(probabilities)

        logger.info(f"{symbol} {timeframe} NN raw: NO_TRADE={probabilities[0]:.3f}, LONG={probabilities[1]:.3f}, SHORT={probabilities[2]:.3f}")

        # Step 3: Assess volatility
        volatility_percentile, volatility_regime = self.assess_volatility(df)
        logger.info(f"{symbol} {timeframe} volatility: {volatility_regime} ({volatility_percentile:.2%})")

        # Step 4: Generate hybrid signal from consensus
        signal_info = self.generate_hybrid_signal(
            ta_result, model_vote, model_detail, volatility_percentile
        )
        logger.info(f"{symbol} {timeframe} FINAL: {signal_info['signal']} ({signal_info['confidence']:.1f}%)")

        # Get current price
        current_price = df["close"].iloc[-1]
        timestamp = df["timestamp"].iloc[-1]

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
            "model": "HYBRID_TA",
            "reason": signal_info["reason"]
        }

        return prediction


def load_model_and_predict(
    symbol: str,
    timeframe: str,
    model_path: str
) -> Optional[Dict]:
    """
    Load trained model and generate prediction using hybrid TA system.
    """
    from data.market import BinanceDataFetcher
    from data.features import FeatureEngine
    from ai.model import create_model

    try:
        logger.info(f"Loading model from {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        num_features = checkpoint["num_features"]
        model_type = checkpoint["model_type"]
        feature_names = checkpoint["feature_names"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_model(num_features, model_type=model_type, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logger.info(f"Model loaded: {model_type} with {num_features} features")

        # Fetch latest data
        logger.info(f"Fetching latest data for {symbol} {timeframe}")
        fetcher = BinanceDataFetcher()
        df = fetcher.fetch_latest_candles(
            symbol,
            timeframe,
            num_candles=config.SEQUENCE_LENGTH + 200
        )

        if df is None:
            raise RuntimeError(f"Failed to fetch market data for {symbol} {timeframe} from exchange API")

        if len(df) < config.SEQUENCE_LENGTH:
            raise RuntimeError(f"Not enough candles: got {len(df)}, need {config.SEQUENCE_LENGTH}")

        # Create features (computes all TA indicators)
        logger.info("Computing features...")
        engine = FeatureEngine()
        df = engine.create_all_features(df)

        # Generate hybrid prediction
        signal_gen = SignalGenerator(model, feature_names, device=device)
        prediction = signal_gen.predict_symbol(df, symbol, timeframe)

        return prediction

    except Exception as e:
        logger.error(f"Prediction failed for {symbol} {timeframe}: {e}")
        raise


def format_signal_output(prediction: Dict) -> str:
    """Format prediction as human-readable text"""
    output = f"""
{prediction['symbol']} | {prediction['timeframe'].upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal: {prediction['signal']}
Confidence: {prediction['confidence']}%
Volatility: {prediction['volatility']['regime'].capitalize()}
Model: Hybrid TA + Neural Network
Price: ${prediction['price']:.4f}
Time: {prediction['timestamp'][:19]} UTC
Reason: {prediction['reason']}

Votes:
  LONG: {prediction['probabilities']['long']}%
  SHORT: {prediction['probabilities']['short']}%
  NO TRADE: {prediction['probabilities']['no_trade']}%
"""
    return output


if __name__ == "__main__":
    logger.add(
        os.path.join(config.LOGS_DIR, "predictions.log"),
        rotation="10 MB",
        level="INFO"
    )

    symbol = "BTCUSDT"
    timeframe = "1h"
    model_filename = f"{symbol}_{timeframe}_{config.MODEL_TYPE}.pt"
    model_path = os.path.join(config.MODELS_DIR, model_filename)

    prediction = load_model_and_predict(symbol, timeframe, model_path)

    if prediction:
        print(format_signal_output(prediction))
    else:
        print("Failed to generate prediction")
