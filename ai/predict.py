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

        # 7. Price Action (recent candles direction)
        vote, detail = self._price_action_signal(df)
        votes.append(vote)
        details.append(f"PA: {detail}")

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
        ema50 = row.get("ema_distance_50", 0)
        ema200 = row.get("ema_distance_200", 0)

        # ema_distance is (price - EMA) / ATR, so positive = price above EMA
        if ema50 > 0.5 and ema200 > 0:
            return 1, "Bullish"
        elif ema50 < -0.5 and ema200 < 0:
            return -1, "Bearish"
        elif ema50 > 0.1:
            return 1, "Mild bull"
        elif ema50 < -0.1:
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


    def _price_action_signal(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Recent price action — are last N candles going up or down?"""
        if len(df) < 5:
            return 0, "NoData"

        # Compare close of last 5 candles
        recent = df["close"].iloc[-5:]
        first_price = recent.iloc[0]
        last_price = recent.iloc[-1]
        change_pct = (last_price - first_price) / first_price * 100

        # Count bullish vs bearish candles in last 5
        directions = df["candle_direction"].iloc[-5:] if "candle_direction" in df.columns else pd.Series([0]*5)
        bull_candles = (directions > 0).sum()
        bear_candles = (directions < 0).sum()

        if change_pct > 0.3 and bull_candles >= 3:
            return 1, f"Up({change_pct:.1f}%)"
        elif change_pct < -0.3 and bear_candles >= 3:
            return -1, f"Down({change_pct:.1f}%)"
        elif change_pct > 0.1:
            return 1, f"MildUp({change_pct:.1f}%)"
        elif change_pct < -0.1:
            return -1, f"MildDn({change_pct:.1f}%)"
        return 0, f"Flat({change_pct:.1f}%)"


class SignalGenerator:
    """
    Hybrid Signal Generator v2 — combines Technical Analysis consensus
    with neural network + ensemble + regime + meta-model for accurate signals.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: list,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ensemble_manager=None,
        regime_detector=None,
        meta_model_inst=None,
        calibrator=None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.model.eval()
        self.ta = TechnicalAnalyzer()
        self.ensemble = ensemble_manager
        self.regime = regime_detector
        self.meta_model_inst = meta_model_inst
        self.calibrator = calibrator

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
        volatility_percentile: float,
        extra_votes: List = None,
        extra_details: List = None,
        nn_probabilities: np.ndarray = None,
    ) -> Dict:
        """
        Generate signal from TA consensus + model vote + ensemble votes.

        NN counts as 3 votes (not 1) — it's trained to predict the future,
        while TA indicators are lagging. Confidence blends vote consensus with
        NN probability so a high-agreement TA signal can't reach 99% if the
        NN disagrees.
        """
        # NN counts as 3 votes to better balance trained predictor vs lagging TA
        nn_weight = 3
        ta_votes = ta_result["votes"]
        ta_details = ta_result["details"]

        # Expand NN vote to nn_weight votes
        nn_votes_list = [model_vote] * nn_weight
        nn_details_list = [model_detail] + [""] * (nn_weight - 1)

        all_votes = ta_votes + nn_votes_list
        all_details = ta_details + [model_detail]

        if extra_votes:
            all_votes.extend(extra_votes)
        if extra_details:
            all_details.extend(extra_details)

        total = len(all_votes)
        long_votes = sum(1 for v in all_votes if v > 0)
        short_votes = sum(1 for v in all_votes if v < 0)
        neutral_votes = sum(1 for v in all_votes if v == 0)

        # Volatility warning
        vol_warning = ""
        if volatility_percentile > 0.90:
            vol_warning = " (Extreme volatility - be careful)"
        elif volatility_percentile > 0.75:
            vol_warning = " (High volatility)"

        # Need stricter majority: at least 55% of all votes
        min_votes_for_signal = max(4, int(total * 0.50))

        if long_votes > short_votes and long_votes >= min_votes_for_signal:
            signal = "LONG"
            signal_class = 1
            directional = long_votes + short_votes
            vote_conf = (long_votes / directional) * 100 if directional > 0 else 50

            if long_votes >= int(total * 0.75):
                reason = f"Strong bullish consensus ({long_votes}/{total} indicators){vol_warning}"
            elif long_votes >= int(total * 0.65):
                reason = f"Bullish signal ({long_votes}/{total} indicators agree){vol_warning}"
            else:
                reason = f"Moderate bullish signal ({long_votes}/{total} indicators){vol_warning}"

        elif short_votes > long_votes and short_votes >= min_votes_for_signal:
            signal = "SHORT"
            signal_class = 2
            directional = long_votes + short_votes
            vote_conf = (short_votes / directional) * 100 if directional > 0 else 50

            if short_votes >= int(total * 0.75):
                reason = f"Strong bearish consensus ({short_votes}/{total} indicators){vol_warning}"
            elif short_votes >= int(total * 0.65):
                reason = f"Bearish signal ({short_votes}/{total} indicators agree){vol_warning}"
            else:
                reason = f"Moderate bearish signal ({short_votes}/{total} indicators){vol_warning}"

        else:
            signal = "NO TRADE"
            signal_class = 0
            vote_conf = 0
            reason = f"No consensus (L:{long_votes} S:{short_votes} N:{neutral_votes}) - wait for setup{vol_warning}"

        # --- Blend NN probability into confidence ---
        # NN was trained to predict direction → weight it 50/50 with vote consensus
        if signal != "NO TRADE" and nn_probabilities is not None:
            nn_idx = 1 if signal == "LONG" else 2
            nn_prob_for_signal = float(nn_probabilities[nn_idx])  # 0-1
            nn_conf = nn_prob_for_signal * 100  # Convert to 0-100

            # 50% vote consensus + 50% NN probability
            confidence = vote_conf * 0.50 + nn_conf * 0.50

            # Penalty: if NN strongly favors the OPPOSITE direction, reduce confidence
            opposite_idx = 2 if signal == "LONG" else 1
            nn_opposite = float(nn_probabilities[opposite_idx])
            if nn_opposite > 0.50:
                confidence *= 0.60  # NN contradicts → big penalty
                reason += f" [NN disagrees: {nn_opposite:.0%} {('SHORT' if signal == 'LONG' else 'LONG')}]"
            elif nn_opposite > 0.35:
                confidence *= 0.80  # Mild disagreement
        else:
            confidence = vote_conf

        # Hard cap: signals are never 100% certain
        confidence = min(float(confidence), 85.0)

        # Build probabilities for API
        prob_long = long_votes / total
        prob_short = short_votes / total
        prob_no_trade = neutral_votes / total

        logger.info(
            f"Hybrid signal: {signal} {confidence:.1f}% | "
            f"Votes: L={long_votes} S={short_votes} N={neutral_votes} | "
            f"{', '.join(all_details)}"
        )

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

    def get_higher_tf_bias(self, symbol: str, timeframe: str) -> Dict:
        """
        Get 4H timeframe bias for multi-timeframe hierarchy.
        Only applies when predicting 15m or 1h.
        Returns: {"direction": "LONG"/"SHORT"/"NEUTRAL", "strength": int, "veto": bool}
        """
        if timeframe == "4h":
            return {"direction": "NEUTRAL", "strength": 0, "veto": False}

        try:
            from data.market import BinanceDataFetcher
            from data.features import FeatureEngine

            fetcher = BinanceDataFetcher()
            df_4h = fetcher.fetch_latest_candles(
                symbol, "4h", num_candles=config.SEQUENCE_LENGTH + 200
            )

            if df_4h is None or len(df_4h) < 50:
                return {"direction": "NEUTRAL", "strength": 0, "veto": False}

            # Compute features for 4H
            engine = FeatureEngine()
            df_4h = engine.create_all_features(df_4h, symbol=symbol)

            # Run TA analysis on 4H
            ta_4h = self.ta.analyze(df_4h)
            long_v = ta_4h["long_votes"]
            short_v = ta_4h["short_votes"]

            if long_v >= 5:
                direction = "LONG"
            elif short_v >= 5:
                direction = "SHORT"
            elif long_v > short_v:
                direction = "LONG"
            elif short_v > long_v:
                direction = "SHORT"
            else:
                direction = "NEUTRAL"

            strength = max(long_v, short_v)
            # Veto if 4H strongly opposes (6+ votes in opposite direction)
            veto = strength >= 6

            logger.info(f"4H bias for {symbol}: {direction} (L={long_v}, S={short_v}, veto={veto})")
            return {"direction": direction, "strength": strength, "veto": veto}

        except Exception as e:
            logger.warning(f"Could not get 4H bias: {e}")
            return {"direction": "NEUTRAL", "strength": 0, "veto": False}

    def compute_atr_sl_tp(self, df: pd.DataFrame, signal: str, current_price: float) -> Dict:
        """
        Compute ATR-based Stop Loss and Take Profit levels.
        """
        atr_norm = df["atr_norm"].iloc[-1] if "atr_norm" in df.columns else 0.02
        atr_price = atr_norm * current_price  # ATR in price units

        sl_mult = config.SL_ATR_MULTIPLIER
        tp_mult = config.TP_ATR_MULTIPLIER

        if signal == "LONG":
            stop_loss = current_price - (sl_mult * atr_price)
            take_profit = current_price + (tp_mult * atr_price)
        elif signal == "SHORT":
            stop_loss = current_price + (sl_mult * atr_price)
            take_profit = current_price - (tp_mult * atr_price)
        else:
            stop_loss = 0
            take_profit = 0

        return {
            "stop_loss": round(float(stop_loss), 6),
            "take_profit": round(float(take_profit), 6),
            "atr_value": round(float(atr_price), 6)
        }

    def predict_symbol(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Dict:
        """
        Generate prediction using hybrid TA + model consensus + MTF hierarchy.
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

        # Calibrate if calibrator available
        if self.calibrator is not None:
            probabilities = self.calibrator.calibrate(probabilities)

        model_vote, model_detail = self.get_model_vote(probabilities)

        logger.info(f"{symbol} {timeframe} NN raw: NO_TRADE={probabilities[0]:.3f}, LONG={probabilities[1]:.3f}, SHORT={probabilities[2]:.3f}")

        # Step 2b: Ensemble predictions (XGBoost + LightGBM + Statistical)
        ensemble_vote = 0
        ensemble_detail = "Ens:off"
        ensemble_preds = {}
        if self.ensemble is not None:
            try:
                ensemble_preds = self.ensemble.predict_all(df, self.feature_names)
                ensemble_vote, ensemble_detail = self.ensemble.get_ensemble_vote(df, self.feature_names)
                logger.info(f"{symbol} {timeframe} Ensemble: {ensemble_detail}")
            except Exception as e:
                logger.warning(f"Ensemble prediction failed: {e}")

        # Step 2c: Regime detection
        regime_info = {"regime": "unknown", "confidence_adj": 1.0}
        if self.regime is not None:
            try:
                regime_info = self.regime.detect(df)
                logger.info(f"{symbol} {timeframe} Regime: {regime_info['regime']} (adj={regime_info['confidence_adj']})")
            except Exception as e:
                logger.warning(f"Regime detection failed: {e}")

        # Step 2d: Meta-model (if trained, combines all model outputs)
        meta_vote = 0
        meta_detail = "Meta:off"
        if self.meta_model_inst is not None:
            try:
                meta_vote, meta_detail = self.meta_model_inst.get_vote(
                    probabilities, ensemble_preds, regime_info, ta_result
                )
                logger.info(f"{symbol} {timeframe} Meta: {meta_detail}")
            except Exception as e:
                logger.warning(f"Meta-model failed: {e}")

        # Step 3: Assess volatility
        volatility_percentile, volatility_regime = self.assess_volatility(df)
        logger.info(f"{symbol} {timeframe} volatility: {volatility_regime} ({volatility_percentile:.2%})")

        # Step 4: Generate hybrid signal from consensus
        # Now with up to 10 voters: 7 TA + NN + Ensemble + Meta
        extra_votes = []
        extra_details = []
        if self.ensemble is not None:
            extra_votes.append(ensemble_vote)
            extra_details.append(ensemble_detail)
        if self.meta_model_inst is not None:
            extra_votes.append(meta_vote)
            extra_details.append(meta_detail)

        signal_info = self.generate_hybrid_signal(
            ta_result, model_vote, model_detail, volatility_percentile,
            extra_votes=extra_votes, extra_details=extra_details,
            nn_probabilities=probabilities,
        )

        # Apply regime confidence adjustment
        if regime_info["confidence_adj"] != 1.0 and signal_info["confidence"] > 0:
            signal_info["confidence"] = round(
                signal_info["confidence"] * regime_info["confidence_adj"], 2
            )
            signal_info["confidence"] = min(signal_info["confidence"], 99)

        # Step 5: Multi-timeframe hierarchy — 4H veto for 15m/1h
        if timeframe in ("15m", "1h") and signal_info["signal"] != "NO TRADE":
            htf_bias = self.get_higher_tf_bias(symbol, timeframe)

            if htf_bias["veto"]:
                # Check if 4H opposes our signal
                signal_dir = signal_info["signal"]  # "LONG" or "SHORT"
                htf_dir = htf_bias["direction"]

                if (signal_dir == "LONG" and htf_dir == "SHORT") or \
                   (signal_dir == "SHORT" and htf_dir == "LONG"):
                    logger.info(f"4H VETO: {signal_dir} vetoed by strong 4H {htf_dir}")
                    signal_info["signal"] = "NO TRADE"
                    signal_info["signal_class"] = 0
                    signal_info["confidence"] = 0
                    signal_info["reason"] = f"Vetoed by 4H trend ({htf_dir}). Wait for alignment."

            elif htf_bias["direction"] == signal_info["signal"]:
                # 4H confirms — boost confidence by 5%
                signal_info["confidence"] = min(signal_info["confidence"] + 5, 99)
                signal_info["reason"] += f" [4H confirms {htf_bias['direction']}]"

        logger.info(f"{symbol} {timeframe} FINAL: {signal_info['signal']} ({signal_info['confidence']:.1f}%)")

        # Get current price
        current_price = df["close"].iloc[-1]
        timestamp = df["timestamp"].iloc[-1]

        # Step 6: Compute ATR-based SL/TP
        sl_tp = self.compute_atr_sl_tp(df, signal_info["signal"], current_price)

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
            "model": "HYBRID_TA_v2",
            "reason": signal_info["reason"],
            "stop_loss": sl_tp["stop_loss"],
            "take_profit": sl_tp["take_profit"],
            "atr_value": sl_tp["atr_value"]
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

        # Load BTC data for correlation features
        btc_df = None
        if symbol.upper() != "BTCUSDT":
            try:
                btc_df = fetcher.fetch_latest_candles(
                    "BTCUSDT", timeframe,
                    num_candles=config.SEQUENCE_LENGTH + 200
                )
            except Exception as e:
                logger.warning(f"Could not fetch BTC data for correlation: {e}")

        # Create features (computes all TA indicators + institutional features)
        logger.info("Computing features...")
        engine = FeatureEngine()
        df = engine.create_all_features(df, btc_df=btc_df, symbol=symbol)

        # Load ensemble models (optional — graceful fallback if not available)
        ensemble_mgr = None
        regime_det = None
        meta_mdl = None
        calibrator_inst = None

        try:
            from ai.ensemble import EnsembleManager
            ensemble_mgr = EnsembleManager()
            ensemble_mgr.load_all(config.MODELS_DIR, symbol, timeframe)
            if ensemble_mgr.xgb_model.is_trained or ensemble_mgr.lgb_model.is_trained:
                logger.info("Ensemble models loaded")
            else:
                ensemble_mgr = None  # No trained ensemble models
        except Exception as e:
            logger.debug(f"Ensemble not available: {e}")

        try:
            from ai.regime import RegimeDetector
            regime_det = RegimeDetector()
            regime_path = os.path.join(config.MODELS_DIR, f"{symbol}_{timeframe}_regime.pkl")
            regime_det.load(regime_path)
            if not regime_det.is_trained:
                regime_det = None
        except Exception as e:
            logger.debug(f"Regime detector not available: {e}")

        try:
            from ai.meta_model import MetaModel
            meta_mdl = MetaModel()
            meta_path = os.path.join(config.MODELS_DIR, f"{symbol}_{timeframe}_meta.pkl")
            meta_mdl.load(meta_path)
            if not meta_mdl.is_trained:
                meta_mdl = None
        except Exception as e:
            logger.debug(f"Meta-model not available: {e}")

        try:
            from ai.calibration import PlattCalibrator
            calibrator_inst = PlattCalibrator()
            cal_path = os.path.join(config.MODELS_DIR, f"{symbol}_{timeframe}_calibration.pkl")
            calibrator_inst.load(cal_path)
            if not calibrator_inst.is_trained:
                calibrator_inst = None
        except Exception as e:
            logger.debug(f"Calibrator not available: {e}")

        # Generate hybrid prediction with all available models
        signal_gen = SignalGenerator(
            model, feature_names, device=device,
            ensemble_manager=ensemble_mgr,
            regime_detector=regime_det,
            meta_model_inst=meta_mdl,
            calibrator=calibrator_inst,
        )
        prediction = signal_gen.predict_symbol(df, symbol, timeframe)

        return prediction

    except Exception as e:
        logger.error(f"Prediction failed for {symbol} {timeframe}: {e}")
        raise


def format_signal_output(prediction: Dict) -> str:
    """Format prediction as human-readable text"""
    sl = prediction.get('stop_loss', 0)
    tp = prediction.get('take_profit', 0)
    atr = prediction.get('atr_value', 0)

    sl_tp_line = ""
    if sl and tp:
        sl_tp_line = f"\nStop Loss: ${sl:.4f}\nTake Profit: ${tp:.4f}\nATR: ${atr:.4f}"

    output = f"""
{prediction['symbol']} | {prediction['timeframe'].upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal: {prediction['signal']}
Confidence: {prediction['confidence']}%
Volatility: {prediction['volatility']['regime'].capitalize()}
Model: {prediction.get('model', 'HYBRID_TA_v2')}
Price: ${prediction['price']:.4f}{sl_tp_line}
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
