"""
Auto Signal Scanner
Runs in background every 15 minutes, scans all 45 pairs/timeframes.
Sends Telegram notifications to subscribers when confidence >= 80%.
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger

from ai.predict import load_model_and_predict
from ai.model_store import ModelStore
from payments.subscriptions import DatabaseManager, SubscriptionManager

# Shared instances (passed from server.py)
_db_manager: DatabaseManager = None
_sub_manager: SubscriptionManager = None
_model_store: ModelStore = None
_notification_counts: dict = None  # daily limit counter shared with server.py

# Import bot send function safely
try:
    from bot.bot import send_signal_notification, send_notification
except Exception as e:
    logger.warning(f"Scanner: could not import bot module: {e}")
    async def send_signal_notification(*args, **kwargs): pass
    async def send_notification(*args, **kwargs): pass


def init_scanner(db_manager: DatabaseManager, sub_manager: SubscriptionManager,
                 model_store: ModelStore, notification_counts: dict):
    """Initialize scanner with shared resources from server.py"""
    global _db_manager, _sub_manager, _model_store, _notification_counts
    _db_manager = db_manager
    _sub_manager = sub_manager
    _model_store = model_store
    _notification_counts = notification_counts
    logger.info("Auto signal scanner initialized")


async def _send_to_subscribers(prediction: dict):
    """Send a strong signal to all eligible subscribers"""
    symbol = prediction["symbol"]
    timeframe = prediction["timeframe"]
    signal = prediction["signal"]
    confidence = prediction["confidence"]
    resend_hours = getattr(config, "SCANNER_RESEND_HOURS", 6)

    # Get all paid subscribers who selected this symbol
    subscribers = _sub_manager.get_subscribers_for_symbol(symbol)
    if not subscribers:
        return

    sent_count = 0
    for sub in subscribers:
        user_id = sub["user_id"]
        telegram_id = sub["telegram_id"]
        daily_limit = sub["daily_alerts"]

        # Check daily limit
        from datetime import date
        today = date.today().isoformat()
        if _notification_counts is not None:
            entry = _notification_counts.get(user_id, {"date": None, "count": 0})
            if entry["date"] != today:
                entry = {"date": today, "count": 0}
            if entry["count"] >= daily_limit:
                continue
            entry["count"] += 1
            _notification_counts[user_id] = entry

        # Check deduplication — don't re-send same signal within resend_hours
        if _sub_manager.was_notified_recently(
            user_id, symbol, timeframe, signal, hours=resend_hours
        ):
            continue

        try:
            await send_signal_notification(telegram_id, prediction)
            _sub_manager.log_notification(user_id, symbol, timeframe, signal, confidence)
            sent_count += 1
        except Exception as e:
            logger.error(f"Scanner: failed to send to {telegram_id}: {e}")

    if sent_count > 0:
        logger.info(
            f"Scanner: sent {signal} {symbol} {timeframe} "
            f"({confidence:.1f}%) to {sent_count} subscribers"
        )


async def _notify_admins(text: str):
    """Send a text message to all admins"""
    for admin_id in config.ADMIN_IDS:
        try:
            await send_notification(admin_id, text)
        except Exception:
            pass


async def scan_all_signals():
    """
    Main scanner job — runs every SCANNER_INTERVAL_MINUTES.
    Scans all 45 symbol/timeframe combos, sends notifications for strong signals.
    """
    if _sub_manager is None or _model_store is None:
        logger.warning("Scanner not initialized, skipping scan")
        return

    min_confidence = getattr(config, "AUTO_SIGNAL_MIN_CONFIDENCE", 80)
    start_time = datetime.utcnow()
    strong_signals = []
    errors = 0

    logger.info(
        f"Scanner: starting full scan of {len(config.SUPPORTED_PAIRS)} pairs "
        f"x {len(config.TIMEFRAMES)} timeframes (threshold: {min_confidence}%)"
    )

    for symbol in config.SUPPORTED_PAIRS:
        for timeframe in config.TIMEFRAMES:
            try:
                model_path = _model_store.get_model_path(symbol, timeframe)
                if not model_path or not os.path.exists(model_path):
                    logger.debug(f"Scanner: no model for {symbol} {timeframe}, skipping")
                    continue

                prediction = load_model_and_predict(symbol, timeframe, model_path)

                if prediction.get("error"):
                    logger.debug(f"Scanner: error for {symbol} {timeframe}: {prediction['error']}")
                    errors += 1
                    continue

                signal = prediction.get("signal", "NO TRADE")
                confidence = prediction.get("confidence", 0)

                if signal in ("LONG", "SHORT") and confidence >= min_confidence:
                    logger.info(
                        f"Scanner: STRONG SIGNAL {signal} {symbol} {timeframe} "
                        f"conf={confidence:.1f}%"
                    )
                    strong_signals.append(prediction)
                    await _send_to_subscribers(prediction)

            except Exception as e:
                logger.error(f"Scanner: exception on {symbol} {timeframe}: {e}")
                errors += 1
                continue

            # Small pause between predictions to avoid rate limits
            await asyncio.sleep(0.5)

    elapsed = (datetime.utcnow() - start_time).total_seconds()
    logger.info(
        f"Scanner: scan complete in {elapsed:.0f}s — "
        f"{len(strong_signals)} strong signals, {errors} errors"
    )

    # Notify admins about scan summary if any strong signals found
    if strong_signals:
        lines = [f"🔔 <b>Auto-scan complete</b>: {len(strong_signals)} strong signal(s)\n"]
        for p in strong_signals:
            emoji = "📈" if p["signal"] == "LONG" else "📉"
            lines.append(
                f"{emoji} <b>{p['signal']}</b> {p['symbol']} {p['timeframe']} "
                f"— {p['confidence']:.1f}%"
            )
        await _notify_admins("\n".join(lines))
