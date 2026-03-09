"""
Telegram Bot Utilities for Alpina Signal
Provides helper functions for sending messages and notifications
NO POLLING - Used only as utility module for Mini App

For Mini App architecture:
- No bot.run_polling()
- No getUpdates loop
- Only message sending functions
"""

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger
from payments.subscriptions import DatabaseManager, SubscriptionManager

# Initialize database
db_manager = DatabaseManager()
sub_manager = SubscriptionManager(db_manager)

# Initialize bot instance (for sending messages only)
bot = None
if config.TELEGRAM_BOT_TOKEN:
    bot = Bot(token=config.TELEGRAM_BOT_TOKEN)


# ========================
# UTILITY FUNCTIONS (NO POLLING)
# ========================

async def send_welcome_message(telegram_id: int, username: str = None, first_name: str = None):
    """
    Send welcome message to user (called from API/webhook)
    NO POLLING - This is a utility function
    """
    if not bot:
        logger.warning("Bot not initialized, cannot send message")
        return

    try:
        # Register user
        user = sub_manager.create_user(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name
        )

        # Get subscription info (with fallback handling)
        subscription = sub_manager.get_subscription(user.id if user else None)

        # Handle fallback mode when no database
        if subscription:
            tier = subscription.tier.upper()
            predictions_used = subscription.predictions_used
            predictions_limit = subscription.predictions_limit
        else:
            # Fallback mode - no database
            tier = "FREE"
            predictions_used = 0
            predictions_limit = 2

        welcome_text = f"""
🚀 Welcome to <b>Alpina Signal</b>

Professional AI-driven crypto market predictions powered by neural networks.

<b>Your Subscription:</b>
• Tier: {tier}
• Predictions: {predictions_used}/{predictions_limit}

<b>Features:</b>
✓ Real neural network models
✓ Probabilistic predictions (no fake certainty)
✓ 15 liquid crypto pairs
✓ Multiple timeframes (15m, 1h, 4h)
✓ Volatility-aware signals

Tap the button below to open the app.
"""

        # Create keyboard with Mini App button
        keyboard = [
            [InlineKeyboardButton(
                "📊 Open Alpina Signal",
                web_app=WebAppInfo(url=config.MINI_APP_URL)
            )]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await bot.send_message(
            chat_id=telegram_id,
            text=welcome_text,
            parse_mode="HTML",
            reply_markup=reply_markup
        )

        logger.info(f"Welcome message sent to user {telegram_id}")

    except Exception as e:
        logger.error(f"Error sending welcome message: {e}")


async def send_notification(telegram_id: int, message: str):
    """
    Send notification to user
    NO POLLING - Utility function for API to call
    """
    if not bot:
        logger.warning("Bot not initialized, cannot send notification")
        return

    try:
        await bot.send_message(
            chat_id=telegram_id,
            text=message,
            parse_mode="HTML"
        )
        logger.info(f"Notification sent to user {telegram_id}")
    except Exception as e:
        logger.error(f"Error sending notification: {e}")


async def send_signal_notification(telegram_id: int, signal_data: dict):
    """
    Send trading signal notification to user
    NO POLLING - Called from API when signal is generated
    """
    if not bot:
        logger.warning("Bot not initialized, cannot send signal")
        return

    try:
        signal_emoji = "📈" if signal_data["signal"] == "LONG" else "📉" if signal_data["signal"] == "SHORT" else "⚠️"

        price = signal_data.get("price", 0)
        # Format price: fewer decimals for large coins, more for small ones
        price_str = f"${price:,.2f}" if price > 1 else f"${price:.6f}"

        sl = signal_data.get("stop_loss", 0)
        tp = signal_data.get("take_profit", 0)
        sl_str = (f"${sl:,.2f}" if sl > 1 else f"${sl:.6f}") if sl else "—"
        tp_str = (f"${tp:,.2f}" if tp > 1 else f"${tp:.6f}") if tp else "—"

        reason = signal_data.get("reason", "")
        reason_line = f"\n<i>{reason[:120]}</i>" if reason else ""

        message = f"""
{signal_emoji} <b>{signal_data['signal']}</b> — {signal_data['symbol']} ({signal_data['timeframe']})

<b>Confidence:</b> {signal_data['confidence']:.1f}%
<b>Price:</b> {price_str}
<b>Stop Loss:</b> {sl_str}
<b>Take Profit:</b> {tp_str}{reason_line}
"""

        keyboard = [
            [InlineKeyboardButton(
                "📊 View in App",
                web_app=WebAppInfo(url=config.MINI_APP_URL)
            )]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await bot.send_message(
            chat_id=telegram_id,
            text=message,
            parse_mode="HTML",
            reply_markup=reply_markup
        )

        logger.info(f"Signal notification sent to user {telegram_id}")

    except Exception as e:
        logger.error(f"Error sending signal notification: {e}")


async def send_subscription_activated(telegram_id: int, tier: str):
    """
    Send subscription activation confirmation
    NO POLLING - Called from API after payment verification
    """
    if not bot:
        logger.warning("Bot not initialized, cannot send confirmation")
        return

    try:
        # Check if this tier has Telegram alerts
        tier_info = config.SUBSCRIPTION_TIERS.get(tier.lower(), {})
        has_alerts = tier_info.get("telegram_alerts", False)

        benefits = """<b>Benefits:</b>
✓ Unlimited predictions
✓ All timeframes
✓ Priority support"""

        if has_alerts:
            benefits += "\n✓ 🔔 <b>Telegram alerts</b> for signals >70%"

        message = f"""
🎉 <b>Subscription Activated!</b>

Your <b>{tier.upper()}</b> subscription is now active.

{benefits}

Thank you for choosing Alpina Signal!
"""

        keyboard = [
            [InlineKeyboardButton(
                "📊 Open App",
                web_app=WebAppInfo(url=config.MINI_APP_URL)
            )]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await bot.send_message(
            chat_id=telegram_id,
            text=message,
            parse_mode="HTML",
            reply_markup=reply_markup
        )

        logger.info(f"Subscription confirmation sent to user {telegram_id}")

    except Exception as e:
        logger.error(f"Error sending subscription confirmation: {e}")


# ========================
# NO MAIN FUNCTION - NO POLLING
# This module is ONLY for utility functions
# Mini App uses HTTP API, not bot polling
# ========================

if __name__ == "__main__":
    logger.error("This module should NOT be run directly!")
    logger.error("Bot polling is disabled for Mini App architecture")
    logger.error("Use main.py to start the API server instead")
    sys.exit(1)
