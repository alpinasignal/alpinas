"""
Telegram Bot for Alpina Signal
Handles user auth, payments, and launches Mini App
"""

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)
import sys
import os
from datetime import datetime
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


# ========================
# COMMAND HANDLERS
# ========================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start command - Welcome message and launch Mini App
    """
    user = update.effective_user
    chat_id = update.effective_chat.id

    # Register user
    sub_manager.create_user(
        telegram_id=user.id,
        username=user.username,
        first_name=user.first_name
    )

    # Get subscription info
    subscription = sub_manager.get_subscription(user.id)

    welcome_text = f"""
🚀 Welcome to <b>Alpina Signal</b>

Professional AI-driven crypto market predictions powered by neural networks.

<b>Your Subscription:</b>
• Tier: {subscription.tier.upper()}
• Predictions: {subscription.predictions_used}/{subscription.predictions_limit}

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
        )],
        [InlineKeyboardButton("💎 Upgrade Subscription", callback_data="upgrade")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="help")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        welcome_text,
        parse_mode="HTML",
        reply_markup=reply_markup
    )


async def subscription_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /subscription command - Show subscription details
    """
    user = update.effective_user
    subscription = sub_manager.get_subscription(user.id)

    if not subscription:
        await update.message.reply_text("No active subscription found.")
        return

    # Format expiry date
    expiry_text = "Never" if not subscription.expires_at else subscription.expires_at.strftime("%Y-%m-%d")

    # Get tier info
    tier_info = config.SUBSCRIPTION_TIERS.get(subscription.tier, {})

    text = f"""
📋 <b>Your Subscription</b>

<b>Tier:</b> {subscription.tier.upper()}
<b>Status:</b> {"Active" if subscription.is_active else "Inactive"}
<b>Predictions Used:</b> {subscription.predictions_used}/{subscription.predictions_limit}
<b>Expires:</b> {expiry_text}

<b>Selected Coins:</b>
{', '.join(subscription.selected_coins) if subscription.selected_coins else "None"}

<b>Tier Details:</b>
• Price: ${tier_info.get('price', 0)} USDT/month
• Access: {tier_info.get('coins', 'All')} coins
"""

    keyboard = [
        [InlineKeyboardButton("💎 Upgrade", callback_data="upgrade")],
        [InlineKeyboardButton("🔄 Manage Coins", callback_data="manage_coins")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        text,
        parse_mode="HTML",
        reply_markup=reply_markup
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /help command - Show help information
    """
    help_text = """
📚 <b>Alpina Signal - Help</b>

<b>What is Alpina Signal?</b>
A professional AI-driven crypto prediction system using real neural networks trained on historical market data.

<b>How it works:</b>
1. Neural networks analyze 3 years of market data
2. Models output probabilities (not fake certainty)
3. Volatility filters prevent risky trades
4. You get signals: LONG / SHORT / NO TRADE

<b>Signal Format:</b>
• <b>LONG</b> - AI predicts upward movement
• <b>SHORT</b> - AI predicts downward movement
• <b>NO TRADE</b> - Low confidence or high volatility

<b>Confidence:</b>
Percentage showing model's certainty (≥60% for signals)

<b>Timeframes:</b>
• 15m - Short-term scalping
• 1h - Intraday trading
• 4h - Swing trading

<b>Important:</b>
⚠️ This is NOT financial advice
⚠️ Past performance ≠ future results
⚠️ AI predictions are probabilistic, not guaranteed

<b>Commands:</b>
/start - Open Mini App
/subscription - View subscription
/help - Show this message

<b>Support:</b>
Contact @AlpinaSignalSupport
"""

    await update.message.reply_text(help_text, parse_mode="HTML")


# ========================
# CALLBACK HANDLERS
# ========================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks"""
    query = update.callback_query
    await query.answer()

    if query.data == "upgrade":
        await show_upgrade_options(query)
    elif query.data == "help":
        await help_callback(query)
    elif query.data == "manage_coins":
        await manage_coins_callback(query)
    elif query.data.startswith("tier_"):
        await handle_tier_selection(query)
    elif query.data == "cancel":
        await query.edit_message_text("Operation cancelled.")


async def show_upgrade_options(query):
    """Show subscription upgrade options"""
    text = """
💎 <b>Upgrade Subscription</b>

Choose your plan:

<b>BASIC</b> - $14.99 USDT/month
• 3 coins of your choice
• Unlimited predictions
• All timeframes

<b>PRO</b> - $24.99 USDT/month
• 7 coins of your choice
• Unlimited predictions
• All timeframes

<b>PREMIUM</b> - $49.99 USDT/month
• All 15 coins
• Unlimited predictions
• All timeframes
• Priority support

Select a plan below:
"""

    keyboard = [
        [InlineKeyboardButton("Basic - $14.99", callback_data="tier_basic")],
        [InlineKeyboardButton("Pro - $24.99", callback_data="tier_pro")],
        [InlineKeyboardButton("Premium - $49.99", callback_data="tier_premium")],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        text,
        parse_mode="HTML",
        reply_markup=reply_markup
    )


async def handle_tier_selection(query):
    """Handle tier selection"""
    tier = query.data.replace("tier_", "")
    tier_config = config.SUBSCRIPTION_TIERS.get(tier)

    if not tier_config:
        await query.edit_message_text("Invalid tier selected.")
        return

    # TODO: Implement payment flow
    # For now, just show payment instructions

    text = f"""
💳 <b>Payment Instructions</b>

<b>Plan:</b> {tier.upper()}
<b>Price:</b> ${tier_config['price']} USDT/month
<b>Coins:</b> {tier_config['coins']}

<b>To complete payment:</b>

1. Send <b>{tier_config['price']} USDT</b> (TRC20) to:
<code>TXYZabc123...</code>

2. After sending, click "I've Paid" below

3. We'll verify and activate your subscription

<b>Note:</b> Payment verification may take 1-5 minutes.
"""

    keyboard = [
        [InlineKeyboardButton("✅ I've Paid", callback_data=f"verify_payment_{tier}")],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        text,
        parse_mode="HTML",
        reply_markup=reply_markup
    )


async def manage_coins_callback(query):
    """Manage selected coins"""
    user_id = query.from_user.id
    subscription = sub_manager.get_subscription(user_id)

    if not subscription:
        await query.edit_message_text("No active subscription found.")
        return

    tier_config = config.SUBSCRIPTION_TIERS.get(subscription.tier)
    num_coins = tier_config.get("coins", 0)

    if subscription.tier == "free":
        num_coins = 2

    text = f"""
🪙 <b>Manage Your Coins</b>

<b>Current Selection:</b>
{', '.join(subscription.selected_coins) if subscription.selected_coins else "None"}

<b>Available:</b> {num_coins} coins

To change your selection, use the Mini App.
"""

    await query.edit_message_text(text, parse_mode="HTML")


async def help_callback(query):
    """Help button callback"""
    help_text = """
📚 <b>Quick Help</b>

<b>Commands:</b>
/start - Open Mini App
/subscription - View subscription
/help - Detailed help

<b>Signals:</b>
• LONG - Buy signal
• SHORT - Sell signal
• NO TRADE - Wait

Use the Mini App for predictions.
"""

    await query.edit_message_text(help_text, parse_mode="HTML")


# ========================
# ERROR HANDLER
# ========================

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")


# ========================
# MAIN
# ========================

def main():
    """Run the bot"""
    # Check token
    if not config.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set in environment")
        return

    # Create application
    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("subscription", subscription_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_callback))

    # Error handler
    application.add_error_handler(error_handler)

    # Start bot
    logger.info("Starting Alpina Signal bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    # Setup logging
    logger.add(
        os.path.join(config.LOGS_DIR, "bot.log"),
        rotation="50 MB",
        level="INFO"
    )

    main()
