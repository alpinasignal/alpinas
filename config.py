"""
Configuration for Alpina Signal AI Trading System
Professional quantitative ML approach - no shortcuts
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ========================
# MARKET CONFIGURATION
# ========================

SUPPORTED_PAIRS: List[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT",
    "LTCUSDT", "OPUSDT", "ARBUSDT", "DOGEUSDT", "TRXUSDT"
]

TIMEFRAMES: List[str] = ["15m", "1h", "4h"]

TIMEFRAME_TO_MINUTES: Dict[str, int] = {
    "15m": 15,
    "1h": 60,
    "4h": 240
}

# ========================
# DATA CONFIGURATION
# ========================

# Minimum 3 years of historical data
HISTORICAL_DAYS = 1095  # 3 years

# Binance API settings (public, no key needed)
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"

# ========================
# FEATURE ENGINEERING
# ========================

# Sequence length for neural network input
SEQUENCE_LENGTH = 48  # 48 candles for more context (was 32)

# Feature windows for rolling calculations
VOLATILITY_WINDOWS = [10, 20, 50]
MOMENTUM_WINDOWS = [5, 10, 20, 50]
EMA_PERIODS = [50, 200]
ATR_PERIOD = 14

# BTC correlation
BTC_CORRELATION_WINDOW = 20

# Market structure
REGRESSION_SLOPE_WINDOW = 20
HH_LL_LOOKBACK = 20

# Walk-forward validation
WALK_FORWARD_SPLITS = 5
MIN_TRAIN_PCT = 0.5

# ATR-based SL/TP
SL_ATR_MULTIPLIER = 1.5
TP_ATR_MULTIPLIER = 2.5

# ========================
# NEURAL NETWORK
# ========================

# Model architecture - IMPROVED for better accuracy
MODEL_TYPE = "transformer"  # "transformer" or "lstm"
HIDDEN_DIM = 128  # Increased from 64 for more capacity
NUM_LAYERS = 3  # Increased from 2 for deeper learning
NUM_HEADS = 8  # Increased from 4 for better attention
DROPOUT = 0.3  # Increased slightly to prevent overfitting
NUM_CLASSES = 3  # NO TRADE, LONG, SHORT

# Training configuration - IMPROVED
BATCH_SIZE = 256  # Larger batches for faster CPU training
LEARNING_RATE = 0.0005  # Lower LR for more stable training
MAX_EPOCHS = 25  # Enough epochs with early stopping
EARLY_STOPPING_PATIENCE = 7  # Balanced patience
MAX_TRAINING_CANDLES = 25000  # Cap data size for speed (most recent candles)
VALIDATION_SPLIT = 0.15
LABEL_SMOOTHING = 0.1

# Walk-forward validation
TRAIN_TEST_SPLIT = 0.80  # 80% train, 20% test

# Class labels (defined above in neural network section)

# Dynamic labeling thresholds (relative to ATR)
# Higher = fewer but higher quality signals (less noise in training data)
LABEL_THRESHOLD_MULTIPLIER = 1.0  # 1.0 * ATR — requires meaningful move to be labeled signal

# Look-ahead for labeling (how many candles forward)
LABEL_LOOKAHEAD = 12  # 12 candles: 3h for 15m, 12h for 1h, 48h for 4h

# ========================
# INFERENCE & SIGNAL GENERATION
# ========================

# Probability thresholds for signals - balanced for more signals
LONG_THRESHOLD = 0.55  # 55%+ confidence for LONG signal
SHORT_THRESHOLD = 0.55  # 55%+ confidence for SHORT signal

# Volatility filter (percentile)
MAX_VOLATILITY_PERCENTILE = 95  # Allow trading in most conditions

# Minimum confidence to show (for UI)
MIN_DISPLAY_CONFIDENCE = 50  # Show signals from 50%

# ========================
# ADMIN CONFIGURATION
# ========================

# Admin Telegram IDs (unlimited access, can view stats)
ADMIN_IDS = [7940666073, 5480212100, 1229980947]  # Add your admin Telegram IDs here

# ========================
# PAYMENT CONFIGURATION
# ========================

# USDT TRC20 receiving wallet address
PAYMENT_WALLET_ADDRESS = "TECGFKQd1SuJdVihGnegeVGfEXKnpCWieY"

# Payment verification timeframe (hours)
PAYMENT_VERIFICATION_TIMEFRAME = 24  # Check last 24 hours for payments

# ========================
# SUBSCRIPTION TIERS
# ========================

SUBSCRIPTION_TIERS = {
    "free": {
        "predictions": 2,
        "price": 0,
        "coins": 0,
        "telegram_alerts": False,
        "daily_alerts": 0
    },
    "basic": {
        "predictions": 999999,
        "price": 29.00,
        "coins": 7,  # 7 coins available
        "telegram_alerts": True,
        "daily_alerts": 999
    },
    "pro": {
        "predictions": 999999,
        "price": 39.00,
        "coins": 15,  # All 15 coins available
        "telegram_alerts": True,
        "daily_alerts": 999
    }
}

# Minimum confidence for Telegram alerts (as percentage, e.g., 70 means 70%)
TELEGRAM_ALERT_MIN_CONFIDENCE = 70  # Only send alerts for 70%+ signals

# ========================
# AUTO SIGNAL SCANNER
# ========================

# Minimum confidence for AUTO signals sent to all subscribers (higher threshold)
AUTO_SIGNAL_MIN_CONFIDENCE = 80  # Only auto-send for 80%+ signals

# How often to scan all pairs (in minutes)
SCANNER_INTERVAL_MINUTES = 15  # Scan every 15 minutes

# How long to wait before re-sending the same signal (hours)
SCANNER_RESEND_HOURS = 6  # Don't spam same signal within 6 hours

# ========================
# API CONFIGURATION
# ========================

API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Alpina Signal AI API"
API_VERSION = "1.0.0"

# ========================
# TELEGRAM BOT
# ========================

# Set via environment variable
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "")

# Mini App URL
MINI_APP_URL = os.getenv("MINI_APP_URL", "https://your-domain.com/webapp")

# ========================
# DATABASE
# ========================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./alpina_signal.db")

# ========================
# PATHS
# ========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data_cache")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ========================
# LOGGING
# ========================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"

# ========================
# SIGNAL OUTPUT FORMAT
# ========================

SIGNAL_CLASSES = {
    0: "NO TRADE",
    1: "LONG",
    2: "SHORT"
}

VOLATILITY_REGIMES = {
    "low": "Low",
    "normal": "Normal",
    "high": "High",
    "extreme": "Extreme"
}
