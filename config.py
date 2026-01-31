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
SEQUENCE_LENGTH = 32  # 32 candles as context (optimized for CPU)

# Feature windows for rolling calculations
VOLATILITY_WINDOWS = [10, 20, 50]
MOMENTUM_WINDOWS = [5, 10, 20, 50]
EMA_PERIODS = [50, 200]
ATR_PERIOD = 14

# ========================
# NEURAL NETWORK
# ========================

# Model architecture - OPTIMIZED FOR CPU TRAINING
MODEL_TYPE = "transformer"  # "transformer" or "lstm"
HIDDEN_DIM = 64  # Compact model for fast CPU training
NUM_LAYERS = 2  # 2 layers - enough for pattern learning
NUM_HEADS = 4  # 4 attention heads (must divide HIDDEN_DIM)
DROPOUT = 0.2  # Moderate regularization
NUM_CLASSES = 3  # NO TRADE, LONG, SHORT

# Training configuration - Fast CPU training
BATCH_SIZE = 128  # Larger batches for faster training
LEARNING_RATE = 0.001  # Higher LR for faster convergence
MAX_EPOCHS = 30  # 30 epochs - enough with early stopping
EARLY_STOPPING_PATIENCE = 7  # Stop early if no improvement
VALIDATION_SPLIT = 0.15
LABEL_SMOOTHING = 0.1  # Label smoothing for better generalization

# Walk-forward validation
TRAIN_TEST_SPLIT = 0.80  # 80% train, 20% test

# Class labels (defined above in neural network section)

# Dynamic labeling thresholds (relative to ATR)
LABEL_THRESHOLD_MULTIPLIER = 0.5  # 0.5 * ATR

# Look-ahead for labeling (how many candles forward)
LABEL_LOOKAHEAD = 10

# ========================
# INFERENCE & SIGNAL GENERATION
# ========================

# Probability thresholds for signals - VERY STRICT for highest accuracy
LONG_THRESHOLD = 0.75  # Only ultra-confident signals (75%+)
SHORT_THRESHOLD = 0.75  # Only ultra-confident signals (75%+)

# Volatility filter (percentile)
MAX_VOLATILITY_PERCENTILE = 85  # Stricter filter - avoid volatile markets

# Minimum confidence to show (for UI)
MIN_DISPLAY_CONFIDENCE = 65  # Higher threshold for display

# ========================
# ADMIN CONFIGURATION
# ========================

# Admin Telegram IDs (unlimited access, can view stats)
ADMIN_IDS = [7940666073]  # Add your admin Telegram IDs here

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
        "coins": [],  # user can pick any 2
        "telegram_alerts": False
    },
    "basic": {
        "predictions": 3,
        "price": 14.99,
        "coins": 3,
        "telegram_alerts": False
    },
    "pro": {
        "predictions": 7,
        "price": 24.99,
        "coins": 7,
        "telegram_alerts": True  # ✓ Telegram alerts for signals >70%
    },
    "premium": {
        "predictions": 15,
        "price": 49.99,
        "coins": 15,
        "telegram_alerts": True  # ✓ Telegram alerts for signals >70%
    }
}

# Minimum confidence for Telegram alerts
TELEGRAM_ALERT_MIN_CONFIDENCE = 0.70  # Only send alerts for 70%+ signals

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
