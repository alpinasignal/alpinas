# Alpina Signal - AI Crypto Prediction System

Professional AI-driven crypto market prediction system for Telegram Mini App.

## Overview

Alpina Signal is a **serious quantitative ML system** that uses real neural networks to generate probabilistic forecasts for cryptocurrency markets. This is NOT a simple indicator bot - it's a production-grade AI system designed for a paid subscription product.

### Key Features

- **Real Neural Networks**: Transformer and LSTM architectures trained on 3 years of historical data
- **Probabilistic Predictions**: Outputs probabilities, not fake certainty
- **Professional Approach**: Walk-forward validation, no data leakage, proper ML engineering
- **15 Liquid Pairs**: BTC, ETH, SOL, BNB, XRP, ADA, AVAX, LINK, DOT, MATIC, LTC, OP, ARB, DOGE, TRX
- **Multiple Timeframes**: 15m, 1h, 4h
- **Telegram Mini App**: Clean UI with all intelligence on the backend
- **Subscription System**: Free and paid tiers with access control

### What This System Does

1. Fetches historical OHLCV data from Binance (no API key required)
2. Engineers 30+ quantitative features (not simple indicators)
3. Trains Transformer/LSTM models on historical data
4. Generates probabilistic predictions (UP/DOWN/FLAT probabilities)
5. Applies volatility filters and confidence thresholds
6. Outputs signals: LONG, SHORT, or NO TRADE
7. Serves predictions via REST API to Telegram Mini App

### What This System Does NOT Do

- ❌ Guarantee profits
- ❌ Use simple indicators pretending to be AI
- ❌ Make predictions without real ML models
- ❌ Run ML in the browser or Mini App

## Architecture

```
Binance Market Data (public API)
        ↓
Feature Engineering (30+ quant features)
        ↓
Neural Network (Transformer/LSTM)
        ↓
Probability Output [P(up), P(down), P(flat)]
        ↓
Volatility Filter + Confidence Threshold
        ↓
Signal Generation (LONG/SHORT/NO TRADE)
        ↓
REST API Server
        ↓
Telegram Mini App (UI only)
```

## Project Structure

```
crypto_ai_bot/
├── main.py                 # Main entry point
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── data/
│   ├── market.py          # Binance data fetcher
│   ├── features.py        # Feature engineering
│   └── datasets.py        # PyTorch datasets
├── ai/
│   ├── model.py           # Neural network models
│   ├── train.py           # Training pipeline
│   ├── predict.py         # Inference logic
│   ├── backtest.py        # Backtesting
│   └── model_store.py     # Model management
├── api/
│   └── server.py          # FastAPI REST API
├── bot/
│   └── bot.py             # Telegram bot
├── payments/
│   └── subscriptions.py   # Subscription management
├── webapp/
│   ├── index.html         # Mini App UI
│   ├── app.js            # Mini App logic
│   └── style.css         # Styling
├── models/                # Trained models
├── data_cache/           # Cached market data
└── logs/                 # Log files
```

## Installation

### Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for faster training

### Setup

1. **Clone the repository**
```bash
cd crypto_ai_bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set environment variables**

Create a `.env` file:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
MINI_APP_URL=https://your-domain.com/webapp
DATABASE_URL=sqlite+aiosqlite:///./alpina_signal.db
```

4. **Download historical data**
```bash
python main.py download
```

This will download 3 years of historical data for all 15 pairs and 3 timeframes (45 datasets total).

## Usage

### Training Models

**Train a single model:**
```bash
python main.py train-single --symbol BTCUSDT --timeframe 1h
```

**Train all models (45 total):**
```bash
python main.py full-train
```

This will train models for all 15 pairs × 3 timeframes. Takes several hours.

### Generate Predictions

```bash
python main.py predict --symbol BTCUSDT --timeframe 1h
```

Output:
```
BTCUSDT | 1H
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal: LONG
Confidence: 64%
Volatility: Normal
Model: Neural Network
Price: $43,250.50
Time: 2024-01-15 14:30:00 UTC

Probabilities:
  LONG: 64.2%
  SHORT: 22.1%
  NO TRADE: 13.7%
```

### Run Backtest

```bash
python main.py backtest --symbol BTCUSDT --timeframe 1h
```

Evaluates model performance on historical data with metrics:
- Win rate
- Average return per trade
- Sharpe ratio
- Max drawdown
- Per-signal-type performance

### Start API Server

```bash
python main.py api
```

Starts FastAPI server on `http://localhost:8000`

API endpoints:
- `POST /api/v1/predict` - Get prediction for symbol
- `POST /api/v1/predict-batch` - Get multiple predictions
- `GET /api/v1/market-status/{symbol}` - Get market status
- `GET /api/v1/subscription/{user_id}` - Get subscription info

### Start Telegram Bot

```bash
python main.py bot
```

Starts the Telegram bot that handles user authentication and launches the Mini App.

## Model Details

### Neural Network Architecture

**Transformer Predictor:**
- Input: Sequences of 128 candles with 30+ features
- Positional encoding for temporal information
- 4 Transformer encoder layers
- 8 attention heads
- Hidden dimension: 256
- Dropout: 0.2
- Output: 3-class softmax (NO TRADE, LONG, SHORT)

**LSTM Predictor (alternative):**
- Bidirectional LSTM
- 4 layers
- Hidden dimension: 256
- More stable for some datasets

### Features (30+)

**Returns & Momentum:**
- Log returns
- Momentum at 5/10/20/50 periods

**Volatility:**
- Rolling volatility at 10/20/50 periods
- ATR normalized
- ATR percentile (regime indicator)

**Price Structure:**
- Normalized candle body/wicks
- Close position within range
- Distance to EMA50/200
- Price z-score (compression)
- Bollinger bandwidth

**Volume:**
- Volume change
- Volume ratio to MA
- Volume-weighted direction

**Microstructure:**
- Trend strength (ADX-like)
- Volatility regime flags
- Range normalization

All features are:
- Normalized/standardized
- Computed without look-ahead bias
- Designed for stationarity

### Training Process

1. **Data Preparation**: 3 years of historical OHLCV
2. **Feature Engineering**: Create 30+ quantitative features
3. **Dynamic Labeling**: Labels based on future return relative to ATR
4. **Walk-Forward Split**: 80% train, 20% validation (time-ordered)
5. **Class Balancing**: Weighted loss function for imbalanced classes
6. **Training**: AdamW optimizer, early stopping, gradient clipping
7. **Validation**: Monitor accuracy and loss on validation set
8. **Model Save**: Save best model based on validation performance

### Inference & Signal Generation

1. **Fetch Latest Data**: Get 128+ recent candles
2. **Feature Computation**: Apply same feature engineering
3. **Neural Network**: Generate probability distribution
4. **Volatility Check**: Filter out high-volatility conditions
5. **Threshold Application**:
   - P(up) ≥ 60% → LONG
   - P(down) ≥ 60% → SHORT
   - Else → NO TRADE
6. **Output**: Signal + confidence + probabilities

## Subscription Tiers

| Tier | Price | Coins | Predictions |
|------|-------|-------|-------------|
| Free | $0 | 2 | 2 total |
| Basic | $14.99/mo | 3 | Unlimited |
| Pro | $24.99/mo | 7 | Unlimited |
| Premium | $49.99/mo | 15 | Unlimited |

## Telegram Mini App

The Mini App is **UI ONLY**. It:
- Displays predictions from the API
- Allows timeframe selection
- Shows confidence and probabilities
- Handles subscription tiers

The Mini App **NEVER**:
- Connects to Binance
- Runs ML models
- Calculates features
- Generates signals

All intelligence lives on the backend.

## Configuration

Edit `config.py` to customize:

```python
# Supported pairs
SUPPORTED_PAIRS = [...]

# Timeframes
TIMEFRAMES = ["15m", "1h", "4h"]

# Model architecture
MODEL_TYPE = "transformer"  # or "lstm"
HIDDEN_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 8

# Training
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Signal thresholds
LONG_THRESHOLD = 0.60
SHORT_THRESHOLD = 0.60
```

## Development Roadmap

- [x] Market data fetcher
- [x] Feature engineering pipeline
- [x] Neural network models (Transformer + LSTM)
- [x] Training pipeline with walk-forward validation
- [x] Inference and signal generation
- [x] Backtesting framework
- [x] REST API server
- [x] Subscription system
- [x] Telegram bot
- [x] Mini App UI
- [ ] Payment integration (USDT TRC20)
- [ ] Real-time prediction updates
- [ ] Model retraining pipeline
- [ ] Performance monitoring dashboard
- [ ] A/B testing framework

## Important Notes

### Disclaimer

⚠️ **This system is for educational and informational purposes only.**

- This is NOT financial advice
- Past performance does NOT guarantee future results
- Cryptocurrency trading involves substantial risk of loss
- AI predictions are probabilistic, not guaranteed
- Always do your own research
- Never invest more than you can afford to lose

### Professional Standards

This system follows professional ML engineering practices:

✓ Real neural networks trained on historical data
✓ Walk-forward validation (no future peeking)
✓ Proper feature engineering (not just indicators)
✓ Class imbalance handling
✓ Early stopping and regularization
✓ Probabilistic outputs (no fake certainty)
✓ Volatility-aware risk management
✓ Production-grade code architecture

### NOT Included

This system does NOT:
- Execute trades automatically
- Guarantee profits
- Use insider information
- Manipulate markets
- Provide financial advice

## Support

For issues and questions:
- GitHub Issues: [Link to your repo]
- Telegram: @AlpinaSignalSupport
- Email: support@alpinasignal.com

## License

[Your License Here]

## Credits

Built by a senior quantitative ML engineer for serious crypto traders.

No hype. No guaranteed profits. Professional quantitative approach.
