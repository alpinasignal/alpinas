# Alpina Signal - Quick Start Guide

Get your AI crypto prediction system running in 30 minutes.

## Step 1: Setup Environment (5 minutes)

```bash
# Navigate to project directory
cd crypto_ai_bot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment Variables (2 minutes)

```bash
# Copy example env file
cp .env.example .env

# Edit .env file with your values
# Minimum required: TELEGRAM_BOT_TOKEN
```

Get your Telegram Bot Token:
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` and follow instructions
3. Copy the token to `.env`

## Step 3: Download Market Data (10 minutes)

```bash
python main.py download
```

This downloads 3 years of historical data for all 15 pairs.

**What's happening:**
- Fetching OHLCV data from Binance public API
- No API key needed
- Data cached locally for training

## Step 4: Train Your First Model (10 minutes)

Start with a single model to test:

```bash
python main.py train-single --symbol BTCUSDT --timeframe 1h
```

**What's happening:**
- Feature engineering (30+ quant features)
- Neural network training (Transformer)
- Walk-forward validation
- Model saved to `models/` directory

Training time: ~5-10 minutes on CPU, ~2 minutes on GPU

## Step 5: Generate Prediction (1 minute)

```bash
python main.py predict --symbol BTCUSDT --timeframe 1h
```

**Output:**
```
BTCUSDT | 1H
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal: LONG
Confidence: 64%
Volatility: Normal
Model: Neural Network
Price: $43,250.50

Probabilities:
  LONG: 64.2%
  SHORT: 22.1%
  NO TRADE: 13.7%
```

## Step 6: Start API Server (1 minute)

In a new terminal:

```bash
python main.py api
```

API will run on `http://localhost:8000`

Test it: Open `http://localhost:8000/docs` in your browser

## Step 7: Start Telegram Bot (1 minute)

In another terminal:

```bash
python main.py bot
```

Now message your bot on Telegram!

## Next Steps

### Train All Models

For production, train all 45 models:

```bash
python main.py full-train
```

This trains models for:
- 15 trading pairs
- 3 timeframes each
- Total: 45 models

Time: 4-8 hours on CPU, 1-2 hours on GPU

**Pro tip:** Train overnight or use a GPU-enabled server

### Run Backtest

Evaluate model performance:

```bash
python main.py backtest --symbol BTCUSDT --timeframe 1h
```

Get metrics:
- Win rate
- Average return per trade
- Sharpe ratio
- Max drawdown

### Deploy Mini App

1. **Host the Mini App:**
   - Upload `webapp/` files to your web server
   - Update `MINI_APP_URL` in `.env`

2. **Configure Bot:**
   - Set Mini App URL in BotFather
   - Send `/mybots` → Select your bot → Bot Settings → Menu Button
   - Set URL to your Mini App

3. **Deploy API:**
   - Use a cloud service (AWS, DigitalOcean, etc.)
   - Ensure API is accessible to Mini App
   - Update API URL in `webapp/app.js`

## Common Issues

### "No module named 'torch'"

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### "TELEGRAM_BOT_TOKEN not set"

Make sure `.env` file exists and contains:
```
TELEGRAM_BOT_TOKEN=your_actual_token_here
```

### "Model not found"

Train the model first:
```bash
python main.py train-single --symbol BTCUSDT --timeframe 1h
```

### Training too slow

- Use GPU if available
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `SEQUENCE_LENGTH` to 64
- Use LSTM instead of Transformer (faster)

## Production Checklist

Before going live:

- [ ] Train all 45 models
- [ ] Run backtests for key pairs
- [ ] Test API endpoints
- [ ] Test Telegram bot
- [ ] Deploy Mini App to production server
- [ ] Set up proper database (PostgreSQL)
- [ ] Configure HTTPS for API
- [ ] Set up monitoring/logging
- [ ] Test payment flow
- [ ] Add rate limiting
- [ ] Set up model retraining schedule

## Performance Tips

### Speed up training:

1. **Use GPU:**
   - Install PyTorch with CUDA
   - Training 10-20x faster

2. **Parallel training:**
   - Train multiple models simultaneously
   - Use separate Python processes

3. **Reduce data:**
   - Use 1 year instead of 3 years for testing
   - Reduce sequence length to 64

### Optimize inference:

1. **Model caching:**
   - Load models once, reuse
   - Implemented in API server

2. **Batch predictions:**
   - Use `/predict-batch` endpoint
   - Generate multiple predictions at once

3. **Data caching:**
   - Cache market data
   - Update periodically instead of every request

## Getting Help

- **Documentation:** See `README.md`
- **Issues:** GitHub Issues
- **Community:** Telegram @AlpinaSignalSupport

## What's Next?

Now that you have the system running:

1. **Test predictions** - Compare with actual market movements
2. **Tune hyperparameters** - Experiment with model settings
3. **Add more features** - Enhance feature engineering
4. **Backtest extensively** - Validate model performance
5. **Deploy to production** - Launch your Mini App

Remember: This is a serious ML system. Take time to understand it, test it, and validate it before going live with a paid product.

No shortcuts. No fake AI. Stability and correctness first.

Happy trading! 🚀
