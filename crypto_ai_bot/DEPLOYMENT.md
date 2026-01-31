# Alpina Signal - Deployment Guide

## Architecture Overview

**Alpina Signal is a Telegram Mini App (WebApp), NOT a polling bot.**

### What this means:
- ✅ FastAPI server runs 24/7
- ✅ Mini App communicates via HTTP API
- ✅ Bot token used ONLY for sending messages
- ❌ NO bot polling (no getUpdates)
- ❌ NO bot command handlers
- ❌ NO long-running bot loop

### Components:
1. **FastAPI API Server** (`api/server.py`)
   - Serves predictions to Mini App
   - Handles subscriptions
   - Admin dashboard API

2. **Telegram Mini App UI** (`webapp/`)
   - HTML/JS interface
   - Opens inside Telegram
   - Communicates with API

3. **Bot Utilities** (`bot/bot.py`)
   - Send notifications (optional)
   - NO polling loop
   - Utility functions only

4. **AI Models** (`ai/`)
   - Neural network predictions
   - Trained on 3 years of data

## Deployment (Railway)

### Environment Variables:
```bash
# Required
TELEGRAM_BOT_TOKEN=your_bot_token_here
DATABASE_URL=postgresql://...
MINI_APP_URL=https://your-domain.com/webapp

# Optional (Railway provides PORT automatically)
PORT=8000
API_HOST=0.0.0.0
```

### Railway Configuration:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python main.py",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

### Start Command:
```bash
python main.py
```

This starts ONLY the FastAPI server (no bot polling).

## Development

### Local Testing:
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python main.py

# Access at http://localhost:8000
```

### Testing Mini App:
1. Deploy to Railway
2. Set MINI_APP_URL in BotFather
3. Open Mini App in Telegram
4. Test API endpoints

## Common Issues

### ❌ Error: "Conflict: terminated by other getUpdates"
**Solution:** This is now FIXED. No polling code exists.

### ❌ Error: "NameError: name 'os' is not defined"
**Solution:** This is now FIXED. `os` imported in main.py.

### ❌ Error: "'NoneType' object has no attribute 'tier'"
**Solution:** This is now FIXED. Fallback handling added for database failures.

## Architecture Benefits

1. **No Conflicts:**
   - No getUpdates conflicts
   - Multiple instances safe
   - No bot polling overhead

2. **Scalability:**
   - API can handle many concurrent users
   - Mini App loads fast
   - Stateless architecture

3. **Reliability:**
   - Database failures don't crash app (fallback mode)
   - API restarts automatically
   - Health checks at `/`

## Production Checklist

- [x] Remove bot polling
- [x] Fix database fallback mode
- [x] Add admin authentication
- [x] Environment variables configured
- [x] MINI_APP_URL set in BotFather
- [x] Railway deployment configured
- [ ] SSL/HTTPS enabled
- [ ] Domain configured
- [ ] Payment verification implemented

## API Endpoints

### Public:
- `GET /` - Health check
- `GET /api/v1/supported-pairs` - List trading pairs
- `POST /api/v1/predict` - Get AI prediction
- `GET /api/v1/market-status/{symbol}` - Current price

### Admin (requires X-Telegram-User-ID header):
- `GET /api/v1/admin/stats` - User statistics (Admin only)

### Subscriptions:
- `GET /api/v1/subscription/{user_id}` - Get subscription
- `POST /api/v1/subscription/upgrade` - Upgrade tier
- `POST /api/v1/verify-payment` - Verify USDT payment

## Bot Token Usage

The bot token is used ONLY for:
1. Sending notifications (optional)
2. Verifying Telegram initData (future)

It is NOT used for:
- ❌ Polling updates
- ❌ Receiving messages
- ❌ Command handling

## Support

For issues:
1. Check Railway logs: `railway logs`
2. Check API health: `curl https://your-domain.com/`
3. Contact: @AlpinaSignalSupport

## Version History

- **v2.0** - Removed bot polling, Mini App architecture
- **v1.0** - Initial release with polling (deprecated)
