# Changelog - Alpina Signal

## v2.0.0 - 2026-01-29 - Railway Production Fix

### ✅ CRITICAL ARCHITECTURAL CHANGES

**Removed Telegram Bot Polling Completely**

This system is now correctly implemented as a **Telegram Mini App (WebApp)**, not a polling bot.

### 🔧 Changes Made

#### 1. `main.py` - Production Entry Point
- ✅ Removed argparse CLI commands
- ✅ Removed `start_telegram_bot()` function
- ✅ Added missing `os` import
- ✅ Changed to production-ready mode (API only)
- ✅ Reads `PORT` from Railway environment
- ❌ NO bot polling

**Before:**
```python
python main.py api    # Required explicit command
python main.py bot    # Would start polling (WRONG)
```

**After:**
```python
python main.py        # Starts API only (CORRECT)
```

#### 2. `bot/bot.py` - Utility Module
- ✅ REMOVED `application.run_polling()` entirely
- ✅ Converted to utility module (message sending only)
- ✅ Added fallback handling for `None` subscription
- ✅ Added utility functions:
  - `send_welcome_message()`
  - `send_notification()`
  - `send_signal_notification()`
  - `send_subscription_activated()`
- ❌ NO bot loop
- ❌ NO getUpdates
- ❌ NO command handlers with polling

**Before:**
```python
# bot.py had application.run_polling()
application.run_polling(allowed_updates=Update.ALL_TYPES)  # WRONG
```

**After:**
```python
# bot.py is now utility-only
# Only async functions for sending messages
# NO polling whatsoever
```

#### 3. `railway.toml` - Deployment Config
- ✅ Created Railway configuration file
- ✅ Start command: `python main.py` (API only)
- ✅ Restart policy: `on_failure` (more stable)
- ✅ Health check enabled
- ❌ NO bot command in startup

#### 4. Documentation
- ✅ Created `DEPLOYMENT.md` - Comprehensive deployment guide
- ✅ Updated `README.md` - Clarified Mini App architecture
- ✅ Created this `CHANGELOG.md`

### 🐛 Bugs Fixed

1. **❌ Error: "Conflict: terminated by other getUpdates"**
   - **Cause:** Bot polling running when it shouldn't
   - **Fix:** Removed all polling code
   - **Status:** ✅ FIXED

2. **❌ Error: "NameError: name 'os' is not defined"**
   - **Cause:** Missing import in main.py
   - **Fix:** Added `import os`
   - **Status:** ✅ FIXED

3. **❌ Error: "'NoneType' object has no attribute 'tier'"**
   - **Cause:** No fallback when database unavailable
   - **Fix:** Added fallback handling in bot/bot.py
   - **Status:** ✅ FIXED

### 🏗️ Architecture

**Old (Wrong):**
```
Railway → python main.py bot & python main.py api
          ↓                    ↓
    Bot Polling          FastAPI Server
    (getUpdates)         (Correct)
    ❌ WRONG            ✅ Correct
```

**New (Correct):**
```
Railway → python main.py
          ↓
    FastAPI Server ONLY
    ✅ Correct

    No polling
    No getUpdates
    Mini App uses HTTP API
```

### 📊 Deployment Status

**Railway Environment Variables Required:**
```bash
TELEGRAM_BOT_TOKEN=your_token        # For sending messages only
DATABASE_URL=postgresql://...        # Supabase connection
MINI_APP_URL=https://your-app.com   # Mini App URL
PORT=8000                            # Auto-provided by Railway
```

**Railway Start Command:**
```bash
python main.py
```

### ✅ Verification Checklist

- [x] No `application.run_polling()` anywhere
- [x] No `start_polling()` anywhere
- [x] No `getUpdates` calls
- [x] main.py starts API only
- [x] bot/bot.py is utility-only
- [x] railway.toml configured correctly
- [x] Database fallback mode working
- [x] Admin features working
- [x] Documentation updated

### 🚀 How to Deploy

1. **Push to GitHub:**
   ```bash
   git push origin main
   ```

2. **Railway will automatically:**
   - Detect changes
   - Install dependencies
   - Run `python main.py`
   - Start ONLY the API server
   - No bot polling

3. **Verify:**
   - Check Railway logs: No "getUpdates" errors
   - Test API: `curl https://your-app.railway.app/`
   - Open Mini App in Telegram
   - Should work without conflicts

### 📝 Migration Notes

**If you have local development setup:**

**Before (OLD):**
```bash
python main.py api     # Start API
python main.py bot     # Start bot (DON'T DO THIS)
```

**After (NEW):**
```bash
python main.py         # Start API only
# Bot polling is REMOVED
```

**For CLI commands (training, predictions, etc.):**
Create separate scripts or use Python modules directly:
```bash
# Example: Train models
python -c "from ai.train import train_all_models; train_all_models()"
```

### 🎯 Key Takeaways

1. **This is a Telegram Mini App (WebApp)**
   - NOT a polling bot
   - Mini App = web interface inside Telegram
   - Communicates with backend via HTTP API

2. **Bot Token Usage**
   - Used ONLY for sending messages (optional)
   - NOT used for receiving updates
   - NOT used for command handling

3. **Deployment**
   - One command: `python main.py`
   - One server: FastAPI
   - No polling whatsoever

### 📚 Resources

- [Telegram Mini Apps Documentation](https://core.telegram.org/bots/webapps)
- [Railway Documentation](https://docs.railway.app/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## Previous Versions

### v1.0.0 - Initial Release (DEPRECATED)
- ❌ Had bot polling (incorrect architecture)
- ❌ Used argparse for CLI commands
- ❌ Missing error handling

**This version is deprecated. Do not use.**
