"""
REST API Server for Alpina Signal
Serves predictions to Telegram Mini App
All intelligence lives here - Mini App is just UI
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from collections import defaultdict
import sys
import os
from datetime import datetime, date
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger

# Import AI components
from ai.predict import load_model_and_predict
from ai.model_store import ModelStore
from data.market import BinanceDataFetcher

# Import subscription management
from payments.subscriptions import DatabaseManager, SubscriptionManager
from payments.tron_verify import TronPaymentVerifier

# Import bot functions for notifications (safe import - don't crash server if bot module fails)
try:
    from bot.bot import send_signal_notification
except Exception as _bot_import_err:
    logger.warning(f"Could not import bot module: {_bot_import_err}")
    async def send_signal_notification(*args, **kwargs):
        pass

# Import scanner (safe import)
try:
    from api.scanner import init_scanner, scan_all_signals
except Exception:
    try:
        from scanner import init_scanner, scan_all_signals
    except Exception as _scanner_err:
        logger.warning(f"Could not import scanner: {_scanner_err}")
        def init_scanner(*args, **kwargs): pass
        async def scan_all_signals(): pass

# APScheduler for background scanning
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    _scheduler = AsyncIOScheduler()
    _scheduler_available = True
except ImportError:
    logger.warning("apscheduler not installed — auto-scan disabled. Run: pip install apscheduler")
    _scheduler_available = False
    _scheduler = None

# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="AI-driven crypto prediction API for Alpina Signal"
)

# CORS middleware for Telegram Mini App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (Mini App UI)
webapp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webapp")
if os.path.exists(webapp_dir):
    app.mount("/webapp", StaticFiles(directory=webapp_dir), name="webapp")
    logger.info(f"Mounted webapp directory: {webapp_dir}")
else:
    logger.warning(f"Webapp directory not found: {webapp_dir}")

# Initialize model store
model_store = ModelStore()

# Cache for loaded models (to avoid reloading)
loaded_models = {}

# Initialize database manager (lazy initialization)
db_manager = DatabaseManager()
subscription_manager = SubscriptionManager(db_manager)

# Daily notification counter (resets on server restart or day change)
# Key: user_id, Value: {"date": "YYYY-MM-DD", "count": int}
_notification_counts = defaultdict(lambda: {"date": None, "count": 0})


def can_send_notification(user_id: int, daily_limit: int) -> bool:
    """Check if user can receive notification (within daily limit)"""
    today = date.today().isoformat()
    if _notification_counts[user_id]["date"] != today:
        _notification_counts[user_id] = {"date": today, "count": 0}
    if _notification_counts[user_id]["count"] >= daily_limit:
        return False
    _notification_counts[user_id]["count"] += 1
    return True


# ========================
# MODELS (Request/Response)
# ========================

class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str
    user_id: int


class PredictionResponse(BaseModel):
    symbol: str
    timeframe: str
    signal: str
    confidence: float
    probabilities: Dict[str, float]
    volatility: Dict[str, Any]
    price: float
    timestamp: str
    model: str
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    atr_value: Optional[float] = None


class BatchPredictionRequest(BaseModel):
    symbols: List[str]
    timeframe: str
    user_id: int


class UserSubscriptionInfo(BaseModel):
    user_id: int
    tier: str
    available_coins: List[str]
    predictions_used: int
    predictions_limit: int


# ========================
# AUTHENTICATION & AUTHORIZATION
# ========================

async def verify_user(user_id: int = Header(...)) -> int:
    """
    Verify user from Telegram
    In production, verify Telegram initData signature
    """
    # TODO: Implement proper Telegram Mini App authentication
    # For now, just accept any user_id
    return user_id


async def check_subscription(
    user_id: int,
    symbol: str
) -> bool:
    """
    Check if user has access to this symbol
    Based on subscription tier
    """
    # TODO: Implement database lookup
    # For now, always allow (will implement properly with database)
    return True


# ========================
# API ENDPOINTS
# ========================

@app.get("/")
async def root():
    """Root endpoint - Serve Mini App or API info"""
    # Check if request is from browser (for Mini App)
    # If webapp exists, serve it, otherwise return API info
    webapp_index = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webapp", "index.html")
    if os.path.exists(webapp_index):
        return FileResponse(webapp_index)
    else:
        # Fallback to API info if webapp not found
        logger.info("Health check request received (webapp not found)")
        return {
            "service": "Alpina Signal API",
            "status": "online",
            "version": config.API_VERSION,
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/health")
async def health():
    """Health check endpoint for Railway"""
    return {"status": "ok"}


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "service": "Alpina Signal API",
        "status": "online",
        "version": config.API_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "docs": "/docs"
    }


@app.get("/api/v1/supported-pairs")
async def get_supported_pairs():
    """Get list of supported trading pairs"""
    return {
        "pairs": config.SUPPORTED_PAIRS,
        "timeframes": config.TIMEFRAMES,
        "total": len(config.SUPPORTED_PAIRS)
    }


class RegisterUserRequest(BaseModel):
    telegram_id: int
    username: Optional[str] = None
    first_name: Optional[str] = None


@app.post("/api/v1/register-user")
async def register_user(request: RegisterUserRequest):
    """
    Register or update user when they open the Mini App.
    Called automatically on app load.
    """
    try:
        user = subscription_manager.create_user(
            telegram_id=request.telegram_id,
            username=request.username,
            first_name=request.first_name
        )

        if user:
            return {
                "success": True,
                "user_id": user.id,
                "telegram_id": user.telegram_id
            }
        else:
            return {
                "success": True,
                "message": "User registered (no DB)"
            }

    except Exception as e:
        logger.error(f"Error registering user: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def get_prediction(
    request: PredictionRequest,
    user_id: int = Depends(verify_user)
):
    """
    Get AI prediction for a symbol/timeframe

    This is where all the AI happens.
    Mini App just displays the result.
    """
    symbol = request.symbol.upper()
    timeframe = request.timeframe.lower()

    # Validate inputs
    if symbol not in config.SUPPORTED_PAIRS:
        raise HTTPException(
            status_code=400,
            detail=f"Symbol {symbol} not supported"
        )

    if timeframe not in config.TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Timeframe {timeframe} not supported"
        )

    # Check subscription
    has_access = await check_subscription(user_id, symbol)
    if not has_access:
        raise HTTPException(
            status_code=403,
            detail="Subscription required for this symbol"
        )

    # Get model path
    model_path = model_store.get_model_path(symbol, timeframe)

    if model_path is None or not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model not found for {symbol} {timeframe}"
        )

    # Generate prediction
    try:
        prediction = load_model_and_predict(symbol, timeframe, model_path)

        if prediction is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate prediction"
            )

        # Check if user should receive Telegram alerts
        # For Starter/Basic/Pro/Premium users with signals >70%
        try:
            signal_type = prediction.get("signal")
            confidence = prediction.get("confidence", 0)
            is_strong_signal = (
                confidence >= config.TELEGRAM_ALERT_MIN_CONFIDENCE and
                signal_type in ["LONG", "SHORT"]
            )

            # Send notification to user if eligible
            subscription = subscription_manager.get_subscription(user_id)
            if subscription and is_strong_signal:
                tier = subscription.tier.lower()
                tier_info = config.SUBSCRIPTION_TIERS.get(tier, {})
                has_alerts = tier_info.get("telegram_alerts", False)
                daily_limit = tier_info.get("daily_alerts", 0)

                if has_alerts and can_send_notification(request.user_id, daily_limit):
                    telegram_id = request.user_id
                    asyncio.create_task(
                        send_signal_notification(
                            telegram_id=telegram_id,
                            signal_data=prediction
                        )
                    )
                    logger.info(f"Telegram alert queued for user {telegram_id}: {signal_type} {confidence:.1f}%")

            # Always notify admin for strong signals (>70% LONG/SHORT)
            if is_strong_signal:
                for admin_id in config.ADMIN_IDS:
                    asyncio.create_task(
                        send_signal_notification(
                            telegram_id=admin_id,
                            signal_data=prediction
                        )
                    )
                logger.info(f"Admin alert queued: {symbol} {signal_type} {confidence:.1f}%")

        except Exception as e:
            # Don't fail the API if notification fails
            logger.warning(f"Failed to send Telegram alert: {e}")

        return PredictionResponse(**prediction)

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error generating prediction for {symbol} {timeframe}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/api/v1/predict-batch")
async def get_batch_predictions(
    request: BatchPredictionRequest,
    user_id: int = Depends(verify_user)
):
    """
    Get predictions for multiple symbols
    Used by Mini App dashboard view
    """
    timeframe = request.timeframe.lower()

    if timeframe not in config.TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Timeframe {timeframe} not supported"
        )

    predictions = []
    errors = []

    for symbol in request.symbols:
        symbol = symbol.upper()

        if symbol not in config.SUPPORTED_PAIRS:
            errors.append(f"{symbol}: not supported")
            continue

        # Check subscription
        has_access = await check_subscription(user_id, symbol)
        if not has_access:
            errors.append(f"{symbol}: subscription required")
            continue

        # Get model path
        model_path = model_store.get_model_path(symbol, timeframe)

        if model_path is None or not os.path.exists(model_path):
            errors.append(f"{symbol}: model not found")
            continue

        # Generate prediction
        try:
            prediction = load_model_and_predict(symbol, timeframe, model_path)
            if prediction:
                predictions.append(prediction)
            else:
                errors.append(f"{symbol}: prediction failed")
        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}")
            errors.append(f"{symbol}: {str(e)}")

    return {
        "predictions": predictions,
        "errors": errors,
        "total": len(predictions),
        "timeframe": timeframe
    }


@app.get("/api/v1/market-status/{symbol}")
async def get_market_status(symbol: str):
    """
    Get current market status for a symbol
    Price, volume, etc.
    """
    symbol = symbol.upper()

    if symbol not in config.SUPPORTED_PAIRS:
        raise HTTPException(
            status_code=400,
            detail=f"Symbol {symbol} not supported"
        )

    try:
        fetcher = BinanceDataFetcher()
        df = fetcher.fetch_latest_candles(symbol, "1h", num_candles=24)

        if df is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch market data"
            )

        current = df.iloc[-1]
        prev = df.iloc[-2]

        change_24h = ((current["close"] - df.iloc[0]["close"]) / df.iloc[0]["close"]) * 100
        change_1h = ((current["close"] - prev["close"]) / prev["close"]) * 100

        return {
            "symbol": symbol,
            "price": float(current["close"]),
            "volume_24h": float(df["volume"].sum()),
            "change_1h": round(change_1h, 2),
            "change_24h": round(change_24h, 2),
            "high_24h": float(df["high"].max()),
            "low_24h": float(df["low"].min()),
            "timestamp": current["timestamp"].isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching market status: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/api/v1/subscription/{user_id}")
async def get_subscription_info(user_id: int):
    """
    Get user subscription info
    """
    # TODO: Implement database lookup
    # For now, return mock data

    return {
        "user_id": user_id,
        "tier": "free",
        "predictions_used": 0,
        "predictions_limit": 2,
        "available_coins": config.SUPPORTED_PAIRS[:2],
        "expires_at": None
    }


@app.post("/api/v1/subscription/upgrade")
async def upgrade_subscription(user_id: int, tier: str):
    """
    Upgrade subscription tier
    Integrates with payment system
    """
    # TODO: Implement payment flow
    # TODO: Update database

    if tier not in config.SUBSCRIPTION_TIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier: {tier}"
        )

    return {
        "success": True,
        "message": "Subscription upgraded",
        "tier": tier,
        "payment_required": True
    }


@app.get("/api/v1/history/{user_id}")
async def get_prediction_history(user_id: int, limit: int = 50):
    """
    Get user's prediction history
    """
    # TODO: Implement database lookup
    # For now, return empty

    return {
        "user_id": user_id,
        "history": [],
        "total": 0
    }


# ========================
# STARTUP / SHUTDOWN
# ========================

@app.get("/api/v1/admin/stats")
async def get_admin_stats(
    x_telegram_user_id: Optional[int] = Header(None, alias="X-Telegram-User-ID")
):
    """
    Get admin statistics (admin only)

    Returns:
        User statistics, subscription counts, recent users
    """
    if x_telegram_user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Telegram User ID header required"
        )

    # Check if user is admin
    if not subscription_manager.is_admin(x_telegram_user_id):
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    try:
        stats = subscription_manager.get_user_stats(x_telegram_user_id)

        if stats is None:
            raise HTTPException(
                status_code=403,
                detail="Access denied"
            )

        return stats

    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        return {"error": str(e)}


class VerifyPaymentRequest(BaseModel):
    user_id: int
    plan: str
    amount: float
    wallet_address: str


@app.post("/api/v1/verify-payment")
async def verify_payment(request: VerifyPaymentRequest):
    """
    Verify USDT TRC20 payment for subscription

    Automatically checks TRON blockchain for payment and activates subscription
    """
    user_id = request.user_id
    plan = request.plan
    amount = request.amount
    logger.info(f"Payment verification requested: user={user_id}, plan={plan}, amount=${amount}")

    try:
        # Initialize payment verifier
        verifier = TronPaymentVerifier(config.PAYMENT_WALLET_ADDRESS)

        # Check for payment on blockchain
        payment_result = verifier.verify_payment(
            expected_amount=amount,
            timeframe_hours=config.PAYMENT_VERIFICATION_TIMEFRAME
        )

        if payment_result and payment_result.get("verified"):
            # Payment found! Activate subscription
            logger.info(f"✅ Payment verified for user {user_id}: {payment_result}")

            # TODO: Create/update user subscription in database
            # For now, just return success
            # In full implementation, call:
            # subscription_manager.upgrade_subscription(user_id, plan)

            return {
                "success": True,
                "payment_verified": True,
                "message": f"Payment verified! {payment_result['amount']:.2f} USDT received.",
                "transaction_hash": payment_result["transaction_hash"],
                "timestamp": payment_result["timestamp"],
                "user_id": user_id,
                "plan": plan,
                "amount": payment_result["amount"]
            }
        else:
            # Payment not found yet
            logger.info(f"❌ No matching payment found for user {user_id}")

            return {
                "success": True,
                "payment_verified": False,
                "message": "Payment not detected yet. Please wait 1-5 minutes after sending USDT.",
                "user_id": user_id,
                "plan": plan,
                "amount": amount
            }

    except Exception as e:
        logger.error(f"Error verifying payment: {e}")

        return {
            "success": False,
            "payment_verified": False,
            "message": f"Payment verification error: {str(e)}",
            "user_id": user_id,
            "plan": plan,
            "amount": amount
        }


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Alpina Signal API")
    logger.info(f"Supported pairs: {len(config.SUPPORTED_PAIRS)}")
    logger.info(f"Timeframes: {config.TIMEFRAMES}")

    # Check models
    latest_models = model_store.get_latest_models()
    logger.info(f"Available models: {len(latest_models)}")

    # Start auto signal scanner
    if _scheduler_available and _scheduler is not None:
        try:
            init_scanner(db_manager, subscription_manager, model_store, _notification_counts)
            interval = getattr(config, "SCANNER_INTERVAL_MINUTES", 15)
            _scheduler.add_job(
                scan_all_signals,
                "interval",
                minutes=interval,
                id="signal_scanner",
                replace_existing=True,
                max_instances=1,  # Don't overlap if scan takes longer than interval
            )
            _scheduler.start()
            logger.info(f"Auto signal scanner started — scanning every {interval} min")
        except Exception as e:
            logger.error(f"Failed to start scanner: {e}")
    else:
        logger.warning("APScheduler not available — auto signal scanning disabled")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Alpina Signal API")
    if _scheduler_available and _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scanner stopped")


# ========================
# RUN SERVER
# ========================

if __name__ == "__main__":
    import uvicorn

    # Setup logging
    logger.add(
        os.path.join(config.LOGS_DIR, "api.log"),
        rotation="50 MB",
        level="INFO"
    )

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )
