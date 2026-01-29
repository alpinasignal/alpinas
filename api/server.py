"""
REST API Server for Alpina Signal
Serves predictions to Telegram Mini App
All intelligence lives here - Mini App is just UI
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loguru import logger

# Import AI components
from ai.predict import load_model_and_predict
from ai.model_store import ModelStore
from data.market import BinanceDataFetcher

# Import subscription management
from payments.subscriptions import DatabaseManager, SubscriptionManager

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

# Initialize model store
model_store = ModelStore()

# Cache for loaded models (to avoid reloading)
loaded_models = {}

# Initialize database manager (lazy initialization)
db_manager = DatabaseManager()
subscription_manager = SubscriptionManager(db_manager)


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
    """Health check endpoint"""
    logger.info("Health check request received")
    return {
        "service": "Alpina Signal API",
        "status": "online",
        "version": config.API_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health():
    """Alternative health check endpoint"""
    return {"status": "ok"}


@app.get("/api/v1/supported-pairs")
async def get_supported_pairs():
    """Get list of supported trading pairs"""
    return {
        "pairs": config.SUPPORTED_PAIRS,
        "timeframes": config.TIMEFRAMES,
        "total": len(config.SUPPORTED_PAIRS)
    }


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

        return PredictionResponse(**prediction)

    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
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


@app.post("/api/v1/verify-payment")
async def verify_payment(
    user_id: int,
    plan: str,
    amount: float,
    wallet_address: str
):
    """
    Verify USDT TRC20 payment for subscription

    In production, this would:
    1. Query TRON blockchain for recent transactions to wallet_address
    2. Check if transaction amount matches plan price
    3. Verify transaction is recent (< 24 hours)
    4. Activate subscription in database

    For now, returns mock response for development
    """
    logger.info(f"Payment verification requested: user={user_id}, plan={plan}, amount=${amount}")

    # TODO: Implement actual blockchain verification using TronGrid API
    # Example: https://api.trongrid.io/v1/accounts/{address}/transactions

    # For development/demo: simulate payment verification
    # In production, this must verify actual blockchain transactions

    return {
        "success": True,
        "payment_verified": False,  # Set to True after actual verification
        "message": "Payment verification service will be implemented with TronGrid API",
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


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Alpina Signal API")


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
