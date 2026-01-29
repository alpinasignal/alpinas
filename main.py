"""
Alpina Signal - Production Entry Point
AI-driven crypto prediction system for Telegram Mini App

For Railway/Production: Automatically starts API server (no bot polling)
For Development CLI commands, use separate scripts in /scripts/
"""

import os
from loguru import logger
import config


def setup_logging():
    """Setup logging configuration"""
    logger.add(
        "logs/alpina_signal.log",
        rotation="100 MB",
        level="INFO",
        format=config.LOG_FORMAT
    )


def start_api_server():
    """Start FastAPI server (Production Mode - NO POLLING)"""
    import uvicorn
    from api.server import app

    # Get port from environment (Railway provides PORT env var)
    port = int(os.environ.get("API_PORT", os.environ.get("PORT", 8000)))
    host = os.environ.get("API_HOST", "0.0.0.0")

    logger.info(f"Starting Alpina Signal API on {host}:{port}")
    logger.info(f"Environment: {'Production' if port != 8000 else 'Development'}")
    logger.info(f"Mini App URL: {config.MINI_APP_URL}")
    logger.info("Mode: Telegram Mini App (NO BOT POLLING)")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


def main():
    """Main entry point - Production mode (API only, NO POLLING)"""
    setup_logging()

    logger.info("=" * 60)
    logger.info("Alpina Signal - AI Crypto Predictions")
    logger.info("Architecture: Telegram Mini App (WebApp)")
    logger.info("Mode: API Server ONLY (No bot polling)")
    logger.info("=" * 60)

    # Start API server
    # NO bot polling - Mini App communicates via HTTP API
    start_api_server()


if __name__ == "__main__":
    main()
