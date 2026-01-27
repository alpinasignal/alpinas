"""
Alpina Signal - Main Entry Point
AI-driven crypto prediction system for Telegram Mini App

Commands:
    python main.py download       # Download historical data
    python main.py train          # Train all models
    python main.py train-single   # Train single model
    python main.py predict        # Generate predictions
    python main.py backtest       # Run backtest
    python main.py api            # Start API server
    python main.py bot            # Start Telegram bot
    python main.py full-train     # Train models for all pairs/timeframes
"""

import sys
import argparse
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


def download_data():
    """Download historical market data for all pairs"""
    logger.info("Starting data download...")
    from data.market import download_all_data
    download_all_data()
    logger.info("Data download complete!")


def train_single_model(symbol: str, timeframe: str):
    """Train a single model"""
    from ai.train import train_model
    from ai.model_store import build_model_path

    logger.info(f"Training model for {symbol} {timeframe}")

    model_path = build_model_path(symbol, timeframe, config.MODEL_TYPE)

    model = train_model(
        symbol=symbol,
        timeframe=timeframe,
        model_type=config.MODEL_TYPE,
        save_path=model_path
    )

    if model:
        logger.info(f"Model trained and saved to {model_path}")
    else:
        logger.error("Training failed")


def train_all_models():
    """Train models for all supported pairs and timeframes"""
    from ai.train import train_model
    from ai.model_store import build_model_path, ModelStore

    model_store = ModelStore()

    total = len(config.SUPPORTED_PAIRS) * len(config.TIMEFRAMES)
    current = 0

    logger.info(f"Training {total} models...")

    for symbol in config.SUPPORTED_PAIRS:
        for timeframe in config.TIMEFRAMES:
            current += 1
            logger.info(f"[{current}/{total}] Training {symbol} {timeframe}")

            model_path = build_model_path(symbol, timeframe, config.MODEL_TYPE)

            try:
                model = train_model(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=config.MODEL_TYPE,
                    save_path=model_path
                )

                if model:
                    # Register model in store
                    model_store.register_model(
                        symbol=symbol,
                        timeframe=timeframe,
                        model_type=config.MODEL_TYPE,
                        model_path=model_path,
                        metrics={}  # TODO: Add actual metrics
                    )

                    logger.info(f"✓ {symbol} {timeframe} complete")
                else:
                    logger.error(f"✗ {symbol} {timeframe} failed")

            except Exception as e:
                logger.error(f"Error training {symbol} {timeframe}: {e}")
                continue

    logger.info("All models trained!")


def generate_prediction(symbol: str, timeframe: str):
    """Generate prediction for a single symbol"""
    from ai.predict import load_model_and_predict, format_signal_output
    from ai.model_store import build_model_path

    model_path = build_model_path(symbol, timeframe, config.MODEL_TYPE)

    logger.info(f"Generating prediction for {symbol} {timeframe}")

    prediction = load_model_and_predict(symbol, timeframe, model_path)

    if prediction:
        print(format_signal_output(prediction))
    else:
        logger.error("Failed to generate prediction")


def run_backtest(symbol: str, timeframe: str):
    """Run backtest for a model"""
    from ai.backtest import backtest_model
    from ai.model_store import build_model_path

    model_path = build_model_path(symbol, timeframe, config.MODEL_TYPE)

    logger.info(f"Running backtest for {symbol} {timeframe}")

    results = backtest_model(symbol, timeframe, model_path)

    logger.info("Backtest complete!")


def start_api_server():
    """Start API server"""
    import uvicorn
    from api.server import app

    logger.info("Starting API server...")

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )


def start_telegram_bot():
    """Start Telegram bot"""
    from bot.bot import main as bot_main

    logger.info("Starting Telegram bot...")
    bot_main()


def main():
    """Main entry point"""
    setup_logging()

    parser = argparse.ArgumentParser(description="Alpina Signal - AI Crypto Predictions")
    parser.add_argument("command", choices=[
        "download", "train", "train-single", "predict", "backtest",
        "api", "bot", "full-train"
    ], help="Command to execute")
    parser.add_argument("--symbol", type=str, help="Trading pair (e.g., BTCUSDT)")
    parser.add_argument("--timeframe", type=str, choices=config.TIMEFRAMES, help="Timeframe")

    args = parser.parse_args()

    if args.command == "download":
        download_data()

    elif args.command == "train-single":
        if not args.symbol or not args.timeframe:
            logger.error("Please specify --symbol and --timeframe")
            sys.exit(1)
        train_single_model(args.symbol.upper(), args.timeframe)

    elif args.command == "full-train" or args.command == "train":
        train_all_models()

    elif args.command == "predict":
        if not args.symbol or not args.timeframe:
            logger.error("Please specify --symbol and --timeframe")
            sys.exit(1)
        generate_prediction(args.symbol.upper(), args.timeframe)

    elif args.command == "backtest":
        if not args.symbol or not args.timeframe:
            logger.error("Please specify --symbol and --timeframe")
            sys.exit(1)
        run_backtest(args.symbol.upper(), args.timeframe)

    elif args.command == "api":
        start_api_server()

    elif args.command == "bot":
        start_telegram_bot()

    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
