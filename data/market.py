"""
Binance Market Data Fetcher
Retrieves historical OHLCV data without API keys
Professional data pipeline with proper error handling
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import time
from loguru import logger
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BinanceDataFetcher:
    """
    Fetches historical OHLCV data from Binance public API
    No authentication required
    """

    def __init__(self):
        self.base_url = config.BINANCE_BASE_URL
        self.session = requests.Session()

    def _timeframe_to_binance(self, timeframe: str) -> str:
        """Convert our timeframe format to Binance format"""
        mapping = {
            "15m": "15m",
            "1h": "1h",
            "4h": "4h"
        }
        return mapping.get(timeframe, "1h")

    def _get_klines(self, symbol: str, interval: str, start_time: int, end_time: int) -> List:
        """
        Fetch klines from Binance API
        Returns list of candles
        """
        endpoint = f"{self.base_url}/api/v3/klines"

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000  # max limit per request for Spot API
        }

        try:
            logger.debug(f"Requesting: {endpoint} params={params}")
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}, body: {e.response.text[:500]}")
            return []

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        days: int = config.HISTORICAL_DAYS
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a symbol

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            timeframe: Timeframe ("15m", "1h", "4h")
            days: Number of days to fetch

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        logger.info(f"Fetching {days} days of {timeframe} data for {symbol}")

        interval = self._timeframe_to_binance(timeframe)

        # Calculate start and end timestamps
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        all_klines = []
        current_start = start_time

        # Fetch data in chunks (Binance limit is 1500 candles per request)
        while current_start < end_time:
            klines = self._get_klines(symbol, interval, current_start, end_time)

            if not klines:
                break

            all_klines.extend(klines)

            # Update start time for next batch
            current_start = klines[-1][0] + 1

            # Rate limiting
            time.sleep(0.1)

            logger.debug(f"Fetched {len(klines)} candles, total: {len(all_klines)}")

        if not all_klines:
            logger.error(f"No data retrieved for {symbol}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        # Keep only relevant columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # Remove any NaN rows
        df = df.dropna()

        logger.info(f"Successfully fetched {len(df)} candles for {symbol} {timeframe}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def fetch_latest_candles(
        self,
        symbol: str,
        timeframe: str,
        num_candles: int = 500
    ) -> Optional[pd.DataFrame]:
        """
        Fetch latest N candles for real-time inference

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            num_candles: Number of recent candles to fetch

        Returns:
            DataFrame with latest candles
        """
        logger.info(f"Fetching latest {num_candles} candles for {symbol} {timeframe}")

        interval = self._timeframe_to_binance(timeframe)
        endpoint = f"{self.base_url}/api/v3/klines"

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(num_candles, 1000)
        }

        try:
            logger.info(f"Requesting: {endpoint} symbol={symbol} interval={interval} limit={params['limit']}")
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()

            if not klines:
                logger.error(f"No data retrieved for {symbol} (empty response)")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])

            # Keep only relevant columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna().reset_index(drop=True)

            logger.info(f"Successfully fetched {len(df)} latest candles for {symbol}")

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching latest candles for {symbol}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}, body: {e.response.text[:500]}")
            return None

    def save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save fetched data to local cache"""
        cache_dir = config.DATA_DIR
        os.makedirs(cache_dir, exist_ok=True)

        filename = f"{symbol}_{timeframe}.parquet"
        filepath = os.path.join(cache_dir, filename)

        df.to_parquet(filepath, index=False)
        logger.info(f"Saved data to cache: {filepath}")

    def load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from local cache if available"""
        cache_dir = config.DATA_DIR
        filename = f"{symbol}_{timeframe}.parquet"
        filepath = os.path.join(cache_dir, filename)

        if os.path.exists(filepath):
            # Check if cache is recent (less than 1 day old)
            cache_age = time.time() - os.path.getmtime(filepath)
            if cache_age < 86400:  # 24 hours
                logger.info(f"Loading data from cache: {filepath}")
                df = pd.read_parquet(filepath)
                return df

        return None

    def fetch_or_load(
        self,
        symbol: str,
        timeframe: str,
        days: int = config.HISTORICAL_DAYS,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from API or load from cache

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            days: Days of historical data
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        if use_cache:
            cached_df = self.load_from_cache(symbol, timeframe)
            if cached_df is not None:
                return cached_df

        # Fetch from API
        df = self.fetch_historical_data(symbol, timeframe, days)

        if df is not None:
            self.save_to_cache(df, symbol, timeframe)

        return df


def download_all_data():
    """
    Download historical data for all supported pairs and timeframes
    This should be run once during initial setup
    """
    fetcher = BinanceDataFetcher()

    total = len(config.SUPPORTED_PAIRS) * len(config.TIMEFRAMES)
    current = 0

    logger.info(f"Starting download of {total} datasets")

    for symbol in config.SUPPORTED_PAIRS:
        for timeframe in config.TIMEFRAMES:
            current += 1
            logger.info(f"[{current}/{total}] Downloading {symbol} {timeframe}")

            df = fetcher.fetch_historical_data(symbol, timeframe)

            if df is not None:
                fetcher.save_to_cache(df, symbol, timeframe)
            else:
                logger.error(f"Failed to download {symbol} {timeframe}")

            # Rate limiting between requests
            time.sleep(0.5)

    logger.info("Download complete!")


if __name__ == "__main__":
    # Setup logging
    from loguru import logger
    logger.add(
        os.path.join(config.LOGS_DIR, "data_download.log"),
        rotation="100 MB",
        level="INFO"
    )

    # Download all data
    download_all_data()
