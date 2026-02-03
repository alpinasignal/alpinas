"""
Market Data Fetcher
Retrieves historical OHLCV data without API keys
Supports Binance with automatic Bybit fallback for US servers
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
    Fetches historical OHLCV data from exchange APIs
    Tries Binance first, auto-falls back to Bybit if blocked (HTTP 451)
    No authentication required
    """

    # Class-level flag: once Binance is detected as blocked, use Bybit for all instances
    _use_bybit = False

    BYBIT_INTERVAL_MAP = {
        "15m": "15",
        "1h": "60",
        "4h": "240"
    }

    def __init__(self):
        self.binance_url = config.BINANCE_BASE_URL  # https://api.binance.com
        self.bybit_url = "https://api.bybit.com"
        self.session = requests.Session()

    def _timeframe_to_binance(self, timeframe: str) -> str:
        """Convert our timeframe format to Binance format"""
        mapping = {
            "15m": "15m",
            "1h": "1h",
            "4h": "4h"
        }
        return mapping.get(timeframe, "1h")

    def _fetch_binance_klines(self, symbol: str, interval: str, limit: int,
                               start_time: int = None, end_time: int = None) -> List:
        """Fetch klines from Binance API"""
        endpoint = f"{self.binance_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        response = self.session.get(endpoint, params=params, timeout=30)

        # Check if blocked (US servers)
        if response.status_code in (451, 403):
            logger.warning(f"Binance blocked (HTTP {response.status_code}), switching to Bybit")
            BinanceDataFetcher._use_bybit = True
            raise requests.exceptions.HTTPError(
                f"Binance blocked: HTTP {response.status_code}", response=response
            )

        response.raise_for_status()
        return response.json()

    def _fetch_bybit_klines(self, symbol: str, interval: str, limit: int,
                             start_time: int = None, end_time: int = None) -> List:
        """
        Fetch klines from Bybit API as fallback
        Converts Bybit response format to Binance-compatible format
        """
        bybit_interval = self.BYBIT_INTERVAL_MAP.get(interval, "60")

        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": min(limit, 1000)
        }
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        endpoint = f"{self.bybit_url}/v5/market/kline"
        logger.debug(f"Bybit request: {endpoint} params={params}")

        response = self.session.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0:
            logger.error(f"Bybit API error: {data.get('retMsg')}")
            return []

        result_list = data.get("result", {}).get("list", [])
        if not result_list:
            return []

        # Convert Bybit format to Binance-compatible format
        # Bybit returns newest first - reverse to get chronological order
        # Bybit candle: [startTime, open, high, low, close, volume, turnover]
        klines = []
        for candle in reversed(result_list):
            klines.append([
                int(candle[0]),   # open time (ms)
                candle[1],        # open
                candle[2],        # high
                candle[3],        # low
                candle[4],        # close
                candle[5],        # volume
                int(candle[0]),   # close time (approx)
                candle[6],        # turnover as quote_volume
                0,                # number of trades (N/A)
                "0",              # taker buy base (N/A)
                "0",              # taker buy quote (N/A)
                "0"               # ignore
            ])

        return klines

    def _get_klines(self, symbol: str, interval: str, start_time: int, end_time: int) -> List:
        """
        Fetch klines with automatic fallback
        Tries Binance first, falls back to Bybit if blocked
        """
        # If already known to be blocked, go straight to Bybit
        if BinanceDataFetcher._use_bybit:
            try:
                return self._fetch_bybit_klines(symbol, interval, 1000, start_time, end_time)
            except Exception as e:
                logger.error(f"Bybit klines error for {symbol}: {e}")
                return []

        # Try Binance first
        try:
            return self._fetch_binance_klines(symbol, interval, 1000, start_time, end_time)
        except requests.exceptions.HTTPError as e:
            if BinanceDataFetcher._use_bybit:
                # Binance was just blocked, try Bybit
                try:
                    return self._fetch_bybit_klines(symbol, interval, 1000, start_time, end_time)
                except Exception as bybit_err:
                    logger.error(f"Bybit fallback also failed for {symbol}: {bybit_err}")
                    return []
            logger.error(f"Binance klines error for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
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

        # Fetch data in chunks
        while current_start < end_time:
            klines = self._get_klines(symbol, interval, current_start, end_time)

            if not klines:
                break

            all_klines.extend(klines)

            # Update start time for next batch
            current_start = int(klines[-1][0]) + 1

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
        exchange = "Bybit" if BinanceDataFetcher._use_bybit else "Binance"

        try:
            # Try appropriate exchange
            if BinanceDataFetcher._use_bybit:
                klines = self._fetch_bybit_klines(symbol, interval, num_candles)
                logger.info(f"Using Bybit for {symbol} {interval} (limit={num_candles})")
            else:
                try:
                    klines = self._fetch_binance_klines(symbol, interval, num_candles)
                    logger.info(f"Using Binance for {symbol} {interval}")
                except requests.exceptions.HTTPError:
                    if BinanceDataFetcher._use_bybit:
                        # Just got blocked, retry with Bybit
                        klines = self._fetch_bybit_klines(symbol, interval, num_candles)
                        logger.info(f"Switched to Bybit for {symbol} {interval}")
                    else:
                        raise

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

            logger.info(f"Successfully fetched {len(df)} latest candles for {symbol} via {exchange}")

            return df

        except Exception as e:
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
