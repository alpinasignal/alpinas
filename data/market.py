"""
Market Data Fetcher
Retrieves historical OHLCV data without API keys
Supports multiple exchanges with automatic fallback:
Binance -> Bybit -> KuCoin
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
    Tries Binance -> Bybit -> KuCoin with automatic fallback
    No authentication required
    """

    # Class-level: track which exchange works (persists across instances)
    _working_exchange = None  # None means not tested yet

    BYBIT_INTERVAL_MAP = {
        "15m": "15",
        "1h": "60",
        "4h": "240"
    }

    KUCOIN_INTERVAL_MAP = {
        "15m": "15min",
        "1h": "1hour",
        "4h": "4hour"
    }

    def __init__(self):
        self.binance_url = "https://api.binance.com"
        self.bybit_url = "https://api.bybit.com"
        self.kucoin_url = "https://api.kucoin.com"
        self.session = requests.Session()

    def _timeframe_to_binance(self, timeframe: str) -> str:
        """Convert our timeframe format to Binance format"""
        mapping = {"15m": "15m", "1h": "1h", "4h": "4h"}
        return mapping.get(timeframe, "1h")

    def _symbol_to_kucoin(self, symbol: str) -> str:
        """Convert BTCUSDT to BTC-USDT for KuCoin"""
        # Remove USDT suffix and add dash
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}-USDT"
        return symbol

    # ==================== BINANCE ====================
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

        if response.status_code in (451, 403):
            raise requests.exceptions.HTTPError(
                f"Binance blocked: HTTP {response.status_code}", response=response
            )

        response.raise_for_status()
        return response.json()

    # ==================== BYBIT ====================
    def _fetch_bybit_klines(self, symbol: str, interval: str, limit: int,
                             start_time: int = None, end_time: int = None) -> List:
        """Fetch klines from Bybit API"""
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
        response = self.session.get(endpoint, params=params, timeout=30)

        if response.status_code in (451, 403):
            raise requests.exceptions.HTTPError(
                f"Bybit blocked: HTTP {response.status_code}", response=response
            )

        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0:
            raise Exception(f"Bybit API error: {data.get('retMsg')}")

        result_list = data.get("result", {}).get("list", [])
        if not result_list:
            return []

        # Convert Bybit format to Binance-compatible format
        # Bybit returns newest first - reverse for chronological order
        klines = []
        for candle in reversed(result_list):
            klines.append([
                int(candle[0]), candle[1], candle[2], candle[3], candle[4], candle[5],
                int(candle[0]), candle[6], 0, "0", "0", "0"
            ])
        return klines

    # ==================== KUCOIN ====================
    def _fetch_kucoin_klines(self, symbol: str, interval: str, limit: int,
                              start_time: int = None, end_time: int = None) -> List:
        """Fetch klines from KuCoin API"""
        kucoin_symbol = self._symbol_to_kucoin(symbol)
        kucoin_interval = self.KUCOIN_INTERVAL_MAP.get(interval, "1hour")

        params = {
            "symbol": kucoin_symbol,
            "type": kucoin_interval
        }
        # KuCoin uses seconds, not milliseconds
        if start_time:
            params["startAt"] = start_time // 1000
        if end_time:
            params["endAt"] = end_time // 1000

        endpoint = f"{self.kucoin_url}/api/v1/market/candles"
        response = self.session.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != "200000":
            raise Exception(f"KuCoin API error: {data.get('msg')}")

        result_list = data.get("data", [])
        if not result_list:
            return []

        # KuCoin format: [time, open, close, high, low, volume, amount]
        # Note: KuCoin returns newest first, time is in seconds
        # Convert to Binance format: [time_ms, open, high, low, close, volume, ...]
        klines = []
        for candle in reversed(result_list):  # Reverse for chronological order
            timestamp_ms = int(candle[0]) * 1000
            klines.append([
                timestamp_ms,       # open time (ms)
                candle[1],          # open
                candle[3],          # high (KuCoin index 3)
                candle[4],          # low (KuCoin index 4)
                candle[2],          # close (KuCoin index 2)
                candle[5],          # volume
                timestamp_ms,       # close time (approx)
                candle[6],          # amount as quote_volume
                0, "0", "0", "0"    # placeholders
            ])

        # Limit results
        if len(klines) > limit:
            klines = klines[-limit:]

        return klines

    # ==================== SMART FETCH WITH FALLBACK ====================
    def _fetch_klines_with_fallback(self, symbol: str, interval: str, limit: int,
                                     start_time: int = None, end_time: int = None) -> List:
        """
        Fetch klines with automatic exchange fallback
        Order: Binance -> Bybit -> KuCoin
        """
        exchanges = [
            ("binance", self._fetch_binance_klines),
            ("bybit", self._fetch_bybit_klines),
            ("kucoin", self._fetch_kucoin_klines),
        ]

        # If we already know which exchange works, try it first
        if BinanceDataFetcher._working_exchange:
            working = BinanceDataFetcher._working_exchange
            # Reorder to try working exchange first
            exchanges = sorted(exchanges, key=lambda x: 0 if x[0] == working else 1)

        last_error = None
        for exchange_name, fetch_func in exchanges:
            try:
                logger.info(f"Trying {exchange_name.upper()} for {symbol} {interval}...")
                klines = fetch_func(symbol, interval, limit, start_time, end_time)
                if klines:
                    # Remember this exchange works
                    BinanceDataFetcher._working_exchange = exchange_name
                    logger.info(f"Successfully fetched {len(klines)} candles from {exchange_name.upper()}")
                    return klines
            except Exception as e:
                logger.warning(f"{exchange_name.upper()} failed: {e}")
                last_error = e
                continue

        logger.error(f"All exchanges failed for {symbol}. Last error: {last_error}")
        return []

    def _get_klines(self, symbol: str, interval: str, start_time: int, end_time: int) -> List:
        """Fetch klines for historical data (with start/end time)"""
        return self._fetch_klines_with_fallback(symbol, interval, 1000, start_time, end_time)

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
            current_start = int(klines[-1][0]) + 1
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

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df.dropna()

        logger.info(f"Successfully fetched {len(df)} candles for {symbol} {timeframe}")
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

        try:
            klines = self._fetch_klines_with_fallback(symbol, interval, num_candles)

            if not klines:
                logger.error(f"No data retrieved for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])

            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna().reset_index(drop=True)

            exchange = BinanceDataFetcher._working_exchange or "unknown"
            logger.info(f"Successfully fetched {len(df)} candles for {symbol} via {exchange.upper()}")

            return df

        except Exception as e:
            logger.error(f"Error fetching latest candles for {symbol}: {e}")
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
            cache_age = time.time() - os.path.getmtime(filepath)
            if cache_age < 86400:  # 24 hours
                logger.info(f"Loading data from cache: {filepath}")
                return pd.read_parquet(filepath)
        return None

    def fetch_or_load(
        self,
        symbol: str,
        timeframe: str,
        days: int = config.HISTORICAL_DAYS,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """Fetch data from API or load from cache"""
        if use_cache:
            cached_df = self.load_from_cache(symbol, timeframe)
            if cached_df is not None:
                return cached_df

        df = self.fetch_historical_data(symbol, timeframe, days)
        if df is not None:
            self.save_to_cache(df, symbol, timeframe)
        return df


def download_all_data():
    """Download historical data for all supported pairs and timeframes"""
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

            time.sleep(0.5)

    logger.info("Download complete!")


if __name__ == "__main__":
    from loguru import logger
    logger.add(
        os.path.join(config.LOGS_DIR, "data_download.log"),
        rotation="100 MB",
        level="INFO"
    )
    download_all_data()
