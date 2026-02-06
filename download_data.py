"""
Download Historical Data for Training
Uses ccxt library - works with VPN or from non-US location
"""

import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# Configuration
PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT",
    "LTC/USDT", "OP/USDT", "ARB/USDT", "DOGE/USDT", "TRX/USDT"
]

TIMEFRAMES = ["15m", "1h", "4h"]
DAYS = 1095  # 3 years

# Create data cache directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
os.makedirs(DATA_DIR, exist_ok=True)


def download_ohlcv(exchange, symbol, timeframe, days=365):
    """Download OHLCV data from exchange"""

    # Calculate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    since = int(start_time.timestamp() * 1000)

    all_data = []

    print(f"Downloading {symbol} {timeframe}...")

    while since < int(end_time.timestamp() * 1000):
        try:
            # Fetch 1000 candles at a time
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)

            if not ohlcv:
                break

            all_data.extend(ohlcv)

            # Update since to last candle time + 1
            since = ohlcv[-1][0] + 1

            # Progress
            print(f"  Downloaded {len(all_data)} candles...")

            # Rate limit
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)
            continue

    if not all_data:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def main():
    """Download all data"""

    print("=" * 60)
    print("DOWNLOADING HISTORICAL DATA FOR TRAINING")
    print("=" * 60)

    # Try exchanges in order
    exchanges_to_try = [
        ('binance', ccxt.binance()),
        ('bybit', ccxt.bybit()),
        ('kucoin', ccxt.kucoin()),
        ('okx', ccxt.okx()),
    ]

    exchange = None
    exchange_name = None

    for name, ex in exchanges_to_try:
        try:
            print(f"Trying {name}...")
            ex.load_markets()
            exchange = ex
            exchange_name = name
            print(f"Using {name}")
            break
        except Exception as e:
            print(f"  {name} failed: {e}")
            continue

    if not exchange:
        print("ERROR: No exchange available. Try using VPN.")
        return

    # Download data for each pair and timeframe
    total = len(PAIRS) * len(TIMEFRAMES)
    completed = 0

    for pair in PAIRS:
        for timeframe in TIMEFRAMES:
            completed += 1
            print(f"\n[{completed}/{total}] {pair} {timeframe}")

            try:
                df = download_ohlcv(exchange, pair, timeframe, days=DAYS)

                if df is not None and len(df) > 0:
                    # Save to CSV
                    symbol = pair.replace("/", "")
                    filename = f"{symbol}_{timeframe}.csv"
                    filepath = os.path.join(DATA_DIR, filename)

                    df.to_csv(filepath, index=False)
                    print(f"  Saved {len(df)} candles to {filename}")
                else:
                    print(f"  No data received")

            except Exception as e:
                print(f"  Error downloading {pair} {timeframe}: {e}")

            # Rate limit between pairs
            time.sleep(1)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print(f"Data saved to: {DATA_DIR}")
    print("=" * 60)

    # List downloaded files
    files = os.listdir(DATA_DIR)
    print(f"\nDownloaded {len(files)} files:")
    for f in sorted(files)[:10]:
        filepath = os.path.join(DATA_DIR, f)
        size = os.path.getsize(filepath) / 1024 / 1024
        print(f"  {f} ({size:.1f} MB)")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")


if __name__ == "__main__":
    main()
