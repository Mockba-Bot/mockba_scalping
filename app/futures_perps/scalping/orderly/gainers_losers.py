import os
import time
import asyncio
import logging
from typing import List, Tuple, Optional
import urllib.parse
import json

import httpx
import pandas as pd
from dotenv import load_dotenv
from base58 import b58decode
from base64 import urlsafe_b64encode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

load_dotenv()

# ✅ Orderly API Config (from your setup)
BASE_URL = os.getenv("ORDERLY_BASE_URL", "https://api.orderly.org").rstrip()
ORDERLY_ACCOUNT_ID = os.getenv("ORDERLY_ACCOUNT_ID")
ORDERLY_SECRET = os.getenv("ORDERLY_SECRET")
ORDERLY_PUBLIC_KEY = os.getenv("ORDERLY_PUBLIC_KEY")

if not ORDERLY_SECRET or not ORDERLY_PUBLIC_KEY:
    raise ValueError("❌ ORDERLY_SECRET or ORDERLY_PUBLIC_KEY environment variables are not set!")

# Clean and decode private key
if ORDERLY_SECRET.startswith("ed25519:"):
    ORDERLY_SECRET = ORDERLY_SECRET.replace("ed25519:", "")
private_key = Ed25519PrivateKey.from_private_bytes(b58decode(ORDERLY_SECRET))

# Redis setup (optional)
try:
    import redis
    redis_client = redis.StrictRedis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        socket_connect_timeout=1
    )
    redis_client.ping()
except (ImportError, redis.ConnectionError, ValueError, OSError):
    redis_client = None

logger = logging.getLogger(__name__)

async def fetch_orderly_klines(symbol: str, interval: str = "5m", limit: int = 2) -> Optional[pd.DataFrame]:
    """Fetch klines data from Orderly (authenticated endpoint)"""
    try:
        timestamp = str(int(time.time() * 1000))
        params = {"symbol": symbol, "type": interval, "limit": limit}
        path = "/v1/kline"
        query = f"?{urllib.parse.urlencode(params)}"
        message = f"{timestamp}GET{path}{query}"
        signature = urlsafe_b64encode(private_key.sign(message.encode())).decode()

        headers = {
            "orderly-timestamp": timestamp,
            "orderly-account-id": ORDERLY_ACCOUNT_ID,
            "orderly-key": ORDERLY_PUBLIC_KEY,
            "orderly-signature": signature,
        }

        url = f"{BASE_URL}{path}{query}"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and "data" in data and "rows" in data["data"]:
                    df = pd.DataFrame(data["data"]["rows"])
                    if not df.empty:
                        # Keep column names consistent with your historical fetcher
                        df = df.rename(columns={
                            "start_timestamp": "timestamp",
                            "open": "open",
                            "high": "high",
                            "low": "low",
                            "close": "close",
                            "volume": "volume"
                        })
                        # Ensure we have the needed columns
                        required = ["timestamp", "open", "high", "low", "close", "volume"]
                        if all(col in df.columns for col in required):
                            return df[required]
            else:
                logger.debug(f"Non-200 response for {symbol}: {response.status_code}")
        return None
    except Exception as e:
        logger.debug(f"Error fetching klines for {symbol}: {e}")
        return None

async def get_orderly_symbols(only_perp: bool = True) -> List[str]:
    """Get all symbols from Orderly (public endpoint is sufficient here)"""
    try:
        url = f"{BASE_URL}/v1/public/info"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                symbols = []
                for row in data.get("data", {}).get("rows", []):
                    symbol = row.get("symbol", "")
                    if only_perp and not symbol.startswith("PERP_"):
                        continue
                    symbols.append(symbol)
                return symbols
        return []
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        return []

async def calculate_symbol_change(symbol: str, interval: str) -> Optional[Tuple[str, float, float]]:
    """Calculate percentage change for a symbol"""
    try:
        df = await fetch_orderly_klines(symbol, interval, 2)
        if df is None or len(df) < 2:
            return None

        open_price = float(df["open"].iloc[0])
        close_price = float(df["close"].iloc[-1])
        volume = float(df["volume"].iloc[-1])

        if open_price <= 0:
            return None

        pct_change = ((close_price - open_price) / open_price) * 100
        return (symbol, pct_change, volume)
    except Exception as e:
        logger.debug(f"Error calculating change for {symbol}: {e}")
        return None

async def get_top_gainers_losers_orderly(
    top_n: int = 20,
    min_volume: float = 500,
    interval_minutes: int = 30,
    use_redis: bool = True
) -> Tuple[List[str], List[str]]:
    """Get top gainers and losers from Orderly"""
    cache_key = f"orderly_gainers_losers:{interval_minutes}m"

    if use_redis and redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                data = json.loads(cached)
                return data['gainers'], data['losers']
        except Exception:
            pass

    # Map minutes to Orderly 'type' values
    interval_map = {
        1: "1m", 3: "3m", 5: "5m", 15: "15m",
        30: "30m", 60: "1h", 120: "2h", 240: "4h"
    }
    interval = interval_map.get(interval_minutes, "5m")

    try:
        symbols = await get_orderly_symbols(only_perp=True)
        if not symbols:
            return [], []

        tasks = [calculate_symbol_change(symbol, interval) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates = []
        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            symbol, pct_change, volume = result
            if volume >= min_volume:
                candidates.append((symbol, pct_change, volume))

        candidates.sort(key=lambda x: x[1], reverse=True)
        gainers = [sym for sym, pct, vol in candidates if pct > 0][:top_n]
        losers = [sym for sym, pct, vol in candidates if pct < 0][:top_n]

        if use_redis and redis_client:
            try:
                redis_client.setex(cache_key, 60, json.dumps({"gainers": gainers, "losers": losers}))
            except Exception:
                pass

        return gainers, losers

    except Exception as e:
        logger.error(f"Error in get_top_gainers_losers_orderly: {e}")
        return [], []

# Sync wrapper — name and signature unchanged
def get_top_gainers_losers(
    top_n: int = 20,
    interval_minutes: int = 5,
    min_interval_volume_usdt: float = 500,
    use_redis: bool = False
) -> Tuple[List[str], List[str]]:
    """Sync wrapper for the async Orderly function"""
    return asyncio.run(get_top_gainers_losers_orderly(
        top_n=top_n,
        min_volume=min_interval_volume_usdt,
        interval_minutes=interval_minutes,
        use_redis=use_redis
    ))

# # Test function (unchanged)
if __name__ == "__main__":
    async def test():
        gainers, losers = await get_top_gainers_losers_orderly(top_n=30)
        print(f"📈 Top Gainers ({len(gainers)}):")
        for g in gainers:
            print(f"  - {g}")
        print(f"📉 Top Losers ({len(losers)}):")
        for l in losers:
            print(f"  - {l}")
    
    asyncio.run(test())