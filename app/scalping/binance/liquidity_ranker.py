import os
from dotenv import load_dotenv
import requests
import json
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# Stablecoins to exclude (same as before)
STABLECOIN_BASES = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'PAXG', 'FDUSD', 
    'USD', 'EUR', 'GBP', 'SUSD', 'CUSDT', 'UST', 'VAI', 'NUSD',
    'QUSD', 'USDP', 'GUSD', 'LUSD', 'MUSD', 'HUSD', 'USDN'
}

def is_stablecoin_pair(symbol: str) -> bool:
    if not symbol.endswith('USDT'):
        return False
    base = symbol[:-4]
    return base in STABLECOIN_BASES

# Redis (optional)
try:
    import redis
    redis_client = redis.StrictRedis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        socket_connect_timeout=1
    )
    redis_client.ping()
except (ImportError, redis.ConnectionError, ValueError):
    redis_client = None

BINANCE_BASE = "https://api.binance.com"  # No trailing spaces!


def fetch_symbol_liquidity(symbol: str) -> Optional[dict]:
    """
    Fetch real-time liquidity metrics for scalping.
    Optimized for small orders ($2–$5).
    """
    try:
        # 1. Order book (top 3 levels)
        ob = requests.get(
            f"{BINANCE_BASE}/api/v3/depth",
            params={"symbol": symbol, "limit": 3},
            timeout=2
        ).json()

        if not ob.get('bids') or not ob.get('asks'):
            return None

        best_bid = float(ob['bids'][0][0])
        best_ask = float(ob['asks'][0][0])
        mid = (best_bid + best_ask) / 2
        spread_pct = (best_ask - best_bid) / mid

        # Still strict on spread (critical for micro profits)
        if spread_pct > 0.0005:  # > 0.05%
            return None

        # Calculate depth in USDT
        bid_depth = sum(float(p) * float(q) for p, q in ob['bids'][:3])
        ask_depth = sum(float(p) * float(q) for p, q in ob['asks'][:3])
        min_depth = min(bid_depth, ask_depth)

        # ✅ SOFTEN: Only require $500 depth (enough for $5 order)
        if min_depth < 500:
            return None

        # 2. Get 5m volume
        klines = requests.get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": "5m", "limit": 1},
            timeout=2
        ).json()

        if not klines:
            return None

        volume_5m = float(klines[0][7])

        # ✅ SOFTEN: $100k min volume in 5 min
        if volume_5m < 100_000:
            return None

        # Liquidity score: prioritize tight spread + decent depth
        # Normalize depth and volume to similar scales
        depth_score = min_depth / 1000      # $1k depth = 1 point
        volume_score = volume_5m / 100_000  # $100k vol = 1 point
        spread_penalty = spread_pct * 1000  # 0.05% → 0.05 penalty

        liquidity_score = (depth_score + volume_score) - spread_penalty

        return {
            "symbol": symbol,
            "spread_pct": spread_pct,
            "depth_usdt": min_depth,
            "volume_5m": volume_5m,
            "liquidity_score": liquidity_score
        }

    except Exception:
        return None


def get_top_liquidity_symbols(
    top_n: int = 20,
    min_24h_volume: float = 500_000,
    use_redis: bool = True
) -> List[str]:
    """
    Return top N most liquid USDT spot pairs for scalping.
    Uses real-time spread, depth, and volume.
    """
    cache_key = "top_liquidity_symbols"
    if use_redis and redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    # Get all 24h tickers
    try:
        tickers = requests.get(f"{BINANCE_BASE}/api/v3/ticker/24hr").json()
    except Exception:
        return []

    # Filter candidates: USDT, non-stable, high 24h volume
    candidates = []
    for t in tickers:
        symbol = t['symbol']
        if not symbol.endswith('USDT'):
            continue
        if is_stablecoin_pair(symbol):
            continue
        if float(t['quoteVolume']) < min_24h_volume:
            continue
        candidates.append(symbol)

    # Parallel fetch liquidity metrics
    liquidity_data = []
    max_workers = min(8, len(candidates) or 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_symbol_liquidity, sym): sym
            for sym in candidates
        }
        for future in as_completed(future_to_symbol):
            result = future.result()
            if result:
                liquidity_data.append(result)

    # Sort by liquidity score (descending)
    liquidity_data.sort(key=lambda x: x['liquidity_score'], reverse=True)

    top_symbols = [item['symbol'] for item in liquidity_data[:top_n]]

    # Cache for 60 seconds
    if use_redis and redis_client:
        try:
            redis_client.setex(cache_key, 60, json.dumps(top_symbols))
        except Exception:
            pass

    return top_symbols


# 🧪 Test
# if __name__ == "__main__":
#     print("🔍 Fetching top 20 most liquid symbols for scalping...")
#     top_liquid = get_top_liquidity_symbols(top_n=20)
#     print(f"\n💎 Top Liquid Symbols ({len(top_liquid)}):")
#     for i, sym in enumerate(top_liquid, 1):
#         print(f"  {i:2}. {sym}")