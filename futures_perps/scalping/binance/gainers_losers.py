import os
from dotenv import load_dotenv
import time
import requests
import json
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# Stablecoin base assets to exclude
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
except (ImportError, redis.ConnectionError, ValueError):
    redis_client = None

# ✅ FIXED: No trailing spaces!
BINANCE_BASE = "https://api.binance.com"


def fetch_symbol_klines(symbol: str, interval: str) -> Optional[Tuple[str, float, float]]:
    try:
        klines = requests.get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": 2},
            timeout=3
        ).json()
        if len(klines) < 2:
            return None
        open_price = float(klines[0][1])
        close_price = float(klines[1][4])
        quote_volume = float(klines[1][7])
        if open_price <= 0:
            return None
        pct_change = ((close_price - open_price) / open_price) * 100
        return (symbol, pct_change, quote_volume)
    except Exception:
        return None


def get_top_gainers_losers(
    top_n: int = 20,
    min_24h_volume_usdt: float = 1_000_000,
    interval_minutes: int = 5,
    min_interval_volume_usdt: float = 50_000,
    use_redis: bool = True
) -> Tuple[List[str], List[str]]:
    cache_key = f"scalp_gainers_losers:{interval_minutes}m"
    
    if use_redis and redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                data = json.loads(cached)
                return data['gainers'], data['losers']
        except Exception:
            pass

    try:
        tickers = requests.get(f"{BINANCE_BASE}/api/v3/ticker/24hr").json()
    except Exception:
        return [], []

    # ✅ Filter: only non-stablecoin XXXUSDT pairs with high volume
    usdt_symbols = []
    for t in tickers:
        symbol = t['symbol']
        if not symbol.endswith('USDT'):
            continue
        if is_stablecoin_pair(symbol):
            continue
        if float(t['quoteVolume']) < min_24h_volume_usdt:
            continue
        usdt_symbols.append(symbol)

    interval_map = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h"}
    binance_interval = interval_map.get(interval_minutes, f"{interval_minutes}m")

    candidates = []
    max_workers = min(8, len(usdt_symbols) or 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_symbol_klines, symbol, binance_interval): symbol
            for symbol in usdt_symbols
        }
        for future in as_completed(future_to_symbol):
            result = future.result()
            if result:
                symbol, pct_change, vol = result
                if vol >= min_interval_volume_usdt:
                    candidates.append((symbol, pct_change, vol))

    candidates.sort(key=lambda x: x[1], reverse=True)
    gainers = [sym for sym, pct, vol in candidates if pct > 0][:top_n]
    losers = [sym for sym, pct, vol in candidates if pct < 0][:top_n]

    if use_redis and redis_client:
        try:
            redis_client.setex(cache_key, 60, json.dumps({"gainers": gainers, "losers": losers}))
        except Exception:
            pass

    return gainers, losers


# if __name__ == "__main__":
#     print(f"🔍 Scanning top gainers/losers over last 5 minutes (filtered, parallelized)...")
#     gainers, losers = get_top_gainers_losers(
#         top_n=10,
#         min_24h_volume_usdt=1_000_000,
#         interval_minutes=5,
#         min_interval_volume_usdt=50_000
#     )
#     print(f"\n📈 Top Gainers ({len(gainers)}):")
#     for g in gainers:
#         print(f"  - {g}")
#     print(f"\n📉 Top Losers ({len(losers)}):")
#     for l in losers:
#         print(f"  - {l}")