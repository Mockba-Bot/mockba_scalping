from datetime import datetime
import os
import time
import requests
import json
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Optional Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

load_dotenv()

# === Binance Config ===
BINANCE_BASE = "https://fapi.binance.com"  # ✅ Futures API

# 🔧 Filters (adjusted for real-world scalping)
MIN_PRICE_USDT = 0.0001
MAX_PRICE_USDT = 1_000_000
MAX_SPREAD_PCT = 0.0004      # 0.04% - crucial for scalping
MIN_DEPTH_USDT = 500000      # Higher liquidity for easy exits
MIN_VOLUME_5M_USDT = 500000  # High volume pairs only
MAX_VOLATILITY_5M = 0.08      # 8% max 5m volatility
# Only trade when there's directional movement
MIN_DIRECTIONAL_MOVEMENT = 0.002  # 0.2% minimum move

# Stable-stable pairs to exclude
STABLE_STABLE_PAIRS = {
    "USDCUSDT", "BUSDUSDT", "DAIUSDT", "TUSDUSDT",
    "PAXGUSDT", "FDUSDUSDT", "USDPUSDT", "LUSDUSDT"
}

# Redis
redis_client = None
if REDIS_AVAILABLE:
    try:
        redis_client = redis.StrictRedis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
            socket_connect_timeout=1
        )
        redis_client.ping()
    except Exception:
        redis_client = None

# === Fetch Tradable USDT-M Futures Symbols ===
def get_binance_perp_symbols() -> List[str]:
    try:
        resp = requests.get(f"{BINANCE_BASE}/fapi/v1/exchangeInfo", timeout=5)
        if resp.status_code != 200:
            return []
        data = resp.json()
        symbols = []
        for s in data.get("symbols", []):
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING":
                symbol = s["symbol"]
                # Skip stable-stable pairs
                if symbol in STABLE_STABLE_PAIRS:
                    continue
                symbols.append(symbol)
        return symbols
    except Exception:
        return []

# === Fetch Order Book (Top 5 Levels) ===
def fetch_binance_orderbook(symbol: str, limit: int = 5) -> Optional[Dict]:
    try:
        resp = requests.get(
            f"{BINANCE_BASE}/fapi/v1/depth",
            params={"symbol": symbol, "limit": limit},
            timeout=3
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        bids = [[float(p), float(q)] for p, q in data.get("bids", [])[:limit]]
        asks = [[float(p), float(q)] for p, q in data.get("asks", [])[:limit]]
        return {"bids": bids, "asks": asks}
    except Exception:
        return None

# === Fetch 5m Klines ===
def fetch_binance_klines(symbol: str, interval: str = "5m", limit: int = 1) -> Optional[Dict]:
    try:
        resp = requests.get(
            f"{BINANCE_BASE}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=3
        )
        if resp.status_code != 200:
            return None
        k = resp.json()[0]
        return {
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "quote_volume": float(k[7]),
        }
    except Exception:
        return None

# === Liquidity Score  ===
def calculate_liquidity_score(spread_pct: float, depth_usdt: float, volume_5m: float) -> float:
    spread_score = max(0, 0.001 - spread_pct) * 20000
    depth_score = min(depth_usdt / 2000, 10.0)
    volume_score = min(volume_5m / 200000, 5.0)
    return round((spread_score * 0.5) + (depth_score * 0.3) + (volume_score * 0.2), 2)

# === Analyze One Symbol ===
def analyze_binance_symbol(symbol: str, debug: bool = False) -> Optional[Dict]:
    try:
        if debug:
            print(f"[{symbol}] Fetching order book...")
        ob = fetch_binance_orderbook(symbol, limit=5)
        if not ob or not ob["bids"] or not ob["asks"]:
            if debug:
                print(f"[{symbol}] ❌ Order book empty or fetch failed")
            return None

        best_bid = ob["bids"][0][0]
        best_ask = ob["asks"][0][0]
        mid_price = (best_bid + best_ask) / 2

        if not (MIN_PRICE_USDT <= mid_price <= MAX_PRICE_USDT):
            if debug:
                print(f"[{symbol}] ❌ Price out of range: {mid_price}")
            return None

        spread_pct = (best_ask - best_bid) / mid_price
        if spread_pct > MAX_SPREAD_PCT:
            if debug:
                print(f"[{symbol}] ❌ Spread too wide: {spread_pct:.4%}")
            return None

        # Depth: top 3 levels
        bid_depth = sum(p * q for p, q in ob["bids"][:3])
        ask_depth = sum(p * q for p, q in ob["asks"][:3])
        min_depth = min(bid_depth, ask_depth)

        if min_depth < MIN_DEPTH_USDT:
            if debug:
                print(f"[{symbol}] ❌ Insufficient depth: ${min_depth:,.2f}")
            return None

        if debug:
            print(f"[{symbol}] Fetching klines...")
        kline = fetch_binance_klines(symbol, "5m", 1)
        if not kline:
            if debug:
                print(f"[{symbol}] ❌ Kline fetch failed")
            return None

        quote_volume = kline["quote_volume"]
        if quote_volume < MIN_VOLUME_5M_USDT:
            if debug:
                print(f"[{symbol}] ❌ Low 5m volume: ${quote_volume:,.2f}")
            return None

        volatility = (kline["high"] - kline["low"]) / kline["low"] if kline["low"] > 0 else 0
        if volatility > MAX_VOLATILITY_5M:
            if debug:
                print(f"[{symbol}] ❌ High volatility: {volatility:.2%}")
            return None

        score = calculate_liquidity_score(spread_pct, min_depth, quote_volume)

        return {
            "symbol": symbol,
            "price": mid_price,
            "spread_pct": round(spread_pct * 100, 4),
            "depth_usdt": round(min_depth, 2),
            "volume_5m": round(quote_volume, 2),
            "volatility_5m": round(volatility * 100, 2),
            "liquidity_score": score,
            "timestamp": time.time()
        }

    except Exception as e:
        if debug:
            print(f"[{symbol}] 💥 Exception: {e}")
        return None

# === Main Functions ===
def get_top_liquidity_symbols(top_n: int = 20, use_redis: bool = True) -> List[str]:
    cache_key = f"binance_liquidity_top_{top_n}"
    if use_redis and redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    symbols = get_binance_perp_symbols()
    if not symbols:
        return []

    # Limit to top 50 to avoid rate limits
    candidates = symbols[:50]
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(analyze_binance_symbol, sym): sym for sym in candidates}
        for future in as_completed(future_to_symbol):
            try:
                res = future.result(timeout=8)
                if res:
                    results.append(res)
            except Exception:
                continue

    results.sort(key=lambda x: x["liquidity_score"], reverse=True)
    top_symbols = [r["symbol"] for r in results[:top_n]]

    if use_redis and redis_client and top_symbols:
        try:
            redis_client.setex(cache_key, 120, json.dumps(top_symbols))
        except Exception:
            pass

    return top_symbols

def get_detailed_liquidity_analysis_binance(top_n: int = 10) -> List[Dict]:
    symbols = get_top_liquidity_symbols_binance(top_n)
    analysis = []
    for sym in symbols:
        data = analyze_binance_symbol(sym)
        if data:
            analysis.append(data)
    return analysis

def print_liquidity_report_binance(top_n: int = 15):
    print(f"\n{'='*80}")
    print(f"📊 BINANCE LIQUIDITY REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    data = get_detailed_liquidity_analysis_binance(top_n)
    if not data:
        print("❌ No liquid symbols found. Retrying with debug on for BTCUSDT...")
        # Debug BTCUSDT
        btc_data = analyze_binance_symbol("BTCUSDT", debug=True)
        eth_data = analyze_binance_symbol("ETHUSDT", debug=True)
        print(f"\n🔍 BTCUSDT result: {btc_data}")
        print(f"🔍 ETHUSDT result: {eth_data}")
        return

    print(f"\n{'Rank':<4} {'Symbol':<12} {'Score':<6} {'Spread':<8} {'Depth':<10} {'5m Vol':<12} {'Price':<10} {'Volatility'}")
    print("-" * 80)
    for i, d in enumerate(data, 1):
        print(
            f"{i:<4} {d['symbol']:<12} {d['liquidity_score']:<6} "
            f"{d['spread_pct']:.3f}%  ${d['depth_usdt']:>7,.0f}  "
            f"${d['volume_5m']:>9,.0f}  ${d['price']:>7.2f}  "
            f"{d['volatility_5m']:.2f}%"
        )

    avg_spread = sum(d['spread_pct'] for d in data) / len(data)
    avg_depth = sum(d['depth_usdt'] for d in data) / len(data)
    avg_vol = sum(d['volume_5m'] for d in data) / len(data)
    print("-" * 80)
    print(f"📈 Averages: Spread: {avg_spread:.3f}% | Depth: ${avg_depth:,.0f} | 5m Vol: ${avg_vol:,.0f}")
    print("=" * 80)

# === Run ===
if __name__ == "__main__":
    print_liquidity_report_binance(20)