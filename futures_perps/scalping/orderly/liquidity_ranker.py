from datetime import datetime
import os
import time
import requests
import json
import threading
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException
from dotenv import load_dotenv
import pandas as pd
import random
from utils import get_orderbook, get_klines

# Optional: Only import if you use Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Load environment
load_dotenv()

# === Orderly Config ===
BASE_URL = os.getenv("ORDERLY_BASE_URL", "https://api.orderly.org").rstrip()
ORDERLY_ACCOUNT_ID = os.getenv("ORDERLY_ACCOUNT_ID")
ORDERLY_SECRET = os.getenv("ORDERLY_SECRET")  # Base58 private key (e.g., "ed25519:...")
ORDERLY_PUBLIC_KEY = os.getenv("ORDERLY_PUBLIC_KEY")

from datetime import datetime, timezone

# === Detect weekend in UTC ===
utc_now = datetime.now(timezone.utc)
is_weekend = utc_now.weekday() >= 5  # 5 = Saturday, 6 = Sunday (UTC)

# === 15m LIQUIDITY THRESHOLDS (Auto-adjust for weekends) ===
MIN_PRICE_FILTER = 0.001
MAX_PRICE_FILTER = 500_000
MAX_SPREAD_PCT = 0.001       # 0.1% max spread
MIN_DEPTH_USDC = 15 if is_weekend else 25
MIN_VOLUME_1h = 150 if is_weekend else 400
MAX_VOLATILITY_1h = 0.09    # 9% (kept constant — alts pump even on weekends)

if not all([BASE_URL, ORDERLY_ACCOUNT_ID, ORDERLY_SECRET, ORDERLY_PUBLIC_KEY]):
    raise ValueError("❌ Missing required ORDERLY_* environment variables")

# === Handle Private Key ===
from base58 import b58decode
from base64 import urlsafe_b64encode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

clean_secret = ORDERLY_SECRET.replace("ed25519:", "").strip()
try:
    private_key_bytes = b58decode(clean_secret)
    if len(private_key_bytes) != 32:
        raise ValueError("Private key must be 32 bytes after Base58 decode")
    private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
except Exception as e:
    raise ValueError(f"❌ Invalid ORDERLY_SECRET: {e}")

# === Rate Limiter (10 RPS) ===
class RateLimiter:
    def __init__(self, max_calls=10, period=1):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()
    def __call__(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if t > now - self.period]
            if len(self.calls) >= self.max_calls:
                sleep_s = self.period - (now - self.calls[0])
                time.sleep(max(0, sleep_s))
            self.calls.append(time.time())

rate_limiter = RateLimiter(max_calls=10, period=1)

# === Redis (Optional) ===
redis_client = None
if REDIS_AVAILABLE:
    try:
        redis_client = redis.StrictRedis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            socket_connect_timeout=1,
            decode_responses=True
        )
        redis_client.ping()
    except Exception:
        redis_client = None

# === Sign Request (Matches your example) ===
def sign_orderly_request(method: str, path: str, query: str = "") -> dict:
    timestamp = str(int(time.time() * 1000))
    message = f"{timestamp}{method}{path}{query}"
    signature = urlsafe_b64encode(private_key.sign(message.encode())).decode()
    return {
        "orderly-timestamp": timestamp,
        "orderly-account-id": ORDERLY_ACCOUNT_ID,
        "orderly-key": ORDERLY_PUBLIC_KEY,
        "orderly-signature": signature,
    }

# === Liquidity Scoring (Updated for 1h) ===
def calculate_liquidity_score(spread_pct: float, depth_usdc: float, volume_1h: float) -> float:
    spread_score = max(0, 0.002 - spread_pct) * 10000
    depth_score = min(depth_usdc / 500, 15.0)
    volume_score = min(volume_1h / 50_000, 5.0)
    return round((spread_score * 0.5) + (depth_score * 0.4) + (volume_score * 0.1), 2)

# === Analyze One Symbol (1h) ===
def fetch_symbol_liquidity(symbol: str) -> Optional[Dict]:
    try:
        ob = get_orderbook(symbol, limit=3)
        if not ob["bids"] or not ob["asks"]:
            return None

        try:
            bids = [(float(p), float(q)) for p, q in ob["bids"][:3]]
            asks = [(float(p), float(q)) for p, q in ob["asks"][:3]]
        except (ValueError, IndexError):
            return None

        if len(bids) < 3 or len(asks) < 3:
            return None

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread_pct = (best_ask - best_bid) / mid_price

        if spread_pct > MAX_SPREAD_PCT:
            return None

        bid_depth = sum(p * q for p, q in bids)
        ask_depth = sum(p * q for p, q in asks)
        min_depth = min(bid_depth, ask_depth)

        if min_depth < MIN_DEPTH_USDC:
            return None

        # === Use 15m klines (last 2 candles for robustness) ===
        klines = get_klines(symbol, "1h", limit=2)
        if not klines or len(klines) < 1:
            return None

        # Average volume over available candles (min 1, ideally 2)
        volumes = []
        volatilities = []
        for k in klines[:2]:
            try:
                amt = float(k.get("amount", 0))
                high = float(k["high"])
                low = float(k["low"])
                vol = (high - low) / low if low > 0 else 0
                volumes.append(amt)
                volatilities.append(vol)
            except (KeyError, ValueError, TypeError):
                continue

        if not volumes:
            return None

        avg_volume = sum(volumes) / len(volumes)
        max_volatility = max(volatilities)  # use worst-case volatility

        if avg_volume < MIN_VOLUME_1h:
            return None
        if max_volatility > MAX_VOLATILITY_1h:
            return None

        score = calculate_liquidity_score(spread_pct, min_depth, avg_volume)
        return {
            "symbol": symbol,
            "price": mid_price,
            "spread_pct": round(spread_pct * 100, 4),
            "depth_usdt": round(min_depth, 2),
            "volume_1h": round(avg_volume, 2),
            "volatility_1h": round(max_volatility * 100, 2),
            "liquidity_score": score,
            "timestamp": time.time()
        }

    except Exception:
        return None

# === Get Tradable Symbols ===
def get_scalping_candidates() -> List[str]:
    rate_limiter()
    try:
        resp = requests.get(f"{BASE_URL}/v1/public/info", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            rows = data.get("data", {}).get("rows", [])
            return [
                r["symbol"] for r in rows
                if isinstance(r, dict)
                and r.get("symbol", "").startswith("PERP_")
                and r.get("symbol", "").endswith("_USDC")
            ]
    except Exception:
        pass
    return []

# === Main Functions ===
def get_top_liquidity_symbols(top_n: int = 20, use_redis: bool = True) -> List[str]:
    cache_key = f"orderly_liquidity_top_{top_n}_15m"
    if use_redis and redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    candidates = get_scalping_candidates()
    if not candidates:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=min(10, len(candidates))) as executor:
        futures = {executor.submit(fetch_symbol_liquidity, sym): sym for sym in candidates[:60]}
        for future in as_completed(futures):
            try:
                res = future.result(timeout=10)
                if res:
                    results.append(res)
            except Exception:
                continue

    results.sort(key=lambda x: x["liquidity_score"], reverse=True)
    top_symbols = [r["symbol"] for r in results[:top_n]]

    if use_redis and redis_client and top_symbols:
        try:
            redis_client.setex(cache_key, 180, json.dumps(top_symbols))  # 3-min cache
        except Exception:
            pass

    return top_symbols

def get_detailed_liquidity_analysis(top_n: int = 10) -> List[Dict]:
    symbols = get_top_liquidity_symbols(top_n, use_redis=False)
    analysis = []
    for s in symbols:
        res = fetch_symbol_liquidity(s)
        if res:
            analysis.append(res)
    return analysis

def print_liquidity_report(top_n: int = 15):
    print(f"\n{'='*80}")
    print(f"📊 ORDERLY LIQUIDITY REPORT (15m) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    data = get_detailed_liquidity_analysis(top_n)
    if not data:
        print("❌ No liquid symbols found")
        return

    print(f"\n{'Rank':<4} {'Symbol':<16} {'Score':<6} {'Spread':<8} {'Depth':<10} {'1h Vol':<12} {'Price':<10} {'Volatility'}")
    print("-" * 80)
    for i, d in enumerate(data, 1):
        print(
            f"{i:<4} {d['symbol']:<16} {d['liquidity_score']:<6} "
            f"{d['spread_pct']:.3f}%  ${d['depth_usdt']:>7,.0f}  "
            f"${d['volume_1h']:>9,.0f}  ${d['price']:>7.2f}  "
            f"{d['volatility_1h']:.2f}%"
        )

    avg_spread = sum(d['spread_pct'] for d in data) / len(data)
    avg_depth = sum(d['depth_usdt'] for d in data) / len(data)
    avg_vol = sum(d['volume_1h'] for d in data) / len(data)
    print("-" * 80)
    print(f"📈 Averages: Spread: {avg_spread:.3f}% | Depth: ${avg_depth:,.0f} | 1h Vol: ${avg_vol:,.0f}")
    print("=" * 80)

# === Run ===
if __name__ == "__main__":
    print_liquidity_report(12)