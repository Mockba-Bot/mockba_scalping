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
import time

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

MIN_PRICE_FILTER = 0.001
MAX_PRICE_FILTER = 500_000
MAX_SPREAD_PCT = 0.008     # keep tight spread (good for scalping)
MIN_DEPTH_USDC = 100        # ↓ from 200
MIN_VOLUME_5M = 300       # ↓ from 50,000
MAX_VOLATILITY_5M = 0.03    # ↑ from 0.03 (5% in 5m is normal)

if not all([BASE_URL, ORDERLY_ACCOUNT_ID, ORDERLY_SECRET, ORDERLY_PUBLIC_KEY]):
    raise ValueError("❌ Missing required ORDERLY_* environment variables")

# === Handle Private Key ===
from base58 import b58decode
from base64 import urlsafe_b64encode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

# Clean and decode private key
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
    """Sign request using Ed25519 private key (Base58 input, Base64 output)."""
    timestamp = str(int(time.time() * 1000))
    message = f"{timestamp}{method}{path}{query}"
    signature = urlsafe_b64encode(private_key.sign(message.encode())).decode()
    return {
        "orderly-timestamp": timestamp,
        "orderly-account-id": ORDERLY_ACCOUNT_ID,
        "orderly-key": ORDERLY_PUBLIC_KEY,
        "orderly-signature": signature,
    }

# === Fetch Orderbook (Public - No Auth Needed) ===
def fetch_order_book_snapshot(symbol: str, depth: int = 200):
    """
    Signed orderbook snapshot. Returns DataFrame with ['price','quantity','side'].
    Adds df.attrs['timestamp_ms'] from the API response.
    """
    rate_limiter()  # Global API rate limiting

    # The API expects max_level (depth) as query param
    path = f"/v1/orderbook/{symbol}"
    query = f"?max_level={depth}"

    timestamp = str(int(time.time() * 1000))
    message = f"{timestamp}GET{path}{query}"
    signature = urlsafe_b64encode(private_key.sign(message.encode())).decode()

    headers = {
        "orderly-timestamp": timestamp,
        "orderly-account-id": ORDERLY_ACCOUNT_ID,
        "orderly-key": ORDERLY_PUBLIC_KEY,
        "orderly-signature": signature,
    }

    url = f"{BASE_URL}{path}{query}"
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        print(f"❌ Error fetching orderbook for {symbol}: {r.text}")
        return None

    payload = r.json() or {}
    if payload.get("success") is False:
        print(f"❌ API returned success=False for {symbol}")
        return None

    data = payload.get("data") or {}
    bids = data.get("bids", []) or []
    asks = data.get("asks", []) or []
    ob_ts = data.get("timestamp")  # ms per docs

    # Build DataFrames
    df_bids = pd.DataFrame(bids, columns=["price", "quantity"]) if bids else pd.DataFrame(columns=["price", "quantity"])
    df_asks = pd.DataFrame(asks, columns=["price", "quantity"]) if asks else pd.DataFrame(columns=["price", "quantity"])

    if not df_bids.empty:
        df_bids["side"] = "bid"
    if not df_asks.empty:
        df_asks["side"] = "ask"

    df = pd.concat([df_bids, df_asks], ignore_index=True)
    if df.empty:
        print(f"⚠️ Empty orderbook for {symbol}")
        return None

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna().sort_values("price").reset_index(drop=True)

    # Attach the exchange-provided OB timestamp (ms)
    if isinstance(ob_ts, (int, float)):
        df.attrs["timestamp_ms"] = int(ob_ts)

    # Basic sanity: best bid <= best ask
    try:
        best_bid = df.loc[df["side"] == "bid", "price"].max()
        best_ask = df.loc[df["side"] == "ask", "price"].min()
        if pd.notna(best_bid) and pd.notna(best_ask) and best_bid > best_ask:
            print(f"⚠️ Crossed book detected for {symbol}, discarding snapshot.")
            return None
    except Exception:
        pass

    return df

# === Fetch Klines (Authenticated) ===
def fetch_klines(symbol: str, interval: str = "5m", limit: int = 2) -> Optional[List[Dict]]:
    rate_limiter()
    params = {"symbol": symbol, "type": interval, "limit": limit}
    query_items = [f"{k}={v}" for k, v in sorted(params.items())]
    query_str = "?" + "&".join(query_items)
    headers = sign_orderly_request("GET", "/v1/kline", query_str)
    url = f"{BASE_URL}/v1/kline{query_str}"
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success", True):
                return data.get("data", {}).get("rows", [])
    except Exception:
        pass
    return None

# === Liquidity Scoring ===
def calculate_liquidity_score(spread_pct: float, depth_usdc: float, volume_5m: float) -> float:
    spread_score = max(0, 0.001 - spread_pct) * 20000
    depth_score = min(depth_usdc / 2000, 10.0)
    volume_score = min(volume_5m / 200000, 5.0)
    return round((spread_score * 0.5) + (depth_score * 0.3) + (volume_score * 0.2), 2)

# === Analyze One Symbol ===
def fetch_symbol_liquidity(symbol: str) -> Optional[Dict]:
    try:
        # ✅ Use the working, signed orderbook fetcher
        df_ob = fetch_order_book_snapshot(symbol, depth=5)
        # print(df_ob)
        if df_ob is None or df_ob.empty:
            return None

        df_bids = df_ob[df_ob["side"] == "bid"].sort_values("price", ascending=False)
        df_asks = df_ob[df_ob["side"] == "ask"].sort_values("price", ascending=True)

        if len(df_bids) < 3 or len(df_asks) < 3:
            return None

        best_bid = float(df_bids.iloc[0]["price"])
        best_ask = float(df_asks.iloc[0]["price"])
        mid_price = (best_bid + best_ask) / 2
        spread_pct = (best_ask - best_bid) / mid_price

        # Apply your filters
        if (spread_pct > MAX_SPREAD_PCT or 
            mid_price < MIN_PRICE_FILTER or 
            mid_price > MAX_PRICE_FILTER):
            return None

        bid_depth = (df_bids.head(3)["price"] * df_bids.head(3)["quantity"]).sum()
        ask_depth = (df_asks.head(3)["price"] * df_asks.head(3)["quantity"]).sum()
        min_depth = min(bid_depth, ask_depth)


        if min_depth < 200:
            return None

        # === Klines (fix this too — see below) ===
        klines = fetch_klines(symbol, "5m", 2)
        if not klines or len(klines) < 1:
            return None

        last = klines[-1]

        # ✅ Orderly returns DICTS, not lists!
        try:
            quote_volume = float(last["amount"])   # USDC volume
            high = float(last["high"])
            low = float(last["low"])
            volatility = (high - low) / low if low > 0 else 0
        except (KeyError, ValueError, TypeError) as e:
            print(f"[{symbol}] ❌ Kline parse error: {e}")
            return None

        # print(f"[{symbol}] 5m → Vol: ${quote_volume:,.0f}, Volatility: {volatility*100:.2f}%")

        if quote_volume < MIN_VOLUME_5M:
            return None
        if volatility > MAX_VOLATILITY_5M:
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
        # For debugging, temporarily print errors
        # print(f"[{symbol}] Error in liquidity eval: {e}")
        return None

# === Get Tradable Symbols ===
def get_scalping_candidates() -> List[str]:
    rate_limiter()
    try:
        resp = requests.get(f"{BASE_URL}/v1/public/info", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # print(data)
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
    cache_key = f"orderly_liquidity_top_{top_n}"
    if use_redis and redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    candidates = get_scalping_candidates()
    # print(f"🔍 Evaluating liquidity for {len(candidates)} candidates")
    if not candidates:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=min(2, len(candidates))) as executor:
        futures = {executor.submit(fetch_symbol_liquidity, sym): sym for sym in candidates[:60]}
        for future in futures:
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
            redis_client.setex(cache_key, 120, json.dumps(top_symbols))
        except Exception:
            pass

    return top_symbols

def get_detailed_liquidity_analysis(top_n: int = 10) -> List[Dict]:
    symbols = get_top_liquidity_symbols(top_n)
    return [fetch_symbol_liquidity(s) for s in symbols if fetch_symbol_liquidity(s)]

def print_liquidity_report(top_n: int = 15):
    print(f"\n{'='*80}")
    print(f"📊 ORDERLY LIQUIDITY REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    data = get_detailed_liquidity_analysis(top_n)
    if not data:
        print("❌ No liquid symbols found")
        return

    print(f"\n{'Rank':<4} {'Symbol':<16} {'Score':<6} {'Spread':<8} {'Depth':<10} {'5m Vol':<12} {'Price':<10} {'Volatility'}")
    print("-" * 80)
    for i, d in enumerate(data, 1):
        print(
            f"{i:<4} {d['symbol']:<16} {d['liquidity_score']:<6} "
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
# if __name__ == "__main__":
#     symbols = print_liquidity_report(20)
