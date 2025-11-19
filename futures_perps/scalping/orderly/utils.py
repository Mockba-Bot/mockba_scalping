from base64 import urlsafe_b64encode
import os
import time
import requests
import urllib.parse
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# --- Orderly Base URL (no auth needed for public endpoints) ---
BASE_URL = os.getenv("ORDERLY_BASE_URL", "https://api.orderly.org").rstrip()
# === Orderly Config ===
BASE_URL = os.getenv("ORDERLY_BASE_URL", "https://api.orderly.org").rstrip()
ORDERLY_ACCOUNT_ID = os.getenv("ORDERLY_ACCOUNT_ID")
ORDERLY_SECRET = os.getenv("ORDERLY_SECRET")  # Base58 private key (e.g., "ed25519:...")
ORDERLY_PUBLIC_KEY = os.getenv("ORDERLY_PUBLIC_KEY")

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

# --- Simple Rate Limiter for Public Endpoints (10 req/sec) ---
class PublicRateLimiter:
    def __init__(self, max_calls: int = 10, period: float = 1.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def __call__(self):
        now = time.time()
        self.calls = [call for call in self.calls if call > now - self.period]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            time.sleep(max(0, sleep_time))
        self.calls.append(time.time())

public_rate_limiter = PublicRateLimiter(max_calls=10, period=1)

# --------------------------------------------------------------------------------------------------
# ✅ DROP-IN REPLACEMENT FUNCTIONS (Binance-style interface, Orderly backend)
# --------------------------------------------------------------------------------------------------

def get_orderbook(symbol: str, limit: int = 5) -> Dict[str, List[List[str]]]:
    """
    Fetch authenticated order book from Orderly (required for PERP_*_USDC).
    Returns: {"bids": [["price","qty"], ...], "asks": [["price","qty"], ...]}
    """
    public_rate_limiter()  # or your global rate_limiter() if preferred

    max_level = min(limit, 500)
    path = f"/v1/orderbook/{symbol}"
    query = f"?max_level={max_level}"

    # Sign the request
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
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return {"bids": [], "asks": []}

        payload = response.json()
        if not payload.get("success") or "data" not in payload:
            return {"bids": [], "asks": []}

        data = payload["data"]
        bids = [[str(b["price"]), str(b["quantity"])] for b in data.get("bids", [])]
        asks = [[str(a["price"]), str(a["quantity"])] for a in data.get("asks", [])]
        return {"bids": bids, "asks": asks}

    except Exception:
        return {"bids": [], "asks": []}


def get_recent_trades(symbol: str, minutes: int = 15) -> List[Dict]:
    """
    Fetch recent PUBLIC market trades from Orderly using /v1/public/market_trades.
    
    Args:
        symbol (str): e.g., "PERP_BTC_USDC"
        minutes (int): Time window in minutes (used to filter after fetch)

    Returns:
        List[Dict]: Each trade has 'price', 'qty', 'time' (ms), 'side'
                    Compatible with your sweep detector.
    """
    public_rate_limiter()
    
    # Orderly allows up to 1000 trades, but we request 1000 to be safe
    limit = 1000
    url = f"{BASE_URL}/v1/public/market_trades"
    try:
        response = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get("success") or "data" not in data or "rows" not in data["data"]:
            return []

        trades = data["data"]["rows"]
        cutoff_ms = (time.time() - minutes * 60) * 1000

        # Convert to Binance-like format and filter by time
        recent_trades = []
        for t in trades:
            ts = t.get("executed_timestamp")
            if ts >= cutoff_ms:
                recent_trades.append({
                    "price": str(t["executed_price"]),
                    "qty": str(t["executed_quantity"]),
                    "time": ts,
                    "side": t.get("side", "UNKNOWN")  # 'BUY' or 'SELL'
                })
        return recent_trades
    except Exception:
        return []


def calc_spread_pct(symbol: str) -> float:
    """Calculate bid-ask spread as % of mid price."""
    ob = get_orderbook(symbol, limit=1)
    if not ob["bids"] or not ob["asks"]:
        return float('inf')
    try:
        bid = float(ob["bids"][0][0])
        ask = float(ob["asks"][0][0])
        if bid <= 0 or ask <= 0:
            return float('inf')
        mid = (bid + ask) / 2.0
        return (ask - bid) / mid
    except (IndexError, ValueError, ZeroDivisionError):
        return float('inf')


def has_sufficient_depth(symbol: str, min_usdt_depth: float = 10) -> bool:
    """
    Check if top 3 levels have enough USDC value (assumes USDC quote).
    """
    ob = get_orderbook(symbol, limit=3)
    try:
        bid_depth = sum(float(p) * float(q) for p, q in ob["bids"][:3])
        ask_depth = sum(float(p) * float(q) for p, q in ob["asks"][:3])
        return min(bid_depth, ask_depth) >= min_usdt_depth
    except Exception:
        return False
    
def get_klines(symbol: str, interval: str, limit: int = 1) -> List[Dict]:
    """
    Fetch klines from Orderly using AUTHENTICATED endpoint.
    Returns list of kline dicts (same as your gainers/losers script).
    Compatible with: kline["amount"] = USDC volume.
    """
    public_rate_limiter()  # or use your global rate_limiter

    # Build query
    params = {"symbol": symbol, "type": interval, "limit": min(limit, 100)}
    query_items = [f"{k}={v}" for k, v in sorted(params.items())]
    query_str = "?" + "&".join(query_items)

    # Sign request
    timestamp = str(int(time.time() * 1000))
    message = f"{timestamp}z{query_str}"
    signature = urlsafe_b64encode(private_key.sign(message.encode())).decode()

    headers = {
        "orderly-timestamp": timestamp,
        "orderly-account-id": ORDERLY_ACCOUNT_ID,
        "orderly-key": ORDERLY_PUBLIC_KEY,
        "orderly-signature": signature,
    }

    url = f"{BASE_URL}/v1/kline{query_str}"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and "data" in data and "rows" in data["data"]:
                return data["data"]["rows"]
    except Exception as e:
        print(f"Kline fetch failed for {symbol}: {e}")
    return []

def get_funding_rate(symbol: str) -> float:
    """Get current funding rate (decimal, e.g., -0.000012)"""
    public_rate_limiter()
    url = f"{BASE_URL}/v1/public/funding_rate"
    try:
        response = requests.get(url, params={"symbol": symbol}, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("success") and "data" in data:
            return float(data["data"].get("funding_rate", "0"))
    except Exception as e:
        print(f"Funding rate fetch failed for {symbol}: {e}")
    return 0.0    