import os
import time
import requests
import urllib.parse
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# --- Orderly Base URL (no auth needed for public endpoints) ---
BASE_URL = os.getenv("ORDERLY_BASE_URL", "https://api.orderly.org").rstrip()

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

def get_orderbook(symbol: str, limit: int = 5) -> Dict:
    """
    Fetch public order book snapshot from Orderly.
    Note: Orderly's public order book endpoint is /v1/public/orderbook/{symbol}
    """
    public_rate_limiter()
    
    # Orderly public orderbook uses 'depth' (not max_level)
    depth_map = {5: 5, 10: 10, 20: 20, 50: 50, 100: 100, 500: 500}
    depth = depth_map.get(limit, 5)
    
    url = f"{BASE_URL}/v1/public/orderbook/{symbol}"
    try:
        response = requests.get(url, params={"depth": depth}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get("success") or "data" not in data:
            return {"bids": [], "asks": []}
        ob = data["data"]
        # Orderly returns: {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}
        bids = [[str(p), str(q)] for p, q in ob.get("bids", [])]
        asks = [[str(p), str(q)] for p, q in ob.get("asks", [])]
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


def has_sufficient_depth(symbol: str, min_usdt_depth: float = 500.0) -> bool:
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