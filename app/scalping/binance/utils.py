import time
import requests
from typing import Dict, List, Optional

# Base URL for Binance Spot REST API (public endpoints)
BINANCE_SPOT_BASE = "https://api.binance.com"  # ← Fixed: no trailing spaces


def get_orderbook(symbol: str, limit: int = 5) -> Dict:
    """
    Fetch the current order book (bid/ask levels) for a Binance Spot symbol.

    Args:
        symbol (str): Trading pair in Binance format (e.g., 'BTCUSDT', not 'BTC/USDT')
        limit (int): Number of bid/ask levels to fetch (valid: 5, 10, 20, 50, 100, 500, 1000)

    Returns:
        Dict: Order book with keys 'lastUpdateId', 'bids', 'asks'
              Example: {'bids': [['170.50', '10.2'], ...], 'asks': [['170.51', '8.7'], ...]}

    Note:
        - Uses public API — no authentication needed.
        - Bids are sorted best (highest) to worst; asks are best (lowest) to worst.
    """
    res = requests.get(
        f"{BINANCE_SPOT_BASE}/api/v3/depth",
        params={"symbol": symbol, "limit": limit}
    )
    res.raise_for_status()  # Raise error on HTTP failure (e.g., 404, 429)
    return res.json()


def get_recent_trades(symbol: str, minutes: int = 15) -> List[Dict]:
    """
    Retrieve recent public trades for a symbol and filter to last N minutes.

    Args:
        symbol (str): Binance trading pair (e.g., 'SOLUSDT')
        minutes (int): Time window in minutes (default: 15)

    Returns:
        List[Dict]: List of trade objects from last `minutes` minutes.
                    Each trade: {'id': int, 'price': str, 'qty': str, 'time': int (ms), ...}

    Note:
        - Binance returns max 1000 most recent trades regardless of time.
        - In high-volume pairs (BTC, ETH, SOL), 1000 trades ≈ <1 minute — sufficient for scalping.
        - In low-volume pairs, may return fewer trades than expected.
    """
    res = requests.get(
        f"{BINANCE_SPOT_BASE}/api/v3/trades",
        params={"symbol": symbol, "limit": 1000}  # Max allowed
    )
    res.raise_for_status()
    trades = res.json()

    # Filter trades to only those within the last `minutes` minutes
    cutoff_ms = (time.time() - minutes * 60) * 1000
    recent_trades = [t for t in trades if t['time'] >= cutoff_ms]
    return recent_trades


def calc_spread_pct(symbol: str) -> float:
    """
    Calculate the current bid-ask spread as a percentage of mid price.

    Args:
        symbol (str): Binance trading pair (e.g., 'BNBUSDT')

    Returns:
        float: Spread percentage (e.g., 0.0004 = 0.04%)

    Purpose:
        - Core part of "Spread Awareness": avoid pairs with wide spreads (>0.05%)
        - Ensures low slippage and fee efficiency for micro trades.
    """
    ob = get_orderbook(symbol, limit=1)
    bid = float(ob['bids'][0][0])
    ask = float(ob['asks'][0][0])
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid


def has_sufficient_depth(symbol: str, min_usdt_depth: float = 500.0) -> bool:
    """
    Check if top levels of order book have enough liquidity (in USDT value) to fill small orders instantly.

    Args:
        symbol (str): Binance trading pair (must be quoted in USDT, e.g., 'XRPUSDT')
        min_usdt_depth (float): Minimum USD value required on both bid and ask sides (default: $500)

    Returns:
        bool: True if both bid and ask sides have ≥ `min_usdt_depth` in top 3 levels

    Why top 3?
        - Ensures your $2–$5 aggressive limit order will fill immediately.
        - Prevents trading in illiquid pairs where price moves during fill.
    """
    ob = get_orderbook(symbol, limit=3)
    # Calculate total USDT value in top 3 bid levels
    bid_depth = sum(float(price) * float(qty) for price, qty in ob['bids'][:3])
    # Calculate total USDT value in top 3 ask levels
    ask_depth = sum(float(price) * float(qty) for price, qty in ob['asks'][:3])
    # Require both sides to be sufficiently deep
    return min(bid_depth, ask_depth) >= min_usdt_depth