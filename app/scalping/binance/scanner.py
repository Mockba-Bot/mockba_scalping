from .utils import get_orderbook, get_recent_trades
from typing import Optional, Dict
import time

# Thresholds — tune these based on manual observation
SWEEP_THRESHOLD_PCT = 0.003    # 0.3% beyond fair price to qualify as a sweep
VOLUME_SPIKE_MULTIPLIER = 2.0  # Volume must be ≥2x recent median
PULLBACK_CONFIRM_PCT = 0.001   # Price must pull back by at least 0.1%
MAX_SPREAD_PCT = 0.0005        # 0.05% max allowed spread
TP_PCT = 0.005                 # 0.5%

def detect_liquidity_sweep(symbol: str, lookback_sec: int = 300) -> Optional[Dict]:
    """
    Detect a liquidity sweep followed by a fast reversal using real-time trade and order book data.
    
    Strategy logic:
      1. A "sweep" occurs when price spikes beyond recent fair value (liquidity grab).
      2. A "reversal" is confirmed when price pulls back into fair range on high volume.
      3. Only valid if spread is tight and order book is deep (checked upstream or here).

    Args:
        symbol (str): Binance trading pair (e.g., 'SOLUSDT')
        lookback_sec (int): Time window (in seconds) to analyze recent activity (default: 300 = 5 min)

    Returns:
        Optional[Dict]: Signal with trade parameters if valid reversal detected, else None.
        Example:
        {
            "symbol": "SOLUSDT",
            "side": "buy",          # or "sell"
            "entry": 170.51,        # aggressive limit: best ask (for buy) or best bid (for sell)
            "stop_loss": 170.40,
            "take_profit": 170.68,
            "confidence": 2.4       # volume spike ratio (higher = stronger signal)
        }

    Note:
        - Uses **tick data (trades)**, not candles → low latency.
        - Designed for **1–5 minute scalping** on high-volume USDT pairs.
        - Assumes caller already filtered for spread and depth (but double-checks spread).
    """
    # Fetch recent public trades (up to 1000 most recent)
    trades = get_recent_trades(symbol, minutes=lookback_sec // 60 + 1)
    if len(trades) < 20:
        return None  # Not enough market activity

    # Filter to only trades within the exact lookback window
    now_ms = time.time() * 1000
    recent_trades = [
        t for t in trades
        if now_ms - t['time'] <= lookback_sec * 1000
    ]
    if len(recent_trades) < 10:
        return None

    # Extract price and volume series
    prices = [float(t['price']) for t in recent_trades]
    volumes = [float(t['qty']) for t in recent_trades]

    # Compute robust "fair price" as median of last 60 seconds (resistant to spikes)
    fair_window = [
        p for t, p in zip(recent_trades, prices)
        if now_ms - t['time'] <= 60_000
    ]
    if not fair_window:
        return None
    fair_price = sorted(fair_window)[len(fair_window) // 2]

    # Get current best bid and ask for entry and pullback confirmation
    ob = get_orderbook(symbol, limit=1)
    if not ob.get('bids') or not ob.get('asks'):
        return None
    best_bid = float(ob['bids'][0][0])
    best_ask = float(ob['asks'][0][0])

    # Double-check spread (defensive: should be filtered upstream)
    mid_price = (best_bid + best_ask) / 2.0
    spread_pct = (best_ask - best_bid) / mid_price
    if spread_pct > MAX_SPREAD_PCT:
        return None

    # Focus on the most recent trade as potential sweep trigger
    last_trade = recent_trades[-1]
    last_price = float(last_trade['price'])
    last_qty = float(last_trade['qty'])

    # Compute recent volume median for spike detection
    recent_vols = volumes[-30:] if len(volumes) >= 30 else volumes
    med_vol = sum(recent_vols) / len(recent_vols)  # simple mean is fine for MVP
    vol_spike = last_qty >= med_vol * VOLUME_SPIKE_MULTIPLIER

    # 🔻 BEARISH LIQUIDITY REVERSAL: sweep above fair, then pullback down
    if last_price > fair_price * (1 + SWEEP_THRESHOLD_PCT):
        # Confirm price has already pulled back below the sweep extreme
        if best_bid < last_price * (1 - PULLBACK_CONFIRM_PCT) and vol_spike:
            stop_loss = last_price * (1 + PULLBACK_CONFIRM_PCT)  # just above sweep wick
            take_profit = best_bid * (1 - TP_PCT)
            return {
                "symbol": symbol,
                "side": "sell",
                "entry": best_bid,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": last_qty / med_vol
            }

    # 🔺 BULLISH LIQUIDITY REVERSAL: sweep below fair, then pullback up
    if last_price < fair_price * (1 - SWEEP_THRESHOLD_PCT):
        if best_ask > last_price * (1 + PULLBACK_CONFIRM_PCT) and vol_spike:
            stop_loss = last_price * (1 - PULLBACK_CONFIRM_PCT)  # just below sweep wick
            take_profit = best_ask * (1 + TP_PCT)
            return {
                "symbol": symbol,
                "side": "buy",
                "entry": best_ask,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": last_qty / med_vol
            }

    return None


# testing
# if __name__ == "__main__":
#     test_symbol = "BTCUSDT"
#     print(f"🔍 Scanning {test_symbol} for liquidity sweep signals...")
#     signal = detect_liquidity_sweep(test_symbol)
#     if signal:
#         print("✅ Scalp signal detected:")
#         for k, v in signal.items():
#             print(f"   {k}: {v}")
#     else:
#         print("❌ No valid scalp signal detected.")