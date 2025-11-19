import os
from typing import Optional, Dict, List
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# Binance Futures API
BINANCE_FUTURES_BASE = "https://fapi.binance.com"

# Fee structure (Binance USDT-M futures)
TAKER_FEE = 0.0004  # 0.04%
MAKER_FEE = 0.0002  # 0.02%
TOTAL_FEES_PCT = TAKER_FEE + MAKER_FEE  # 0.06%

# Load parameters from .env
TP_NET_TARGET = float(os.getenv("BINANCE_TP_NET_TARGET", "0.0012"))
TP_PCT = TP_NET_TARGET + TOTAL_FEES_PCT
SWEEP_THRESHOLD_PCT = float(os.getenv("BINANCE_SWEEP_THRESHOLD_PCT", "0.0003"))
VOLUME_SPIKE_MULTIPLIER = float(os.getenv("BINANCE_VOLUME_SPIKE_MULTIPLIER", "1.8"))
PULLBACK_CONFIRM_PCT = float(os.getenv("BINANCE_PULLBACK_CONFIRM_PCT", "0.0"))  # unused but kept for compatibility
MAX_SPREAD_PCT = float(os.getenv("BINANCE_MAX_SPREAD_PCT", "0.001"))
MIN_CONFIDENCE = float(os.getenv("BINANCE_MIN_CONFIDENCE", "0.9"))
MIN_TRADES_FOR_ANALYSIS = int(os.getenv("BINANCE_MIN_TRADES_FOR_ANALYSIS", "6"))
SL_MULTIPLIER = float(os.getenv("SL_MULTIPLIER", "5.0"))  # Default: 5x sweep threshold

# Cache for fair prices
fair_price_cache: Dict = {}
CACHE_TTL = 15  # seconds

# ======================
# Binance Data Fetchers
# ======================

def get_binance_orderbook(symbol: str, limit: int = 20) -> Optional[Dict]:
    try:
        resp = requests.get(
            f"{BINANCE_FUTURES_BASE}/fapi/v1/depth",
            params={"symbol": symbol, "limit": limit},
            timeout=2
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
            "asks": [[float(p), float(q)] for p, q in data.get("asks", [])]
        }
    except Exception:
        return None

def get_binance_recent_trades(symbol: str, limit: int = 100) -> List[Dict]:
    try:
        resp = requests.get(
            f"{BINANCE_FUTURES_BASE}/fapi/v1/trades",
            params={"symbol": symbol, "limit": limit},
            timeout=2
        )
        if resp.status_code != 200:
            return []
        trades = resp.json()
        # Binance returns 'time' in ms, 'price', 'qty'
        return [
            {
                "price": t["price"],
                "qty": t["qty"],
                "time": t["time"]  # ms
            }
            for t in trades
        ]
    except Exception:
        return []

def calc_binance_spread_pct(symbol: str) -> float:
    ob = get_binance_orderbook(symbol, limit=5)
    if not ob or not ob["bids"] or not ob["asks"]:
        return float('inf')
    best_bid = float(ob["bids"][0][0])
    best_ask = float(ob["asks"][0][0])
    if best_bid <= 0 or best_ask <= 0:
        return float('inf')
    mid = (best_bid + best_ask) / 2
    return (best_ask - best_bid) / mid

def has_sufficient_depth_binance(symbol: str, min_usdt_depth: float = 500) -> bool:
    ob = get_binance_orderbook(symbol, limit=5)
    if not ob:
        return False
    bid_depth = sum(float(p) * float(q) for p, q in ob["bids"][:3])
    ask_depth = sum(float(p) * float(q) for p, q in ob["asks"][:3])
    return min(bid_depth, ask_depth) >= min_usdt_depth

# ======================
# Core Logic
# ======================

def calculate_vwap(trades: List[Dict]) -> float:
    total_volume = sum(float(t["qty"]) for t in trades)
    if total_volume == 0:
        return 0.0
    volume_price_sum = sum(float(t["price"]) * float(t["qty"]) for t in trades)
    return volume_price_sum / total_volume

def get_cached_fair_price(symbol: str, trades: List[Dict]) -> float:
    cache_key = f"binance_{symbol}_fair_price"
    now = time.time()
    
    if cache_key in fair_price_cache:
        cached_time, fair_price = fair_price_cache[cache_key]
        if now - cached_time < CACHE_TTL:
            return fair_price
    
    vwap_price = calculate_vwap(trades)
    prices = [float(t["price"]) for t in trades]
    now_ms = time.time() * 1000
    fair_window = [
        p for t, p in zip(trades, prices)
        if now_ms - t["time"] <= 60_000
    ]
    if not fair_window:
        fair_price_cache[cache_key] = (now, vwap_price)
        return vwap_price
        
    median_price = sorted(fair_window)[len(fair_window) // 2]
    fair_price = (vwap_price * 0.7) + (median_price * 0.3)
    fair_price_cache[cache_key] = (now, fair_price)
    return fair_price

def calculate_net_profit(entry_price: float, exit_price: float, side: str) -> float:
    if side == 'buy':
        price_change = (exit_price - entry_price) / entry_price
    else:
        price_change = (entry_price - exit_price) / entry_price
    return price_change - TOTAL_FEES_PCT

def detect_liquidity_sweep_binance(symbol: str, lookback_sec: int = 60) -> Optional[Dict]:
    try:
        trades = get_binance_recent_trades(symbol, limit=200)
        if len(trades) < MIN_TRADES_FOR_ANALYSIS:
            return None

        now_ms = time.time() * 1000
        recent_trades = [t for t in trades if now_ms - t["time"] <= lookback_sec * 1000]
        if len(recent_trades) < 5:
            return None

        fair_price = get_cached_fair_price(symbol, recent_trades)
        if fair_price == 0:
            return None

        ob = get_binance_orderbook(symbol, limit=5)
        if not ob or not ob["bids"] or not ob["asks"]:
            return None

        best_bid = float(ob["bids"][0][0])
        best_ask = float(ob["asks"][0][0])

        spread_pct = calc_binance_spread_pct(symbol)
        if spread_pct > MAX_SPREAD_PCT:
            return None

        if not has_sufficient_depth_binance(symbol, min_usdt_depth=500):
            return None

        # Use last 10 trades
        last_trades = recent_trades[-10:]
        prices = [float(t["price"]) for t in last_trades]
        volumes = [float(t["qty"]) for t in last_trades]

        if not prices:
            return None

        current_price = prices[-1]
        price_displacement = (current_price - fair_price) / fair_price

        is_buy_sweep = price_displacement > SWEEP_THRESHOLD_PCT
        is_sell_sweep = price_displacement < -SWEEP_THRESHOLD_PCT

        if not (is_buy_sweep or is_sell_sweep):
            return None

        # Volume spike check
        avg_vol = sum(volumes[-5:]) / len(volumes[-5:]) if len(volumes) >= 5 else 1
        last_vol = volumes[-1] if volumes else 0
        volume_ratio = last_vol / avg_vol if avg_vol > 0 else 0

        if volume_ratio < VOLUME_SPIKE_MULTIPLIER:
            return None

        # Confidence
        displacement_score = abs(price_displacement) / SWEEP_THRESHOLD_PCT
        confidence = min((displacement_score * 0.6 + volume_ratio * 0.4), 4.0)

        min_sl_pct = 0.001  # 0.1% absolute minimum

        # === BUY SIGNAL ===
        if is_buy_sweep:
            entry = best_ask
            stop_loss = entry * (1 - SWEEP_THRESHOLD_PCT * SL_MULTIPLIER)
            take_profit = entry * (1 + TP_PCT)

            # Enforce min SL distance
            actual_sl_pct = (entry - stop_loss) / entry
            if actual_sl_pct < min_sl_pct:
                stop_loss = entry * (1 - min_sl_pct)

            risk = entry - stop_loss
            reward = take_profit - entry
            risk_reward_ratio = reward / risk if risk > 0 else 0
            net_profit_pct = calculate_net_profit(entry, take_profit, 'buy')

            if risk_reward_ratio < 1.0 or net_profit_pct < 0.0010:
                return None

            signal_data = {
                "symbol": symbol,
                "side": "buy",
                "entry": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence,
                "risk_reward_ratio": round(risk_reward_ratio, 2),
                "net_profit_pct": round(net_profit_pct * 100, 4),
                "sweep_price": current_price,
                "fair_price": fair_price,
                "volume_ratio": volume_ratio,
                "timestamp": time.time(),
                "exchange": "binance"
            }

        # === SELL SIGNAL ===
        elif is_sell_sweep:
            entry = best_bid
            stop_loss = entry * (1 + SWEEP_THRESHOLD_PCT * SL_MULTIPLIER)
            take_profit = entry * (1 - TP_PCT)

            # Enforce min SL distance
            actual_sl_pct = (stop_loss - entry) / entry
            if actual_sl_pct < min_sl_pct:
                stop_loss = entry * (1 + min_sl_pct)

            risk = stop_loss - entry
            reward = entry - take_profit
            risk_reward_ratio = reward / risk if risk > 0 else 0
            net_profit_pct = calculate_net_profit(entry, take_profit, 'sell')

            if risk_reward_ratio < 1.0 or net_profit_pct < 0.0010:
                return None

            signal_data = {
                "symbol": symbol,
                "side": "sell",
                "entry": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence,
                "risk_reward_ratio": round(risk_reward_ratio, 2),
                "net_profit_pct": round(net_profit_pct * 100, 4),
                "sweep_price": current_price,
                "fair_price": fair_price,
                "volume_ratio": volume_ratio,
                "timestamp": time.time(),
                "exchange": "binance"
            }

        else:
            return None

        if signal_data["confidence"] >= MIN_CONFIDENCE:
            return signal_data

        return None

    except Exception as e:
        # Optional: log e for debugging during development
        return None

# ======================
# Scanner Interface
# ======================

def scan_binance_symbols(symbols: List[str], lookback_sec: int = 60) -> List[Dict]:
    signals = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {
            executor.submit(detect_liquidity_sweep_binance, symbol, lookback_sec): symbol
            for symbol in symbols
        }
        for future in as_completed(future_to_symbol):
            try:
                signal = future.result(timeout=8)
                if signal:
                    signals.append(signal)
            except Exception:
                continue
    return sorted(signals, key=lambda x: x["confidence"], reverse=True)