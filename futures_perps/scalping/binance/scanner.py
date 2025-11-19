import os
from typing import Optional, Dict, List
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# Binance Futures API
BINANCE_FUTURES_BASE = "https://fapi.binance.com"

# Fee structure (Binance USDT-M futures) - UPDATED
TAKER_FEE = float(os.getenv("TAKER_FEE", "0.001"))
MAKER_FEE = 0.0002
TOTAL_FEES_PCT = TAKER_FEE + MAKER_FEE

# Load parameters from .env - WIDER TARGETS
TP_NET_TARGET = float(os.getenv("BINANCE_TP_NET_TARGET", "0.0075"))  # 0.75% (was 0.25%)
TP_PCT = TP_NET_TARGET + TOTAL_FEES_PCT
SWEEP_THRESHOLD_PCT = float(os.getenv("BINANCE_SWEEP_THRESHOLD_PCT", "0.0020"))  # 0.20% (was 0.10%)
VOLUME_SPIKE_MULTIPLIER = float(os.getenv("BINANCE_VOLUME_SPIKE_MULTIPLIER", "1.8"))
PULLBACK_CONFIRM_PCT = float(os.getenv("BINANCE_PULLBACK_CONFIRM_PCT", "0.0"))
MAX_SPREAD_PCT = float(os.getenv("BINANCE_MAX_SPREAD_PCT", "0.0005"))  # 0.05% (was 0.1%)
MIN_CONFIDENCE = float(os.getenv("BINANCE_MIN_CONFIDENCE", "1.8"))  # STRONGER SIGNALS (was 1.3)
MIN_TRADES_FOR_ANALYSIS = int(os.getenv("BINANCE_MIN_TRADES_FOR_ANALYSIS", "6"))
SL_MULTIPLIER = float(os.getenv("SL_MULTIPLIER", "3.0"))  # 0.60% SL (was 0.20%)

# Risk management
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))
MAX_LEVERAGE_VERY_STRONG = int(os.getenv("MAX_LEVERAGE_VERY_STRONG", "5"))
MAX_LEVERAGE_STRONG = int(os.getenv("MAX_LEVERAGE_STRONG", "3"))
MAX_LEVERAGE_MODERATE = int(os.getenv("MAX_LEVERAGE_MODERATE", "2"))

# Additional thresholds
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "5.0"))
MIN_TOP5_QUOTE_DEPTH = float(os.getenv("MIN_TOP5_QUOTE_DEPTH", "1000.0"))  # $1000 (was $500)
START_NOTIONAL = float(os.getenv("START_NOTIONAL", "20"))
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT", "0.75"))  # 0.75% (was 0.5%)

# Dynamic hubs configuration
MAX_HUBS = int(os.getenv("MAX_HUBS", "50"))
HUB_REFRESH_SECS = int(os.getenv("HUB_REFRESH_SECS", "300"))
FORCED_HUBS = os.getenv("FORCED_HUBS", "USDT,FDUSD,DAI,BTC,ETH,BNB,SOL,XRP,SUI,TON,LINK,AAVE").split(",")

# Trend filters
TREND_PERIOD_HOURS = int(os.getenv("TREND_PERIOD_HOURS", "4"))  # Check 4h trend
MIN_TREND_STRENGTH = float(os.getenv("MIN_TREND_STRENGTH", "0.002"))  # 0.2% minimum trend

# Additional parameters
MIN_RISK_REWARD_RATIO = float(os.getenv("MIN_RISK_REWARD_RATIO", "2.0"))  # 2:1 (was 1.5)
MIN_NET_PROFIT = float(os.getenv("MIN_NET_PROFIT", "0.0070"))  # 0.70% after fees
MAX_POSITION_SIZE_USDT = float(os.getenv("MAX_POSITION_SIZE_USDT", "50"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.02"))
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "10"))  # Reduced (was 20)

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

def get_binance_klines(symbol: str, interval: str, limit: int = 100) -> List[List]:
    """Fetch OHLCV klines from Binance Futures."""
    try:
        resp = requests.get(
            f"{BINANCE_FUTURES_BASE}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": min(limit, 1000)},
            timeout=3
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        pass
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

def has_sufficient_depth_binance(symbol: str, min_usdt_depth: float = 1000) -> bool:  # $1000 (was $500)
    ob = get_binance_orderbook(symbol, limit=5)
    if not ob:
        return False
    bid_depth = sum(float(p) * float(q) for p, q in ob["bids"][:3])
    ask_depth = sum(float(p) * float(q) for p, q in ob["asks"][:3])
    return min(bid_depth, ask_depth) >= min_usdt_depth

# ======================
# Trend Analysis
# ======================

def get_trend_direction(symbol: str, hours: int = 4) -> float:
    """Calculate trend direction over specified hours"""
    interval = "1h"
    limit = hours
    klines = get_binance_klines(symbol, interval, limit)
    
    if len(klines) < 2:
        return 0.0
    
    # Calculate trend from first to last
    first_close = float(klines[0][4])
    last_close = float(klines[-1][4])
    trend = (last_close - first_close) / first_close
    
    return trend

def is_trend_aligned(signal_side: str, trend_direction: float) -> bool:
    """Check if signal aligns with trend"""
    if trend_direction > MIN_TREND_STRENGTH and signal_side == "buy":
        return True
    elif trend_direction < -MIN_TREND_STRENGTH and signal_side == "sell":
        return True
    return False

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

def detect_liquidity_sweep_binance(symbol: str, lookback_sec: int = 120) -> Optional[Dict]:
    """Now uses momentum trend signals instead of just sweeps"""
    return detect_momentum_trend_signal_binance(symbol, lookback_sec)

# ======================
# Scanner Interface
# ======================
def scan_binance_symbols(symbols: List[str], lookback_sec: int = 120) -> List[Dict]:
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

# ====================================
# Detect_momentum_trend_signal_binance
# ====================================
def detect_momentum_trend_signal_binance(symbol: str, lookback_sec: int = 120) -> Optional[Dict]:
    """Detect momentum signals aligned with trend instead of just sweeps"""
    try:
        trades = get_binance_recent_trades(symbol, limit=300)  # More data points
        if len(trades) < MIN_TRADES_FOR_ANALYSIS:
            return None

        now_ms = time.time() * 1000
        recent_trades = [t for t in trades if now_ms - t["time"] <= lookback_sec * 1000]
        if len(recent_trades) < 10:  # Need more trades
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

        if not has_sufficient_depth_binance(symbol, min_usdt_depth=1500):  # Higher depth
            return None

        # Get trend direction (4-hour trend)
        trend_direction = get_trend_direction(symbol, TREND_PERIOD_HOURS)
        
        # Get recent price action (momentum)
        prices = [float(t["price"]) for t in recent_trades]
        volumes = [float(t["qty"]) for t in recent_trades]
        
        if len(prices) < 20:  # Need sufficient data
            return None

        # Calculate momentum over last 20 trades
        recent_price = prices[-1]
        momentum_start = prices[-20]
        momentum_pct = (recent_price - momentum_start) / momentum_start

        # Volume momentum
        avg_vol = sum(volumes[-10:]) / len(volumes[-10:]) if len(volumes) >= 10 else 1
        current_vol = volumes[-1] if volumes else 0
        volume_ratio = current_vol / avg_vol if avg_vol > 0 else 0

        # Only trade if momentum aligns with trend AND shows acceleration
        if abs(momentum_pct) < 0.0015:  # 0.15% minimum move
            return None

        # Check if momentum is in same direction as trend
        if trend_direction > MIN_TREND_STRENGTH and momentum_pct > 0:  # Bullish trend + bullish momentum
            signal_side = 'buy'
        elif trend_direction < -MIN_TREND_STRENGTH and momentum_pct < 0:  # Bearish trend + bearish momentum
            signal_side = 'sell'
        else:
            return None  # Momentum doesn't align with trend

        # Calculate entry based on momentum acceleration
        if signal_side == 'buy':
            entry = best_ask
            # SL below recent support
            recent_low = min(prices[-15:])
            stop_loss = recent_low * 0.998  # 0.2% below recent low
            
            # TP based on momentum target
            take_profit = entry * (1 + TP_PCT)
            
            # Ensure SL is not too close to entry
            if (entry - stop_loss) / entry < 0.002:  # 0.2% minimum risk
                stop_loss = entry * 0.998

        else:  # sell
            entry = best_bid
            # SL above recent resistance
            recent_high = max(prices[-15:])
            stop_loss = recent_high * 1.002  # 0.2% above recent high
            
            # TP based on momentum target
            take_profit = entry * (1 - TP_PCT)
            
            # Ensure SL is not too close to entry
            if (stop_loss - entry) / entry < 0.002:  # 0.2% minimum risk
                stop_loss = entry * 1.002

        # Calculate risk metrics
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        net_profit_pct = calculate_net_profit(entry, take_profit, signal_side)

        # Confidence calculation based on trend alignment + momentum + volume
        trend_alignment_score = abs(trend_direction) * 10  # Higher trend = higher confidence
        momentum_strength = abs(momentum_pct) / 0.0015  # Normalize momentum
        volume_boost = min(volume_ratio * 0.3, 1.0)  # Volume boost up to 1.0
        
        confidence = 1.5 + trend_alignment_score + momentum_strength + volume_boost
        confidence = min(confidence, 4.0)  # Cap at 4.0

        if risk_reward_ratio < MIN_RISK_REWARD_RATIO or net_profit_pct < MIN_NET_PROFIT:
            return None

        if confidence < MIN_CONFIDENCE:  # 1.8 (was checking sweeps with 1.3)
            return None

        signal_data = {
            "symbol": symbol,
            "side": signal_side,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "risk_reward_ratio": round(risk_reward_ratio, 2),
            "net_profit_pct": round(net_profit_pct * 100, 4),
            "trend_direction": trend_direction,
            "momentum_pct": round(momentum_pct * 100, 4),
            "volume_ratio": volume_ratio,
            "timestamp": time.time(),
            "exchange": "binance"
        }

        return signal_data

    except Exception as e:
        return None