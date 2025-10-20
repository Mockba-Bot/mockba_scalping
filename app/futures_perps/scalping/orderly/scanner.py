import os
from .utils import get_orderbook, get_recent_trades, calc_spread_pct, has_sufficient_depth
from typing import Optional, Dict, List
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# Fee-aware parameters for Apolo DEX
TAKER_FEE = 0.0006  # 0.06%
MAKER_FEE = 0.0004  # 0.04%
TOTAL_FEES_PCT = TAKER_FEE + MAKER_FEE  # 0.1%

# RELAXED PARAMETERS for better signal detection
TP_NET_TARGET = float(os.getenv("TP_NET_TARGET", 0.0015))                    # 0.15% net profit after fees
TP_PCT = TP_NET_TARGET + TOTAL_FEES_PCT                               # 0.35% gross TP
SWEEP_THRESHOLD_PCT = float(os.getenv("SWEEP_THRESHOLD_PCT", 0.0010))        # 0.10% beyond fair price
VOLUME_SPIKE_MULTIPLIER = float(os.getenv("VOLUME_SPIKE_MULTIPLIER", 1.2))   # Volume must be ≥1.2x recent average
PULLBACK_CONFIRM_PCT = float(os.getenv("PULLBACK_CONFIRM_PCT", 0.0006))      # Price must pull back by at least 0.06%
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", 0.0005))                  # 0.05% max allowed spread
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 1.1))                     # Minimum confidence score
MIN_TRADES_FOR_ANALYSIS = int(os.getenv("MIN_TRADES_FOR_ANALYSIS", 8))     # Minimum trades needed for reliable analysis

# Cache for fair prices (thread-safe with timestamps)
fair_price_cache = {}
CACHE_TTL = 15  # seconds

def calculate_vwap(trades: List[Dict]) -> float:
    total_volume = 0
    volume_price_sum = 0
    
    for trade in trades:
        price = float(trade['price'])
        volume = float(trade['qty'])
        volume_price_sum += price * volume
        total_volume += volume
    
    return volume_price_sum / total_volume if total_volume > 0 else 0

def get_cached_fair_price(symbol: str, trades: List[Dict]) -> float:
    cache_key = f"{symbol}_fair_price"
    now = time.time()
    
    if cache_key in fair_price_cache:
        cached_time, fair_price = fair_price_cache[cache_key]
        if now - cached_time < CACHE_TTL:
            return fair_price
    
    vwap_price = calculate_vwap(trades)
    prices = [float(t['price']) for t in trades]
    now_ms = time.time() * 1000
    fair_window = [
        p for t, p in zip(trades, prices)
        if now_ms - t['time'] <= 60_000
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
        net_profit = price_change - TOTAL_FEES_PCT
    else:
        price_change = (entry_price - exit_price) / entry_price  
        net_profit = price_change - TOTAL_FEES_PCT
    
    return net_profit

def detect_liquidity_sweep(symbol: str, lookback_sec: int = 120) -> Optional[Dict]:
    try:
        trades = get_recent_trades(symbol, minutes=lookback_sec // 60 + 1)
        if len(trades) < MIN_TRADES_FOR_ANALYSIS:
            return None

        now_ms = time.time() * 1000
        recent_trades = [
            t for t in trades
            if now_ms - t['time'] <= lookback_sec * 1000
        ]
        if len(recent_trades) < 5:
            return None

        prices = [float(t['price']) for t in recent_trades]
        volumes = [float(t['qty']) for t in recent_trades]

        fair_price = get_cached_fair_price(symbol, recent_trades)
        if fair_price == 0:
            return None

        ob = get_orderbook(symbol, limit=3)
        if not ob.get('bids') or not ob.get('asks') or len(ob['bids']) < 1 or len(ob['asks']) < 1:
            return None

        best_bid = float(ob['bids'][0][0])
        best_ask = float(ob['asks'][0][0])

        spread_pct = calc_spread_pct(symbol)
        if spread_pct > MAX_SPREAD_PCT:
            return None
            
        if not has_sufficient_depth(symbol, min_usdt_depth=500):
            return None

        if len(volumes) >= 5:
            avg_volume = sum(volumes[-5:]) / 5
        else:
            avg_volume = sum(volumes) / len(volumes) if volumes else 0

        recent_trades_to_analyze = recent_trades[-5:]
        signal_data = None

        for trade in reversed(recent_trades_to_analyze):
            trade_price = float(trade['price'])
            trade_volume = float(trade['qty'])
            
            volume_ratio = trade_volume / avg_volume if avg_volume > 0 else 0
            vol_spike = volume_ratio >= VOLUME_SPIKE_MULTIPLIER

            if not vol_spike:
                continue

            if trade_price > fair_price * (1 + SWEEP_THRESHOLD_PCT):
                if best_bid < trade_price * (1 - PULLBACK_CONFIRM_PCT):
                    stop_loss = trade_price * (1 + PULLBACK_CONFIRM_PCT * 1.2)
                    take_profit = best_bid * (1 - TP_PCT)
                    
                    risk = stop_loss - best_bid
                    reward = best_bid - take_profit
                    risk_reward_ratio = reward / risk if risk > 0 else 0
                    
                    net_profit_pct = calculate_net_profit(best_bid, take_profit, 'sell')
                    
                    if risk_reward_ratio >= 1.2 and net_profit_pct >= 0.0010:
                        signal_data = {
                            "symbol": symbol,
                            "side": "sell",
                            "entry": best_bid,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "confidence": min(volume_ratio * 0.7, 4.0),
                            "risk_reward_ratio": round(risk_reward_ratio, 2),
                            "net_profit_pct": round(net_profit_pct * 100, 4),
                            "sweep_price": trade_price,
                            "fair_price": fair_price,
                            "volume_ratio": volume_ratio,
                            "timestamp": time.time()
                        }
                        break

            elif trade_price < fair_price * (1 - SWEEP_THRESHOLD_PCT):
                if best_ask > trade_price * (1 + PULLBACK_CONFIRM_PCT):
                    stop_loss = trade_price * (1 - PULLBACK_CONFIRM_PCT * 1.2)
                    take_profit = best_ask * (1 + TP_PCT)
                    
                    risk = best_ask - stop_loss
                    reward = take_profit - best_ask
                    risk_reward_ratio = reward / risk if risk > 0 else 0
                    
                    net_profit_pct = calculate_net_profit(best_ask, take_profit, 'buy')
                    
                    if risk_reward_ratio >= 1.2 and net_profit_pct >= 0.0010:
                        signal_data = {
                            "symbol": symbol,
                            "side": "buy",
                            "entry": best_ask,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "confidence": min(volume_ratio * 0.7, 4.0),
                            "risk_reward_ratio": round(risk_reward_ratio, 2),
                            "net_profit_pct": round(net_profit_pct * 100, 4),
                            "sweep_price": trade_price,
                            "fair_price": fair_price,
                            "volume_ratio": volume_ratio,
                            "timestamp": time.time()
                        }
                        break

        if signal_data and signal_data['confidence'] >= MIN_CONFIDENCE:
            return signal_data

        return None

    except Exception as e:
        return None

def enhanced_sweep_detection(symbol: str, lookback_sec: int = 120) -> Optional[Dict]:
    basic_signal = detect_liquidity_sweep(symbol, lookback_sec)
    
    if not basic_signal:
        return None

    try:
        ob = get_orderbook(symbol, limit=10)
        bid_volume = sum(float(qty) for _, qty in ob['bids'][:5])
        ask_volume = sum(float(qty) for _, qty in ob['asks'][:5])
        total_volume = bid_volume + ask_volume
        orderbook_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

        if basic_signal['side'] == 'buy':
            if orderbook_imbalance < 0.05:
                return None
            basic_signal['confidence'] *= (1 + min(orderbook_imbalance, 0.2))
                
        elif basic_signal['side'] == 'sell':
            if orderbook_imbalance > -0.05:
                return None
            basic_signal['confidence'] *= (1 + min(abs(orderbook_imbalance), 0.2))

        if basic_signal['confidence'] >= MIN_CONFIDENCE:
            basic_signal['orderbook_imbalance'] = round(orderbook_imbalance, 3)
            return basic_signal

    except Exception as e:
        pass
    
    return None

def scan_multiple_symbols_parallel(symbols: List[str], lookback_sec: int = 120, max_workers: int = 8) -> List[Dict]:
    signals = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(enhanced_sweep_detection, symbol, lookback_sec): symbol 
            for symbol in symbols
        }
        
        for future in as_completed(future_to_symbol):
            try:
                signal = future.result(timeout=10)
                if signal:
                    signals.append(signal)
            except Exception:
                continue
    
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    return signals

def scan_multiple_symbols(symbols: List[str], lookback_sec: int = 120) -> List[Dict]:
    return scan_multiple_symbols_parallel(symbols, lookback_sec, max_workers=8)

def validate_signal_quality(signal: Dict) -> bool:
    checks = [
        signal['confidence'] >= MIN_CONFIDENCE,
        signal['risk_reward_ratio'] >= 1.2,
        signal['volume_ratio'] >= VOLUME_SPIKE_MULTIPLIER,
        signal['net_profit_pct'] >= 0.1,
        abs(signal['entry'] - signal['sweep_price']) / signal['sweep_price'] > 0.0003,
    ]
    
    return all(checks)