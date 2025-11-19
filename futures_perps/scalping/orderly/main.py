from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import psutil
import os
import sys
from datetime import datetime, timezone

import requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from futures_perps.scalping.orderly.utils import calc_spread_pct, get_funding_rate, get_klines
from liquidity_ranker import get_top_liquidity_symbols
from scanner import detect_liquidity_sweep
from logs.log_config import trader_logger as orderly
from trading_bot.send_bot_message import send_bot_message

# === GLOBAL CACHE FOR SYMBOLS (updated every 60s) ===
TOP_SYMBOLS_CACHE = []
LAST_RANK_TIME = 0
CACHE_LOCK = threading.Lock()

def get_cached_top_symbols(top_n: int = 60) -> List[str]:
    global TOP_SYMBOLS_CACHE, LAST_RANK_TIME
    now = time.time()
    
    with CACHE_LOCK:
        if now - LAST_RANK_TIME > 60:  # Refresh every 60 seconds
            try:
                TOP_SYMBOLS_CACHE = get_top_liquidity_symbols(top_n=top_n)
                LAST_RANK_TIME = now
                orderly.info(f"🔄 Refreshed top {top_n} symbols: {len(TOP_SYMBOLS_CACHE)} candidates")
            except Exception as e:
                orderly.error(f"Failed to refresh symbols: {e}")
                # Keep old cache if refresh fails
        return TOP_SYMBOLS_CACHE.copy()
    
# Lock file to prevent multiple instances
LOCK_FILE = Path("/tmp/scalp_scanner_orderly.lock")
DEEP_SEEK_API_KEY = os.getenv("DEEP_SEEK_API_KEY")

def get_confidence_level(confidence: float) -> str:
    """Map confidence score to human-readable level"""
    if confidence >= 2.5:
        return "🚀 VERY STRONG"
    elif confidence >= 1.8:
        return "💪 STRONG"
    elif confidence >= 1.3:
        return "👍 MODERATE"
    elif confidence >= 1.0:
        return "⚠️ WEAK"
    else:
        return "❌ VERY WEAK"
    
# --- HELPERS --- #
def get_last_n_candles(symbol: str, interval: str, n: int) -> List[Dict]:
    raw_klines = get_klines(symbol, interval, n)
    
    if not isinstance(raw_klines, list):
        orderly.warning(f"Invalid klines format for {symbol}: expected list, got {type(raw_klines)}")
        return []

    candles = []
    for k in raw_klines[-n:]:
        if not isinstance(k, dict):
            orderly.warning(f"Kline is not a dict for {symbol}: {k}")
            continue

        try:
            # Orderly provides timestamps in ms
            ts = k['start_timestamp']  # or 'end_timestamp' — but start is standard for open time
            o = k['open']
            h = k['high']
            l = k['low']
            c = k['close']
            v = k['volume']

            candles.append({
                "time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v)
            })
        except KeyError as e:
            orderly.warning(f"Missing key in kline for {symbol}: {e} | Kline: {k}")
            continue
        except (ValueError, TypeError) as e:
            orderly.warning(f"Invalid value in kline for {symbol}: {e} | Kline: {k}")
            continue

    return candles

def get_current_spread_bps(symbol: str) -> float:
    return calc_spread_pct(symbol) * 10_000

def get_orderly_funding_rate(symbol: str) -> float:
    return get_funding_rate(symbol)

# --- END HELPERS --- #
    

def consult_deepseek_agent(symbol: str, signal: dict, candles_5m: list, spread_bps: float, funding_rate: float) -> dict | None:
    """
    Consult DeepSeek as a KILLER 70%+ win rate scalper for Orderly signals.
    Returns: {'approve': bool, 'reason': str, 'confidence_boost': float}
    """
    try:
        # Calculate recent momentum
        recent_closes = [c['close'] for c in candles_5m[-5:]]
        momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100 if len(recent_closes) >= 2 else 0
        
        # Volume analysis
        volumes = [c['volume'] for c in candles_5m]
        avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        current_volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
        
        # Calculate RR ratio
        if signal['side'].upper() == 'LONG':
            rr_ratio = abs((signal['take_profit'] - signal['entry']) / (signal['entry'] - signal['stop_loss']))
        else:
            rr_ratio = abs((signal['entry'] - signal['take_profit']) / (signal['stop_loss'] - signal['entry']))

        candles_str = "\n".join([
            f"- {c['time'].strftime('%H:%M')}: O={c['open']:.4f} H={c['high']:.4f} L={c['low']:.4f} C={c['close']:.4f} V={c['volume']:.0f}"
            for c in candles_5m[-5:]
        ])

        prompt = f"""You are a scalper on Orderly Network (DEX) - VOLUME IS NATURALLY LOWER than CEX.
        DEX REALITY CHECK:
        - Volume is 50-80% lower than Binance normally
        - Spreads are typically 2-8bps (vs 1-3bps on CEX)
        - Focus on RELATIVE volume spikes rather than absolute amounts
        
        ADJUSTED FILTERS for DEX environment:
        - Spread ACCEPTABLE if < 12.0 bps (current: {spread_bps:.2f} bps)
        - Volume spike GOOD if > 0.8x average (current: {current_volume_ratio:.1f}x)
        - Momentum should align with direction (current: {momentum:+.2f}%)
        - RR ratio should be > 1:1.3 (current: {rr_ratio:.2f}:1)

        SIGNAL ANALYSIS (DEX CONTEXT):
        Symbol: {symbol} | Side: {signal['side'].upper()}
        Entry: {signal['entry']:.4f} | SL: {signal['stop_loss']:.4f} | TP: {signal['take_profit']:.4f}
        Volume Ratio: {current_volume_ratio:.1f}x (DEX: >0.5x is acceptable, >1.2x is good)
        Spread: {spread_bps:.2f} bps (DEX: <6.0bps is acceptable)
        RR Ratio: {rr_ratio:.2f}:1

        Recent Price Action:
        {candles_str}

        DEX-SPECIFIC DECISION:
        ✅ APPROVE if: 
           - Spread < 6.0 bps AND Volume > 0.8x 
           - Momentum confirms direction
           - Clean candle structure
        ✅ WEAK APPROVE if spread 6-8bps but other factors strong
        ❌ REJECT only if: spread > 8bps OR volume < 0.5x OR bad RR

        Respond with RAW JSON only:
        {{
        "approve": true/false,
        "reason": "≤10 words. DEX-aware.",
        "confidence_boost": -0.5 to +0.5
        }}"""

        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEP_SEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-coder",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,  # Slight creativity for edge cases
                "max_tokens": 150
            },
            timeout=3
        )

        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content'].strip()
            # Clean JSON response
            if content.startswith("```json"):
                content = content[7:].split("```")[0]
            elif content.startswith("```"):
                content = content[3:].split("```")[0]
            return json.loads(content)

    except Exception as e:
        orderly.warning(f"DeepSeek agent failed for {symbol}: {e}")

    return None

def should_call_deepseek(signal: dict, spread_bps: float, volume_ratio: float) -> bool:
    """DEX-optimized pre-filter for Orderly — adjusted for real market conditions"""
    orderly.info(f"🧪 Pre-filter evaluation: conf={signal['confidence']:.2f}, spread={spread_bps:.1f} bps, vol_ratio={volume_ratio:.1f}x")
    
    # ✅ 1. Keep confidence threshold low (good)
    if signal['confidence'] < 1.2:  # even lower — sweeps are rare
        orderly.info("❌ Pre-filter failed: confidence < 1.2")
        return False

    # ✅ 2. Relax spread tolerance — 12 bps is normal on Orderly for mid-caps
    if spread_bps >= 12.0:  # ← increased from 8.0
        orderly.info("❌ Pre-filter failed: spread >= 12.0 bps")
        return False

    # ✅ 3. Volume ratio: allow lower activity (0.3x is meaningful)
    if volume_ratio < 0.3:  # ← lowered from 0.5
        orderly.info("❌ Pre-filter failed: volume ratio < 0.3x")
        return False

    # ✅ 4. RR ratio: keep at 1.2 (reasonable)
    if signal.get('risk_reward_ratio', 0) < 1.2:
        orderly.info("❌ Pre-filter failed: RR < 1.2")
        return False

    orderly.info("✅ Pre-filter passed — eligible for DeepSeek")
    return True

def process_signal(symbol: str, signal: dict):
    """Process a single signal (log + alert)"""
    try:
        confidence_level = get_confidence_level(signal['confidence'])
        log_msg = (
            f"✅ ORDERLY Signal: {symbol} | {signal['side'].upper()} | "
            f"Entry: {signal['entry']:.4f} | SL: {signal['stop_loss']:.4f} | "
            f"TP: {signal['take_profit']:.4f} | Conf: {signal['confidence']:.2f} | {confidence_level}"
        )
        orderly.info(log_msg)

        # 🟢 FAST PATH: High-confidence signals bypass DeepSeek
        if signal['confidence'] >= 1.8:
            enabled_chats = ["556159355"]
            if not enabled_chats:
                orderly.info("⏭️ No enabled chats - skipping high-conf signal")
                return

            # Basic viability check (no LLM)
            candles_5m = get_last_n_candles(symbol, '5m', n=10)
            if len(candles_5m) < 2:
                orderly.info(f"⏭️ Not enough candles for {symbol}")
                return

            spread_bps = get_current_spread_bps(symbol)
            volumes = [c['volume'] for c in candles_5m]
            current_volume = volumes[-1]

            if spread_bps <= 15.0 and current_volume >= 200:
                message = (
                    f"🚨 ORDERLY - Scalp Signal Detected!\n"
                    f"Symbol: {symbol}\n"
                    f"Side: {signal['side'].upper()}\n"
                    f"Entry: {signal['entry']:.4f}\n"
                    f"Stop Loss: {signal['stop_loss']:.4f}\n"
                    f"Take Profit: {signal['take_profit']:.4f}\n"
                    f"Confidence: {signal['confidence']:.2f} - {get_confidence_level(signal['confidence'])}\n"
                    f"RR: {signal['risk_reward_ratio']:.2f} | Vol: ${current_volume:,.0f}\n"
                    f"🎯 HIGH-CONFIDENCE LIQUIDITY SWEEP (No LLM delay)"
                )
                send_bot_message("556159355", message)
                orderly.info(f"✅ HIGH-CONFIDENCE ALERT SENT for {symbol} | Spread: {spread_bps:.1f}bps | Vol: ${current_volume:,.0f}")
            else:
                orderly.info(f"⏭️ High-conf signal skipped: spread={spread_bps:.1f}bps or vol=${current_volume:,.0f} too low")
            return

        # 🟡 MEDIUM CONFIDENCE: Use DeepSeek for validation/boost
        candles_5m = get_last_n_candles(symbol, '5m', n=10)
        if len(candles_5m) < 2:
            return

        spread_bps = get_current_spread_bps(symbol)
        volumes = [c['volume'] for c in candles_5m]
        avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        current_volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

        orderly.info(f"🔍 Medium-conf signal: {symbol} | Spread: {spread_bps:.1f}bps | VolRatio: {current_volume_ratio:.1f}x")

        if should_call_deepseek(signal, spread_bps, current_volume_ratio):
            orderly.info(f"🎯 Calling DeepSeek for {symbol}")
            llm_verdict = consult_deepseek_agent(
                symbol=symbol,
                signal=signal,
                candles_5m=candles_5m,
                spread_bps=spread_bps,
                funding_rate=get_orderly_funding_rate(symbol)
            )

            if llm_verdict:
                approve = llm_verdict.get('approve', True)
                boost = float(llm_verdict.get('confidence_boost', 0.0))
                reason = llm_verdict.get('reason', 'No reason')
                final_confidence = max(0.0, signal['confidence'] + boost)
                orderly.info(f"🧠 DeepSeek for {symbol}: {'✅' if approve else '❌'} | {reason} | Boost: {boost:+.2f}")

                if approve and final_confidence >= 1.8:
                    current_volume = volumes[-1]
                    message = (
                        f"🚨 ORDERLY - Scalp Signal Detected!\n"
                        f"Symbol: {symbol}\n"
                        f"Side: {signal['side'].upper()}\n"
                        f"Entry: {signal['entry']:.4f}\n"
                        f"Stop Loss: {signal['stop_loss']:.4f}\n"
                        f"Take Profit: {signal['take_profit']:.4f}\n"
                        f"Confidence: {final_confidence:.2f} - {get_confidence_level(final_confidence)}\n"
                        f"RR: {signal['risk_reward_ratio']:.2f} | Vol: ${current_volume:,.0f}\n"
                        f"🎯 LLM-BOOSTED SETUP"
                    )
                    send_bot_message("556159355", message)
                    orderly.info(f"✅ LLM-BOOSTED ALERT SENT for {symbol}")
                else:
                    orderly.info(f"⏭️ LLM rejected or final confidence < 1.8 for {symbol}")
            else:
                orderly.info(f"⏭️ DeepSeek API call failed for {symbol}")
        else:
            orderly.info(f"⏭️ Medium-conf signal skipped (pre-filter failed) for {symbol}")

    except Exception as e:
        orderly.warning(f"⚠️ Error processing signal for {symbol}: {e}")

def scan_symbol(symbol: str) -> dict | None:
    """Scan a single symbol and return valid signal"""
    try:
        signal = detect_liquidity_sweep(symbol)
        
        if not signal:
            return None
            
        # Skip stale signals (>2 seconds old)
        if time.time() - signal['timestamp'] > 2.0:
            return None
            
        return signal
    except Exception as e:
        orderly.warning(f"⚠️ Error scanning {symbol}: {e}")
        return None

def scan_for_scalp_opportunities():
    """Run one full scan cycle with parallel execution"""
    start_time = time.time()
    orderly.info("🔍 Starting Orderly scalp candidate scan...")

    try:
        # Get cached top symbols (only top 60 for speed)
        candidates = get_cached_top_symbols(top_n=60)
        if not candidates:
            orderly.warning("⚠️ No candidates available - skipping scan")
            return []

        orderly.info(f"🎯 Scanning {len(candidates)} candidate(s)")

        # Parallel scan with 15 workers (adjust based on your CPU)
        signals = []
        with ThreadPoolExecutor(max_workers=15) as executor:
            future_to_symbol = {
                executor.submit(scan_symbol, symbol): symbol 
                for symbol in candidates
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    signal = future.result(timeout=8)  # 8s timeout per symbol
                    if signal:
                        signals.append(signal)
                        # Process immediately to reduce latency
                        process_signal(symbol, signal)
                except Exception as e:
                    orderly.warning(f"⚠️ Scan timeout/error for {symbol}: {e}")

        elapsed = time.time() - start_time
        orderly.info(f"✅ Orderly scan completed in {elapsed:.2f}s | {len(signals)} signal(s) found.")
        return signals

    except Exception as e:
        orderly.error(f"💥 Orderly scan cycle failed: {e}", exc_info=True)
        return []
        
def get_market_volatility_interval_orderly():
    """Smart interval based on market conditions for Orderly"""    
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()
    
    # Market sessions (UTC) - Orderly might have different patterns
    market_hours = {
        "high_volatility": [0, 1, 2, 8, 9, 14, 15, 20, 21],  # Major overlaps
        "normal": [3, 4, 5, 10, 11, 12, 13, 16, 17, 18, 19, 22, 23],
        "low_volatility": [6, 7]  # Dead periods
    }
    
    # Weekend detection
    if weekday >= 5:
        return 60  # Even more reduced on weekends for Orderly
    
    # Determine volatility level
    if hour in market_hours["high_volatility"]:
        return 10  # Slightly more conservative than Binance
    elif hour in market_hours["normal"]:
        return 20
    else:  # low_volatility
        return 30
    
def main_loop_orderly():
    """Main loop with adaptive intervals for Orderly"""
    orderly.info("🚀 Orderly adaptive scalp signal monitor started.")
    
    while True:
        cycle_start = time.time()
        
        # Get adaptive interval (now more aggressive)
        interval = get_market_volatility_interval_orderly()
        
        # Check for existing lock
        if LOCK_FILE.exists():
            try:
                pid = int(LOCK_FILE.read_text().strip())
                if psutil.pid_exists(pid):
                    orderly.info(f"🔒 Orderly scanner already running (PID {pid}). Sleeping {interval}s.")
                    time.sleep(interval)
                    continue
                else:
                    orderly.warning(f"⚠️ Stale lock detected (PID {pid} not running). Removing...")
                    LOCK_FILE.unlink()
            except Exception as e:
                orderly.error(f"⚠️ Error reading lock file. Removing lock: {e}")
                LOCK_FILE.unlink()

        orderly.info(f"⏱️ Starting Orderly scan cycle at {datetime.now().isoformat()} | Interval: {interval}s")

        # Acquire lock
        try:
            LOCK_FILE.write_text(str(os.getpid()))
            
            # Run scan
            scan_for_scalp_opportunities()

        except Exception as e:
            orderly.error(f"🚨 Unexpected error in Orderly scan cycle: {e}")

        finally:
            # Always release lock
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()

        # Adaptive sleep
        cycle_time = time.time() - cycle_start
        remaining = max(1, interval - cycle_time)
        
        orderly.info(f"📊 ORDERLY CYCLE: {cycle_time:.2f}s | Market Interval: {interval}s | Sleeping {remaining:.1f}s")
        time.sleep(remaining)

if __name__ == "__main__":
    import threading  # Add this import at top if not already present
    main_loop_orderly()