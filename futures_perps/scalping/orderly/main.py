import json
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
from gainers_losers import get_top_gainers_losers
from scanner import detect_liquidity_sweep
from logs.log_config import trader_logger as orderly
from trading_bot.send_bot_message import send_bot_message

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
    candles = []
    for k in raw_klines[-n:]:
        ts, o, h, l, c, v = k
        candles.append({
            "time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(v)
        })
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

        prompt = f"""You are a KILLER 70%+ win rate futures scalper. You ONLY enter high-probability setups.
        CRITICAL FILTERS for 70%+ success rate:
        - Spread MUST be < 3.0 bps for clean entries/exits (current: {spread_bps:.2f} bps)
        - Volume spike MUST be > 2.5x average (current: {current_volume_ratio:.1f}x)
        - Momentum MUST align with signal direction (current: {momentum:+.2f}%)
        - Funding rate favorable or neutral (current: {funding_rate:.6f})
        - RR ratio MUST be > 1:1.5 (current: {rr_ratio:.2f}:1)
        - No major wicks against position in last 2 candles

        SIGNAL ANALYSIS:
        Symbol: {symbol} | Side: {signal['side'].upper()}
        Entry: {signal['entry']:.4f} | SL: {signal['stop_loss']:.4f} | TP: {signal['take_profit']:.4f}
        Risk: {abs((signal['stop_loss'] - signal['entry']) / signal['entry']) * 100:.3f}%
        Reward: {abs((signal['take_profit'] - signal['entry']) / signal['entry']) * 100:.3f}%
        RR Ratio: {rr_ratio:.2f}:1
        Volume Ratio: {current_volume_ratio:.1f}x | Spread: {spread_bps:.2f} bps
        Momentum (5-candle): {momentum:+.2f}%
        Funding: {funding_rate:.6f} ({funding_rate * 100:.4f}%)

        Recent Price Action:
        {candles_str}

        DECISION MATRIX:
        ✅ APPROVE if ALL conditions met: 
           - Spread < 3.0 bps AND Volume > 2.5x AND RR > 1.5
           - Momentum confirms direction (positive for LONG, negative for SHORT)  
           - Clean candle structure with minimal opposition wicks
        ❌ REJECT if any red flags: poor RR, weak volume, bad spread, counter momentum

        Respond with RAW JSON only:
        {{
        "approve": true/false,
        "reason": "≤10 words. Be brutal.",
        "confidence_boost": -0.5 to +0.5 (adjust based on filter compliance)
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
    """Pre-filter to avoid unnecessary DeepSeek calls"""
    if (signal['confidence'] < 1.8 or 
        spread_bps >= 3.0 or 
        volume_ratio < 2.0 or
        signal.get('risk_reward_ratio', 0) < 1.3):
        return False
    return True

def scan_for_scalp_opportunities():
    """Run one full scan cycle and return valid signals."""
    start_time = time.time()
    orderly.info("🔍 Starting Orderly scalp candidate scan...")

    try:
        # Get high-liquidity symbols (top 20–30)
        liquid_symbols = set(get_top_liquidity_symbols(top_n=25))

        # Get movers from multiple timeframes
        gainers_5m, losers_5m = get_top_gainers_losers(interval_minutes=5)
        gainers_15m, losers_15m = get_top_gainers_losers(interval_minutes=15)

        # Combine all movers
        all_movers = set(gainers_5m + losers_5m + gainers_15m + losers_15m)

        # Prioritize: liquid + moving
        high_priority = liquid_symbols & all_movers
        candidates = sorted(high_priority or liquid_symbols)

        orderly.info(f"🎯 Scanning {len(candidates)} Orderly candidates: {candidates}")
        orderly.debug(f"📈 Gainers: {gainers_5m} {gainers_15m}")
        orderly.debug(f"📉 Losers: {losers_5m} {losers_15m}")


        signals = []
        for symbol in candidates:
            try:
                signal = detect_liquidity_sweep(symbol)
                if not signal:
                    continue

                # Skip stale signals (>2 seconds old)
                if time.time() - signal['timestamp'] > 2.0:
                    continue

                confidence_level = get_confidence_level(signal['confidence'])
                signals.append(signal)

                log_msg = (
                    f"✅ ORDERLY Signal: {symbol} | {signal['side'].upper()} | "
                    f"Entry: {signal['entry']:.4f} | SL: {signal['stop_loss']:.4f} | "
                    f"TP: {signal['take_profit']:.4f} | Conf: {signal['confidence']:.2f} | {confidence_level}"
                )
                orderly.info(log_msg)

                # Send Telegram alert (only for MODERATE+)
                if signal['confidence'] >= 1.8:  # Increased from 1.3 for 70%+ win rate
                    # Gather contextual data
                    candles_5m = get_last_n_candles(symbol, '5m', n=10)
                    spread_bps = get_current_spread_bps(symbol)
                    funding_rate = get_orderly_funding_rate(symbol)
                    
                    # Calculate current volume ratio for pre-filter
                    volumes = [c['volume'] for c in candles_5m]
                    avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
                    current_volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

                    # 🔍 Consult DeepSeek killer scalper agent
                    approve = True
                    final_confidence = signal['confidence']

                    # 🎯 APPLY PRE-FILTER HERE - ONLY ADDITION
                    if should_call_deepseek(signal, spread_bps, current_volume_ratio):
                        llm_verdict = consult_deepseek_agent(
                            symbol=symbol,
                            signal=signal,
                            candles_5m=candles_5m,
                            spread_bps=spread_bps,
                            funding_rate=funding_rate
                        )

                        if llm_verdict:
                            approve = llm_verdict.get('approve', True)
                            boost = float(llm_verdict.get('confidence_boost', 0.0))
                            reason = llm_verdict.get('reason', 'No reason')
                            final_confidence = max(0.0, signal['confidence'] + boost)
                            orderly.info(f"🧠 DeepSeek for {symbol}: {'✅' if approve else '❌'} | {reason} | Boost: {boost:+.2f}")
                        else:
                            orderly.info(f"⏭️ DeepSeek skipped (fallback to original signal)")
                    else:
                        llm_verdict = None
                        orderly.info(f"⏭️ DeepSeek skipped due to pre-filters | Spread: {spread_bps:.1f}bps, Vol: {current_volume_ratio:.1f}x")

                    # Final gate: must be approved AND still ≥1.8 after boost
                    if approve and final_confidence >= 1.8:
                        message = (
                            f"🚨 ORDERLY - KILLER Scalp Signal!\n"
                            f"Symbol: {symbol}\n"
                            f"Side: {signal['side'].upper()}\n"
                            f"Entry: {signal['entry']:.4f}\n"
                            f"Stop Loss: {signal['stop_loss']:.4f}\n"
                            f"Take Profit: {signal['take_profit']:.4f}\n"
                            f"Confidence: {final_confidence:.2f} - {get_confidence_level(final_confidence)}\n"
                            f"RR: {signal['risk_reward_ratio']:.2f} | Vol Spike: {signal['volume_ratio']:.1f}x\n"
                            f"🎯 70%+ WIN RATE SETUP"
                        )
                        chat_id = "556159355"
                        send_bot_message(chat_id, message)
                    else:
                        orderly.info(f"⏭️ Signal for {symbol} rejected after LLM review.")

            except Exception as e:
                orderly.warning(f"⚠️ Error scanning {symbol} on Orderly: {e}")
                continue

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
        
        # Get adaptive interval
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
    main_loop_orderly()