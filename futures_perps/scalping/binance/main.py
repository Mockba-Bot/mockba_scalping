import json
import time
from datetime import datetime
from pathlib import Path
import psutil
import os
import sys
import threading
from dotenv import load_dotenv
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Load environment variables
load_dotenv()

# 🔁 Binance-specific modules
from liquidity_ranker import get_top_liquidity_symbols
from scanner import calc_binance_spread_pct, detect_liquidity_sweep_binance
from logs.log_config import binance_trader_logger as binance
from trading_bot.send_bot_message import send_bot_message
from trading_bot.futures_executor_binance import place_futures_order
from db.db_ops import insert_position_with_orders, get_open_positions, update_position_pnl, get_all_signal_statuses, initialize_database_tables

# Binance client for position monitoring
from binance.client import Client as BinanceClient
monitor_client = BinanceClient(
    api_key=os.getenv("BINANCE_API_KEY"),
    api_secret=os.getenv("BINANCE_SECRET_KEY"),
    testnet=False
)

# Lock file to prevent multiple instances
LOCK_FILE = Path("/tmp/scalp_scanner_binance_logs.lock")

DEEP_SEEK_API_KEY = os.getenv("DEEP_SEEK_API_KEY")
BINANCE_FUTURES_BASE = "https://fapi.binance.com"

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

# ======================
# Binance LLM Context Helpers
# ======================

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
        binance.warning(f"Kline fetch failed for {symbol}: {e}")
    return []

def get_binance_funding_rate(symbol: str) -> float:
    """Get current funding rate (e.g., -0.000012 = -0.0012%)"""
    try:
        resp = requests.get(
            f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate",
            params={"symbol": symbol},
            timeout=3
        )
        if resp.status_code == 200:
            data = resp.json()
            if data:
                return float(data[0]["fundingRate"])
    except Exception as e:
        binance.warning(f"Funding rate fetch failed for {symbol}: {e}")
    return 0.0

def get_last_n_candles_binance(symbol: str, interval: str, n: int) -> List[Dict]:
    raw = get_binance_klines(symbol, interval, n)
    candles = []
    for k in raw[-n:]:
        # Binance kline: [open_time, open, high, low, close, volume, ...]
        ts, o, h, l, c, v = k[0], k[1], k[2], k[3], k[4], k[5]
        # Convert timestamp (ms) to timezone-aware UTC datetime
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        candles.append({
            "time": dt.isoformat(),  # Already includes 'Z' or +00:00
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(v)
        })
    return candles

def get_current_spread_bps_binance(symbol: str) -> float:
    spread_pct = calc_binance_spread_pct(symbol)
    return spread_pct * 10_000  # to basis points

def get_funding_rate_binance(symbol: str) -> float:
    return get_binance_funding_rate(symbol)


def update_position_from_binance(position_id: int, db_row: dict):
    """Update position with real fill data from Binance."""
    try:
        # Get actual fill price from entry order
        order_info = monitor_client.futures_get_order(
            symbol=db_row['symbol'],
            orderId=db_row['entry_order_id']
        )
        
        if order_info['status'] == 'FILLED':
            fill_price = float(order_info['avgPrice'])
            
            # Calculate current PnL
            current_price = float(monitor_client.futures_symbol_ticker(symbol=db_row['symbol'])['price'])
            
            # Convert DB Decimal values to float
            notional_usd = float(db_row['notional_usd'])
            side = db_row['side']  # string, safe

            if side == 'BUY':
                pnl_pct = (current_price - fill_price) / fill_price * 100
            else:
                pnl_pct = (fill_price - current_price) / fill_price * 100
                
            pnl_usd = (pnl_pct / 100) * notional_usd  # ✅ float * float
            
            # Update DB with fill price
            update_position_pnl(
                position_id,
                pnl_pct=pnl_pct,
                pnl_usd=pnl_usd,
                fill_price=fill_price
            )
            
            # Check if TP/SL hit
            tp_info = monitor_client.futures_get_order(symbol=db_row['symbol'], orderId=db_row['tp_order_id'])
            sl_info = monitor_client.futures_get_order(symbol=db_row['symbol'], orderId=db_row['sl_order_id'])
            
            if tp_info['status'] == 'FILLED':
                update_position_pnl(position_id, pnl_pct=pnl_pct, pnl_usd=pnl_usd, status='TP')
            elif sl_info['status'] == 'FILLED':
                update_position_pnl(position_id, pnl_pct=pnl_pct, pnl_usd=pnl_usd, status='SL')

            # Return payload for notification purposes
            return {
                'symbol': db_row['symbol'],
                'side': side,
                'fill_price': fill_price,
                'current_price': current_price,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd
            }        
                
    except Exception as e:
        binance.error(f"Error updating position {position_id}: {e}")

def position_monitor_loop():
    """Continuously monitor all open positions."""
    binance.info("🚀 Starting position monitor...")
    
    while True:
        try:
            open_positions = get_open_positions()
            
            if open_positions:
                binance.info(f"Monitoring {len(open_positions)} open positions")
                
                for pos in open_positions:
                    closed_info = update_position_from_binance(pos['id'], pos)
                    if closed_info:
                         # 📢 Send closure alert
                        emoji = "🟢" if closed_info['pnl_usd'] >= 0 else "🔴"
                        message = (
                            f"{emoji} POSITION UPDATE on BINANCE\n"
                            f"Symbol: {closed_info['symbol']}\n"
                            f"Side: {closed_info['side'].upper()}\n"
                            f"Fill Price: {closed_info['fill_price']:.4f}\n"
                            f"Current Price: {closed_info['current_price']:.4f}\n"
                            f"PnL: {closed_info['pnl_pct']:.4f}% | ${closed_info['pnl_usd']:.4f}"
                        )
                        # send bot message to all chats
                        chat_statuses = get_all_signal_statuses()
                        for status in chat_statuses:
                            send_bot_message(status['chat_id'], message)

                    time.sleep(0.1)  # Respect Binance rate limits (10 req/sec)

            time.sleep(60)  # Check every 60 seconds when no positions

        except KeyboardInterrupt:
            binance.info("Position monitor stopped by user")
            break
        except Exception as e:
            binance.error(f"Position monitor error: {e}")
            time.sleep(5)


def consult_deepseek_agent_binance(symbol: str, signal: dict) -> dict | None:
    """Consult DeepSeek as a KILLER 70%+ win rate scalper for Binance signals."""

    try:
        # Fetch context
        candles = get_last_n_candles_binance(symbol, "5m", 10)
        spread_bps = get_current_spread_bps_binance(symbol)
        funding_rate = get_funding_rate_binance(symbol)
        
        # Calculate recent momentum
        recent_closes = [c['close'] for c in candles[-5:]]
        momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100
        
        # Volume analysis
        volumes = [c['volume'] for c in candles]
        avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        current_volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

        candles_str = "\n".join([
            f"- {c['time']}: O={c['open']:.4f} H={c['high']:.4f} L={c['low']:.4f} C={c['close']:.4f} V={c['volume']:.0f}"
            for c in candles[-5:]
        ])

        prompt = f"""You are a KILLER 70%+ win rate futures scalper. You ONLY enter high-probability setups.
        CRITICAL FILTERS for 70%+ success rate:
        - Spread MUST be < 8.0 bps for clean entries/exits
        - Volume should be sufficient (notional > $10), spike preferred but not required
        - Momentum SHOULD align with signal direction (current: {momentum:+.2f}%)
        - Funding rate favorable or neutral (current: {funding_rate:.6f})
        - RR ratio MUST be > 1:1.2 (current: {abs((signal['take_profit'] - signal['entry']) / (signal['entry'] - signal['stop_loss'])):.2f}:1)

        SIGNAL ANALYSIS:
        Symbol: {symbol} | Side: {signal['side'].upper()}
        Entry: {signal['entry']:.4f} | SL: {signal['stop_loss']:.4f} | TP: {signal['take_profit']:.4f}
        Risk: {abs((signal['stop_loss'] - signal['entry']) / signal['entry']) * 100:.3f}%
        Reward: {abs((signal['take_profit'] - signal['entry']) / signal['entry']) * 100:.3f}%
        RR Ratio: {abs((signal['take_profit'] - signal['entry']) / (signal['entry'] - signal['stop_loss'])):.2f}:1
        Volume Ratio: {current_volume_ratio:.1f}x | Spread: {spread_bps:.2f} bps
        Momentum (5-candle): {momentum:+.2f}%
        Funding: {funding_rate:.6f} ({funding_rate * 100:.4f}%)

        Recent Price Action:
        {candles_str}

        DECISION MATRIX:
        ✅ APPROVE if: 
        - Spread < 8.0 bps AND notional > $10 AND RR > 1.2
        - Momentum generally aligns (minor divergence acceptable for VERY STRONG signals)
        - No catastrophic rejection wick in last candle
        ❌ REJECT only for clear red flags (e.g., wide spread + low notional + adverse funding)

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
        binance.warning(f"DeepSeek agent failed for {symbol}: {e}")

    return None

def should_call_deepseek(signal: dict, spread_bps: float, volume_ratio: float, symbol: str, current_price: float, current_volume: float) -> bool:
    """
    Decide whether to consult DeepSeek based on dynamic market conditions.
    Now uses absolute notional instead of volume ratio and relaxes spread.
    """
    confidence = signal['confidence']
    rr_ratio = signal.get('risk_reward_ratio', 0)

    # 🔥 Bypass all pre-filters for VERY STRONG signals (confidence ≥ 3.5)
    if confidence >= 3.5:
        binance.info("🔥 Bypassing pre-filters: VERY STRONG signal (conf ≥ 3.5)")
        return True

    # Basic sanity: must have minimal RR
    if rr_ratio < 1.2:
        binance.info("❌ Pre-filter failed: RR ratio too low (<1.2)")
        return False

    # ✅ Spread: allow up to 8.0 bps for alts (was 5.0)
    if spread_bps >= 8.0:
        binance.info(f"❌ Pre-filter failed: spread too high ({spread_bps:.1f} bps ≥ 8.0)")
        return False

    # ✅ Use absolute notional instead of volume ratio
    current_notional = current_price * current_volume
    min_notional = 10.0  # Require at least $10 notional in last candle

    if current_notional < min_notional:
        binance.info(f"❌ Pre-filter failed: notional too low (${current_notional:.2f} < ${min_notional})")
        return False

    # Optional: still allow very low volume_ratio if notional is OK
    # (e.g., quiet market but sufficient liquidity)
    binance.info(
        f"✅ Pre-filter passed | Conf={confidence:.2f}, Spread={spread_bps:.1f}bps, "
        f"Notional=${current_notional:.2f}, RR={rr_ratio:.2f}"
    )
    return True

def scan_for_scalp_opportunities_binance():
    """Run one full Binance scan cycle and return valid signals."""
    start_time = time.time()
    binance.info("🔍 Starting Binance scalp candidate scan...")

    try:
        # Get high-liquidity symbols (top 20–30)
        liquid_symbols = set(get_top_liquidity_symbols(top_n=30))

        candidates = sorted(liquid_symbols)

        binance.info(f"🎯 Scanning {len(candidates)}")

        signals = []
        for symbol in candidates:
            try:
                signal = detect_liquidity_sweep_binance(symbol)
                if not signal:
                    continue

                # Skip stale signals (>2 seconds old)
                if time.time() - signal['timestamp'] > 2.0:
                    continue

                confidence_level = get_confidence_level(signal['confidence'])
                signals.append(signal)

                log_msg = (
                    f"✅ BINANCE Signal: {symbol} | {signal['side'].upper()} | "
                    f"Entry: {signal['entry']:.4f} | SL: {signal['stop_loss']:.4f} | "
                    f"TP: {signal['take_profit']:.4f} | Conf: {signal['confidence']:.2f} | {confidence_level}"
                )
                binance.info(log_msg)

                # Send Telegram alert (only for STRONG+)
                if signal['confidence'] >= 1.8:
                    binance.info(f"🎯 HIGH-CONFIDENCE SIGNAL: {symbol} {signal['confidence']:.2f} - Checking pre-filters...")
                    chat_statuses = get_all_signal_statuses()
                    enabled_chats = [e['chat_id'] for e in chat_statuses if e['signals_enabled']]
                    # binance.info(f"📱 Chat statuses: {len(chat_statuses)} total, {len(enabled_chats)} enabled")
                    if not enabled_chats:
                        continue

                    # 🔍 Get market data for pre-filtering
                    candles = get_last_n_candles_binance(symbol, "5m", 10)
                    spread_bps = get_current_spread_bps_binance(symbol)

                    binance.info(f"📈 Market data: {len(candles)} candles, spread={spread_bps:.1f}bps")
                    
                    # Calculate current volume ratio for pre-filter
                    volumes = [c['volume'] for c in candles]
                    avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
                    current_volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

                    binance.info(f"🔍 Volume analysis: current={volumes[-1]:.0f}, avg={avg_volume:.0f}, ratio={current_volume_ratio:.1f}x")

                    # 🔍 Consult DeepSeek agent (only for borderline or high-value signals)
                    llm_approved = True
                    final_confidence = signal['confidence']

                    binance.info(f"🔍 DeepSeek pre-filter check: Confidence={signal['confidence']:.2f}, Spread={spread_bps:.1f}bps, VolRatio={current_volume_ratio:.1f}x")
                    

                    # 🎯 APPLY PRE-FILTER HERE - ONLY ADDITION
                    current_price = float(candles[-1]['close'])
                    current_volume = float(candles[-1]['volume'])

                    binance.info(f"🔍 DeepSeek pre-filter check: Conf={signal['confidence']:.2f}, "
                                f"Spread={spread_bps:.1f}bps, VolRatio={current_volume_ratio:.1f}x, "
                                f"Notional=${current_price * current_volume:.2f}")

                    if should_call_deepseek(signal, spread_bps, current_volume_ratio, symbol, current_price, current_volume):
                        binance.info(f"🎯 Calling DeepSeek for {symbol} - all pre-filters passed")
                        llm_verdict = consult_deepseek_agent_binance(symbol, signal)
                        if llm_verdict:
                            llm_approved = llm_verdict.get('approve', True)
                            boost = float(llm_verdict.get('confidence_boost', 0.0))
                            reason = llm_verdict.get('reason', 'No reason')
                            final_confidence = max(0.0, signal['confidence'] + boost)
                            binance.info(f"🧠 DeepSeek for {symbol}: {'✅' if llm_approved else '❌'} | {reason} | Boost: {boost:+.2f}")
                        else:
                            binance.info(f"⏭️ DeepSeek skipped (fallback to original signal)")
                    else:
                        llm_verdict = None
                        llm_approved = False
                        binance.info(f"⏭️ DeepSeek skipped due to pre-filters | Spread: {spread_bps:.1f}bps, Vol: {current_volume_ratio:.1f}x")

                    # Final gate
                    if llm_approved and final_confidence >= 1.8:

                        # Attempt to place a futures order; the executor returns a dict on success or None/False on failure
                        order_result = place_futures_order(signal)
                        if order_result:
                            for chat_id in enabled_chats:
                                confirmation_msg = (
                                    f"🚨 BINANCE - Scalp Signal Detected!\n"
                                    f"✅ POSITION OPENED\n"
                                    f"Symbol: {symbol}\n"
                                    f"Side: {signal['side'].upper()}\n"
                                    f"Qty: {order_result['quantity']:.6f}\n"
                                    f"Entry: {signal['entry']:.4f}\n"
                                    f"TP: {signal['take_profit']:.4f} (MARKET)\n"
                                    f"SL: {signal['stop_loss']:.4f} (MARKET)\n"
                                    f"Confidence: {final_confidence:.2f} ({get_confidence_level(final_confidence)})\n"
                                    f"⚠️ Auto-closing on TP/SL hit"
                                )
                                send_bot_message(chat_id, confirmation_msg)
                                insert_position_with_orders(
                                    chat_id=int(chat_id),
                                    signal=signal,
                                    order_result=order_result,
                                    exchange="BINANCE"
                                )
                        else:
                            binance.warning(f"⚠️ Failed to place order for {symbol}")
                    else:
                        binance.info(f"⏭️ Signal for {symbol} DeepSeek skipped.")          

            except Exception as e:
                binance.warning(f"⚠️ Error scanning {symbol} on Binance: {e}")
                continue

        elapsed = time.time() - start_time
        binance.info(f"✅ Binance scan completed in {elapsed:.2f}s | {len(signals)} signal(s) found.")
        return signals

    except Exception as e:
        binance.error(f"💥 Binance scan cycle failed: {e}", exc_info=True)
        return []

def get_market_volatility_interval():
    """Smart interval based on market conditions for Binance"""   
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()
    
    # Extended high-volatility sessions for crypto
    market_hours = {
        "high_volatility": [0, 1, 2, 8, 9, 13, 14, 15, 20, 21, 22],  # Added crypto prime hours
        "normal": [3, 4, 5, 10, 11, 12, 16, 17, 18, 19, 23],
        "low_volatility": [6, 7]  # Early Asia dead zone
    }
    
    # Weekend detection - crypto is more active on weekends
    if weekday >= 5:
        return 30  # Slightly more frequent on weekends for crypto
    
    if hour in market_hours["high_volatility"]:
        return 8
    elif hour in market_hours["normal"]:
        return 15
    else:
        return 25

def main_loop_binance():
    """Main loop with adaptive intervals for Binance"""
    binance.info("🚀 Binance adaptive scalp signal monitor started.")
    
    while True:
        cycle_start = time.time()
        
        # Get adaptive interval
        interval = get_market_volatility_interval()
        
        # Check for existing lock
        if LOCK_FILE.exists():
            try:
                pid = int(LOCK_FILE.read_text().strip())
                if psutil.pid_exists(pid):
                    binance.info(f"🔒 Binance scanner already running (PID {pid}). Sleeping {interval}s.")
                    time.sleep(interval)
                    continue
                else:
                    binance.warning(f"⚠️ Stale lock detected (PID {pid} not running). Removing...")
                    LOCK_FILE.unlink()
            except Exception as e:
                binance.error(f"⚠️ Error reading lock file. Removing lock: {e}")
                LOCK_FILE.unlink()

        binance.info(f"⏱️ Starting Binance scan cycle at {datetime.now().isoformat()} | Interval: {interval}s")

        # Acquire lock
        try:
            LOCK_FILE.write_text(str(os.getpid()))
            
            # Run scan
            scan_for_scalp_opportunities_binance()

        except Exception as e:
            binance.error(f"🚨 Unexpected error in Binance scan cycle: {e}")

        finally:
            # Always release lock
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()

        # Adaptive sleep based on market conditions
        cycle_time = time.time() - cycle_start
        remaining = max(1, interval - cycle_time)
        
        binance.info(f"📊 BINANCE CYCLE: {cycle_time:.2f}s | Market Interval: {interval}s | Sleeping {remaining:.1f}s")
        time.sleep(remaining)


if __name__ == "__main__":
    # 1. Initialize DB schema
    initialize_database_tables()

    # Start position monitor in background thread
    monitor_thread = threading.Thread(
        target=position_monitor_loop,
        daemon=True,
        name="PositionMonitor"
    )
    monitor_thread.start()
    binance.info("✅ Position monitor started in background")
    
    # Start main scanner loop
    main_loop_binance()