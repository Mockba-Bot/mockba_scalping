import time
from datetime import datetime
from pathlib import Path
import psutil
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 🔁 Binance-specific modules
from liquidity_ranker import get_top_liquidity_symbols
from gainers_losers import get_top_gainers_losers
from scanner import detect_liquidity_sweep_binance
from logs.log_config import binance_trader_logger as binance
from trading_bot.send_bot_message import send_bot_message

# Lock file to prevent multiple instances
LOCK_FILE = Path("/tmp/scalp_scanner_binance_logs.lock")

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

def scan_for_scalp_opportunities_binance():
    """Run one full Binance scan cycle and return valid signals."""
    start_time = time.time()
    binance.info("🔍 Starting Binance scalp candidate scan...")

    try:
        # Get high-liquidity symbols (top 20–30)
        liquid_symbols = set(get_top_liquidity_symbols(top_n=25))

        # Get short-term movers (5-minute gainers/losers)
        gainers, losers = get_top_gainers_losers(interval_minutes=5)

        # Prioritize: liquid + moving
        high_priority = liquid_symbols & (set(gainers) | set(losers))
        candidates = sorted(high_priority or liquid_symbols)

        binance.info(f"🎯 Scanning {len(candidates)} Binance candidates: {candidates}")
        binance.debug(f"📈 Gainers: {gainers}")
        binance.debug(f"📉 Losers: {losers}")

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

                # Send Telegram alert (only for MODERATE+)
                if signal['confidence'] >= 1.3:
                    message = (
                        f"🚨 BINANCE - Scalp Signal Detected!\n"
                        f"Symbol: {symbol}\n"
                        f"Side: {signal['side'].upper()}\n"
                        f"Entry: {signal['entry']:.4f}\n"
                        f"Stop Loss: {signal['stop_loss']:.4f}\n"
                        f"Take Profit: {signal['take_profit']:.4f}\n"
                        f"Confidence: {signal['confidence']:.2f} - {confidence_level}\n"
                        f"RR: {signal['risk_reward_ratio']:.2f} | Vol Spike: {signal['volume_ratio']:.1f}x"
                    )
                    chat_id = "556159355"
                    send_bot_message(chat_id, message)

            except Exception as e:
                binance.warning(f"⚠️ Error scanning {symbol} on Binance: {e}")
                continue

        elapsed = time.time() - start_time
        binance.info(f"✅ Binance scan completed in {elapsed:.2f}s | {len(signals)} signal(s) found.")
        return signals

    except Exception as e:
        binance.error(f"💥 Binance scan cycle failed: {e}", exc_info=True)
        return []

def main_loop_binance(interval_seconds: int = 5):
    """Main loop with lock protection for Binance."""
    binance.info("🚀 Binance scalp signal monitor started. Press Ctrl+C to stop.")

    while True:
        cycle_start = time.time()
        
        # Check for existing lock
        if LOCK_FILE.exists():
            try:
                pid = int(LOCK_FILE.read_text().strip())
                if psutil.pid_exists(pid):
                    binance.warning(f"🔒 Binance scanner already running (PID {pid}). Skipping cycle.")
                    time.sleep(interval_seconds)
                    continue
                else:
                    binance.warning(f"⚠️ Stale lock detected (PID {pid} not running). Removing...")
                    LOCK_FILE.unlink()
            except Exception as e:
                binance.error(f"⚠️ Error reading lock file. Removing lock: {e}")
                LOCK_FILE.unlink()

        binance.info(f"⏱️ Starting Binance scan cycle at {datetime.now().isoformat()}")

        # Acquire lock
        try:
            LOCK_FILE.write_text(str(os.getpid()))
            binance.info(f"🔐 Lock acquired (PID: {os.getpid()})")

            # Run scan
            signals = scan_for_scalp_opportunities_binance()

            # 🔜 TODO: Add trade execution logic here
            # if signals:
            #     execute_binance_trades(signals)

        except Exception as e:
            binance.error(f"🚨 Unexpected error in Binance scan cycle: {e}")

        finally:
            # Always release lock
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
                binance.info("🔓 Binance lock released.")

        # Sleep until next cycle
        cycle_time = time.time() - cycle_start
        remaining = max(1, interval_seconds - cycle_time)  # min 1s sleep
        
        binance.info(f"📊 BINANCE CYCLE COMPLETE: {cycle_time:.2f}s | Sleeping {remaining:.1f}s")
        time.sleep(remaining)

if __name__ == "__main__":
    main_loop_binance(interval_seconds=5)