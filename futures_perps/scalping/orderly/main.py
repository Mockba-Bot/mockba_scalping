import time
from datetime import datetime
from pathlib import Path
import psutil
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from liquidity_ranker import get_top_liquidity_symbols
from gainers_losers import get_top_gainers_losers
from scanner import detect_liquidity_sweep
from logs.log_config import trader_logger as orderly
from trading_bot.send_bot_message import send_bot_message

# Lock file to prevent multiple instances
LOCK_FILE = Path("/tmp/scalp_scanner_orderly.lock")

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

def scan_for_scalp_opportunities():
    """Run one full scan cycle and return valid signals."""
    start_time = time.time()
    orderly.info("🔍 Starting Orderly scalp candidate scan...")

    try:
        # Get top liquid symbols (focus on top 30 for HFT)
        liquid_symbols = set(get_top_liquidity_symbols(top_n=30))  # Reduced from 30

        # Get short-term movers (use 5–10 min, not 30)
        gainers, losers = get_top_gainers_losers(interval_minutes=30)

        # Prioritize: liquid + moving
        high_priority = liquid_symbols & (set(gainers) | set(losers))
        # Fallback: just liquid (if no movers)
        candidates = sorted(high_priority or liquid_symbols)

        orderly.info(f"🎯 Scanning {len(candidates)} high-priority candidates: {candidates}")
        orderly.debug(f"📈 Gainers: {gainers}")
        orderly.debug(f"📉 Losers: {losers}")

        signals = []
        for symbol in candidates:
            try:
                signal = detect_liquidity_sweep(symbol)
                if not signal:
                    continue

                # Optional: skip if signal too old (e.g., >2s)
                if time.time() - signal['timestamp'] > 2.0:
                    continue

                confidence_level = get_confidence_level(signal['confidence'])
                signals.append(signal)

                log_msg = (
                    f"✅ Signal: {symbol} | {signal['side'].upper()} | "
                    f"Entry: {signal['entry']:.4f} | SL: {signal['stop_loss']:.4f} | "
                    f"TP: {signal['take_profit']:.4f} | Conf: {signal['confidence']:.2f} | {confidence_level}"
                )
                orderly.info(log_msg)

                # Send Telegram alert (only for MODERATE+)
                if signal['confidence'] >= 1.3:
                    message = (
                        f"🚨 ORDERLY - Scalp Signal Detected!\n"
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
                orderly.warning(f"⚠️ Error scanning {symbol}: {e}")
                continue

        elapsed = time.time() - start_time
        orderly.info(f"✅ Scan completed in {elapsed:.2f}s | {len(signals)} signal(s) found.")
        return signals

    except Exception as e:
        orderly.error(f"💥 Scan cycle failed: {e}", exc_info=True)
        return []

def main_loop(interval_seconds: int = 60):
    """Main loop with lock protection."""
    orderly.info("🚀 Scalp signal monitor started. Press Ctrl+C to stop.")

    while True:
        cycle_start = time.time()
        
        # Check for existing lock
        if LOCK_FILE.exists():
            try:
                pid = int(LOCK_FILE.read_text().strip())
                if psutil.pid_exists(pid):
                    orderly.warning(f"🔒 Scanner already running (PID {pid}). Skipping this cycle.")
                    time.sleep(interval_seconds)
                    continue
                else:
                    orderly.warning(f"⚠️ Stale lock detected (PID {pid} not running). Removing...")
                    LOCK_FILE.unlink()
            except Exception as e:
                orderly.error(f"⚠️ Error reading lock file. Removing lock: {e}")
                LOCK_FILE.unlink()

        orderly.info(f"⏱️ Starting scan cycle at {datetime.now().isoformat()}")

        # Acquire lock
        try:
            LOCK_FILE.write_text(str(os.getpid()))
            orderly.info(f"🔐 Lock acquired (PID: {os.getpid()})")

            # Run scan
            signals = scan_for_scalp_opportunities()

            # 🔜 TODO: Add trade execution logic here
            # if signals:
            #     execute_trades(signals)

        except Exception as e:
            orderly.error(f"🚨 Unexpected error in scan cycle: {e}")

        finally:
            # Always release lock
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
                orderly.info("🔓 Lock released.")

        # Log complete cycle time and calculate sleep
        cycle_time = time.time() - cycle_start
        remaining = max(5, interval_seconds - cycle_time)
        
        orderly.info(f"📊 CYCLE COMPLETE: Total cycle time: {cycle_time:.2f}s | Sleeping {remaining:.1f}s until next cycle")
        
        time.sleep(remaining)

if __name__ == "__main__":
    main_loop(interval_seconds=5)