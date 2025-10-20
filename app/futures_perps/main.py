import time
from datetime import datetime
from pathlib import Path
import psutil
import os
import sys
sys.path.append('./app')
from futures_perps.scalping.orderly.liquidity_ranker import get_top_liquidity_symbols
from futures_perps.scalping.orderly.gainers_losers import get_top_gainers_losers
from futures_perps.scalping.orderly.scanner import detect_liquidity_sweep
from logs.log_config import trader_logger as logging
from trading_bot.send_bot_message import send_bot_message

# Lock file to prevent multiple instances
LOCK_FILE = Path("/tmp/scalp_scanner.lock")

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
    logging.info("🔍 Starting scalp candidate scan...")

    try:
        # Get high-liquidity symbols
        liquid_symbols = set(get_top_liquidity_symbols(top_n=30))

        # Get volatile movers (5-min gainers/losers)
        gainers, losers = get_top_gainers_losers(interval_minutes=30)

        # Union of all candidates
        all_symbols = liquid_symbols | set(gainers) | set(losers)
        candidates = sorted(all_symbols)

        logging.info(f"🎯 Scanning {len(candidates)} unique candidates")

        signals = []
        for symbol in candidates:
            try:
                signal = detect_liquidity_sweep(symbol)
                if signal:
                    confidence_level = get_confidence_level(signal['confidence'])
                    signals.append(signal)
                    logging.info(f"✅ Signal: {symbol} | {signal['side'].upper()} | Entry: {signal['entry']:.4f} | SL: {signal['stop_loss']:.4f} | TP: {signal['take_profit']:.4f} | Conf: {signal['confidence']:.2f} | {confidence_level}")
                    # Send Telegram alert
                    message = (f"🚨 Scalp Signal Detected!\n"
                               f"Symbol: {symbol}\n"
                               f"Side: {signal['side'].upper()}\n"
                               f"Entry: {signal['entry']:.4f}\n"
                               f"Stop Loss: {signal['stop_loss']:.4f}\n"
                               f"Take Profit: {signal['take_profit']:.4f}\n"
                               f"Confidence: {signal['confidence']:.2f} - {confidence_level}\n"
                               f"Level: {confidence_level.split()[-1]}")
                    chat_id = "556159355"
                    send_bot_message(chat_id, message)
            except Exception as e:
                logging.warning(f"⚠️ Error scanning {symbol}: {e}")
                continue

        elapsed = time.time() - start_time
        logging.info(f"✅ Scan completed in {elapsed:.2f}s | {len(signals)} signal(s) found.")
        return signals

    except Exception as e:
        logging.error(f"💥 Scan cycle failed: {e}")
        return []

def main_loop(interval_seconds: int = 60):
    """Main loop with lock protection."""
    logging.info("🚀 Scalp signal monitor started. Press Ctrl+C to stop.")

    while True:
        cycle_start = time.time()
        
        # Check for existing lock
        if LOCK_FILE.exists():
            try:
                pid = int(LOCK_FILE.read_text().strip())
                if psutil.pid_exists(pid):
                    logging.warning(f"🔒 Scanner already running (PID {pid}). Skipping this cycle.")
                    time.sleep(interval_seconds)
                    continue
                else:
                    logging.warning(f"⚠️ Stale lock detected (PID {pid} not running). Removing...")
                    LOCK_FILE.unlink()
            except Exception as e:
                logging.error(f"⚠️ Error reading lock file. Removing lock: {e}")
                LOCK_FILE.unlink()

        logging.info(f"⏱️ Starting scan cycle at {datetime.now().isoformat()}")

        # Acquire lock
        try:
            LOCK_FILE.write_text(str(os.getpid()))
            logging.info(f"🔐 Lock acquired (PID: {os.getpid()})")

            # Run scan
            signals = scan_for_scalp_opportunities()

            # 🔜 TODO: Add trade execution logic here
            # if signals:
            #     execute_trades(signals)

        except Exception as e:
            logging.error(f"🚨 Unexpected error in scan cycle: {e}")

        finally:
            # Always release lock
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
                logging.info("🔓 Lock released.")

        # Log complete cycle time and calculate sleep
        cycle_time = time.time() - cycle_start
        remaining = max(5, interval_seconds - cycle_time)
        
        logging.info(f"📊 CYCLE COMPLETE: Total cycle time: {cycle_time:.2f}s | Sleeping {remaining:.1f}s until next cycle")
        
        time.sleep(remaining)

if __name__ == "__main__":
    main_loop(interval_seconds=60)