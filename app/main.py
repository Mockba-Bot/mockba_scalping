import time
from datetime import datetime
from scalping.binance.liquidity_ranker import get_top_liquidity_symbols
from scalping.binance.gainers_losers import get_top_gainers_losers
from scalping.binance.scanner import detect_liquidity_sweep
from logs.log_config import trader_logger as logging
from trading_bot.send_bot_message import send_bot_message

def scan_for_scalp_opportunities():
    """Run one full scan cycle and return valid signals."""
    start_time = time.time()
    logging.info("🔍 Starting scalp candidate scan...")

    try:
        # Get high-liquidity symbols
        liquid_symbols = set(get_top_liquidity_symbols(top_n=30))

        # Get volatile movers (5-min gainers/losers)
        gainers, losers = get_top_gainers_losers(interval_minutes=5)

        # Union of all candidates
        all_symbols = liquid_symbols | set(gainers) | set(losers)
        candidates = sorted(all_symbols)

        logging.info(f"🎯 Scanning {len(candidates)} unique candidates: {', '.join(candidates[:10])}{'...' if len(candidates) > 10 else ''}")

        signals = []
        for symbol in candidates:
            try:
                signal = detect_liquidity_sweep(symbol)
                if signal:
                    signals.append(signal)
                    logging.info(f"✅ Signal: {symbol} | {signal['side'].upper()} | Entry: {signal['entry']:.4f} | SL: {signal['stop_loss']:.4f} | TP: {signal['take_profit']:.4f} | Conf: {signal['confidence']:.2f}")
                    # Send Telegram alert
                    message = (f"🚨 Scalp Signal Detected!\n"
                               f"Symbol: {symbol}\n"
                               f"Side: {signal['side'].upper()}\n"
                               f"Entry: {signal['entry']:.4f}\n"
                               f"Stop Loss: {signal['stop_loss']:.4f}\n"
                               f"Take Profit: {signal['take_profit']:.4f}\n"
                               f"Confidence: {signal['confidence']:.2f}")
                    send_bot_message(message)
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
    """Main loop: runs scan every `interval_seconds`, aligned to wall-clock minutes."""
    logging.info("🚀 Scalp signal monitor started. Press Ctrl+C to stop.")
    next_run = time.time()

    while True:
        try:
            # Align to nearest minute boundary (optional but cleaner)
            now = time.time()
            sleep_time = max(0, next_run - now)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Run scan
            signals = scan_for_scalp_opportunities()

            # 🔜 TODO: Add trade execution logic here (e.g., submit orders via broker API)
            # For now, just collect or log signals
            # if signals:
            #     execute_trades(signals)

            # Schedule next run
            next_run += interval_seconds

            # Guard against drift: if we're behind, snap to next clean interval
            now = time.time()
            if next_run < now:
                # Skip missed cycles; align to next full minute
                next_run = now + interval_seconds

        except KeyboardInterrupt:
            logging.info("🛑 Shutting down gracefully...")
            break
        except Exception as e:
            logging.error(f"🔥 Unexpected error in main loop: {e}")
            time.sleep(5)  # brief pause before retry

if __name__ == "__main__":
    main_loop(interval_seconds=60)