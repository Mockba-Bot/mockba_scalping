import time
from datetime import datetime
from pathlib import Path
import psutil
import os
import sys
import threading
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Load environment variables
load_dotenv()

# 🔁 Binance-specific modules
from liquidity_ranker import get_top_liquidity_symbols
from gainers_losers import get_top_gainers_losers
from scanner import detect_liquidity_sweep_binance
from logs.log_config import binance_trader_logger as binance
from trading_bot.send_bot_message import send_bot_message
from trading_bot.futures_executor_binance import place_futures_order
from db.db_ops import insert_position_with_orders, get_open_positions, update_position_pnl, get_all_signal_statuses

# Binance client for position monitoring
from binance.client import Client as BinanceClient
monitor_client = BinanceClient(
    api_key=os.getenv("BINANCE_API_KEY"),
    api_secret=os.getenv("BINANCE_SECRET_KEY"),
    testnet=False
)

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
            qty = float(order_info['executedQty'])
            
            # Calculate current PnL
            current_price = float(monitor_client.futures_symbol_ticker(symbol=db_row['symbol'])['price'])
            if db_row['side'] == 'BUY':
                pnl_pct = (current_price - fill_price) / fill_price * 100
            else:
                pnl_pct = (fill_price - current_price) / fill_price * 100
                
            pnl_usd = (pnl_pct / 100) * db_row['notional_usd']
            
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
                update_position_pnl(position_id, pnl_pct, pnl_usd, status='TP')
            elif sl_info['status'] == 'FILLED':
                update_position_pnl(position_id, pnl_pct, pnl_usd, status='SL')
                
    except Exception as e:
        binance.error(f"Error updating position {position_id}: {e}")

def position_monitor_loop():
    """Continuously monitor all open positions."""
    binance.info("🚀 Starting position monitor...")
    
    while True:
        try:
            open_positions = get_open_positions()
            
            if open_positions:
                binance.debug(f"Monitoring {len(open_positions)} open positions")
                
                for pos in open_positions:
                    update_position_from_binance(pos['id'], pos)
                    time.sleep(0.1)  # Respect Binance rate limits (10 req/sec)
                
            time.sleep(2)  # Check every 2 seconds when no positions
            
        except KeyboardInterrupt:
            binance.info("Position monitor stopped by user")
            break
        except Exception as e:
            binance.error(f"Position monitor error: {e}")
            time.sleep(5)

def scan_for_scalp_opportunities_binance():
    """Run one full Binance scan cycle and return valid signals."""
    start_time = time.time()
    binance.info("🔍 Starting Binance scalp candidate scan...")

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

        binance.info(f"🎯 Scanning {len(candidates)} Binance candidates: {candidates}")
        binance.debug(f"📈 Gainers: {gainers_5m} {gainers_15m}")
        binance.debug(f"📉 Losers: {losers_5m} {losers_15m}")

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
                    chat_statuses = get_all_signal_statuses()
                    enabled_chats = [e['chat_id'] for e in chat_statuses if e['signals_enabled']]

                    if not enabled_chats:
                       continue

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

                    # 🔥 Open position immediately
                    order_result = place_futures_order(signal)

                    if order_result:
                        for chat_id in enabled_chats:
                            # ✅ Send confirmation that position was opened
                            confirmation_msg = (
                                f"✅ POSITION OPENED on BINANCE\n"
                                f"Symbol: {symbol}\n"
                                f"Side: {signal['side'].upper()}\n"
                                f"Qty: {order_result['quantity']:.6f}\n"
                                f"Entry: {signal['entry']:.4f}\n"
                                f"TP: {signal['take_profit']:.4f} (MARKET)\n"
                                f"SL: {signal['stop_loss']:.4f} (MARKET)\n"
                                f"⚠️ Auto-closing on TP/SL hit"
                            )
                            send_bot_message(chat_id, confirmation_msg)

                            # ✅ INSERT POSITION INTO DATABASE
                            insert_position_with_orders(
                                chat_id=int(chat_id),
                                signal=signal,
                                order_result=order_result,
                                exchange="BINANCE"
                            )
                    else:
                        binance.warning(f"⚠️ Failed to place order for {symbol} on Binance, no alerts sent.")        

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
    # Start position monitor in background thread
    monitor_thread = threading.Thread(
        target=position_monitor_loop,
        daemon=True,
        name="PositionMonitor"
    )
    monitor_thread.start()
    binance.info("✅ Position monitor started in background")
    
    # Start main scanner loop
    main_loop_binance(interval_seconds=5)