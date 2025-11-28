import os
import math
import json
import time
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from binance.exceptions import BinanceAPIException
import redis
import sys
from trading_bot.send_bot_message import send_bot_message

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logs.log_config import binance_trader_logger as logger

load_dotenv()

# Initialize Binance Futures client
client = Client(
    api_key=os.getenv("BINANCE_API_KEY"),
    api_secret=os.getenv("BINANCE_SECRET_KEY"),
    testnet=False
)

# Initialize Redis connection
redis_url = os.getenv("REDIS_URL")
if redis_url:
    try:
        redis_client = redis.from_url(redis_url)
        redis_client.ping()
        logger.info("Connected to Redis successfully")
    except redis.ConnectionError as e:
        logger.warning(f"Redis not available (optional caching disabled): {e}")
        redis_client = None
else:
    logger.info("Redis not configured (optional caching disabled)")
    redis_client = None

# Risk parameters - SAFER VALUES
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.3"))  # Reduced to 0.3%


def get_confidence_level(confidence: float) -> str:
    if confidence >= 3.0:  # STRONGER thresholds
        return "🚀 VERY STRONG"
    elif confidence >= 2.0:
        return "💪 STRONG"
    elif confidence >= 1.8:
        return "👍 MODERATE"
    else:
        return "⚠️ WEAK"

def get_futures_exchange_info(symbol: str):
    cache_key = f"futures_info_{symbol}"
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                filters = {f['filterType']: f for f in s['filters']}
                result = {
                    'pricePrecision': s['pricePrecision'],
                    'quantityPrecision': s['quantityPrecision'],
                    'stepSize': float(filters['LOT_SIZE']['stepSize']),
                    'minQty': float(filters['LOT_SIZE']['minQty']),
                    'maxQty': float(filters['LOT_SIZE']['maxQty']),
                    'minNotional': float(filters.get('MIN_NOTIONAL', {}).get('notional', 0)) or 5.0
                }
                if redis_client:
                    redis_client.setex(cache_key, 3600, json.dumps(result))
                return result
    except Exception as e:
        logger.error(f"Failed to get futures info for {symbol}: {e}")
    return None

def get_available_balance(asset: str = "USDT") -> float:
    try:
        account = client.futures_account()
        if 'assets' not in account:
            logger.error("Invalid futures account response")
            return 0.0
        for asset_info in account['assets']:
            if asset_info['asset'] == asset:
                balance = float(asset_info.get('availableBalance', 0))
                logger.info(f"Available {asset} balance: {balance}")
                return balance
        logger.warning(f"Asset {asset} not found in futures account")
    except Exception as e:
        logger.error(f"Failed to get balance: {e}")
    return 0.0

def set_leverage(symbol: str, leverage: int):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        logger.info(f"Set leverage for {symbol} to {leverage}x")
    except Exception as e:
        logger.error(f"Failed to set leverage for {symbol}: {e}")

def round_step_size(quantity: float, step_size: float) -> float:
    if step_size <= 0:
        return quantity
    precision = max(0, int(round(-math.log(step_size, 10), 0)))
    return round(quantity - (quantity % step_size), precision)

def calculate_position_size_with_margin_cap(signal: dict, available_balance: float, leverage: int, symbol_info: dict) -> float:
    """
    Calculate position size based on:
    1. Risk amount = balance * RISK_PER_TRADE_PCT / |entry - SL|
    2. Cap by max affordable notional = balance * leverage * 0.95
    """
    entry = float(signal['entry'])
    sl = float(signal['stop_loss'])
    side = signal['side'].upper()

    risk_amount = available_balance * (RISK_PER_TRADE_PCT / 100)
    risk_per_unit = abs(entry - sl)

    if risk_per_unit <= 0:
        logger.warning("Invalid stop loss placement")
        return 0.0

    # Prevent division by zero
    if risk_per_unit < 1e-10:  # Very small risk per unit
        logger.warning("Risk per unit too small, skipping trade")
        return 0.0

    qty_by_risk = risk_amount / risk_per_unit

    # Margin-based cap - SAFER
    max_notional = available_balance * leverage * 0.5  # 50% of available margin (was 75%)
    
    # Prevent division by zero for entry price
    if entry <= 0:
        logger.warning("Invalid entry price")
        return 0.0
        
    qty_by_margin = max_notional / entry

    qty = min(qty_by_risk, qty_by_margin)

    # Round and validate
    qty = round_step_size(qty, symbol_info['stepSize'])
    if qty < symbol_info['minQty']:
        logger.warning(f"Qty {qty} below minQty {symbol_info['minQty']}")
        return 0.0

    notional = qty * entry
    if notional < symbol_info['minNotional']:
        logger.warning(f"Notional ${notional:.2f} below min ${symbol_info['minNotional']}")
        return 0.0

    return qty

def place_futures_order(signal: dict):
    print(signal)
    symbol = signal['symbol']
    side = signal['side'].upper()
    
    leverage = signal.get('leverage')
    if leverage is None or leverage <= 0:
        logger.error(f"Invalid leverage in signal: {leverage}")
        return None
   
    info = get_futures_exchange_info(symbol)
    if not info:
        logger.error(f"Could not get symbol info for {symbol}")
        return None

    available_balance = get_available_balance()
    if available_balance <= 15.0:  # Minimum $15 balance required (was $10)
        logger.error(f"Insufficient balance: ${available_balance:.2f}")
        return None

    # ✅ Use margin-aware position sizing
    qty = calculate_position_size_with_margin_cap(signal, available_balance, leverage, info)
    if qty <= 0:
        logger.warning(f"Position size calculation failed for {symbol}")
        return None

    entry_price = float(signal['entry'])
    notional = qty * entry_price

    # Check if notional is reasonable for scalping
    max_safe_notional = available_balance * leverage * 0.5  # Match the margin cap from position sizing
    if notional < 15 or notional > max_safe_notional:
        logger.warning(f"Notional ${notional:.2f} outside safe range for {symbol} (max: ${max_safe_notional:.2f})")
        return None

    set_leverage(symbol, leverage)

    order_side = SIDE_BUY if side == 'BUY' else SIDE_SELL
    close_side = SIDE_SELL if side == 'BUY' else SIDE_BUY

    # Round prices
    entry_price = round(entry_price, info['pricePrecision'])
    tp_price = round(float(signal['take_profit']), info['pricePrecision'])
    sl_price = round(float(signal['stop_loss']), info['pricePrecision'])

    # SAFETY CHECK: Ensure TP and SL are valid
    if side == 'BUY':
        if tp_price <= entry_price or sl_price >= entry_price:
            logger.warning(f"Invalid TP/SL for BUY: TP={tp_price}, Entry={entry_price}, SL={sl_price}")
            return None
    else:  # SELL
        if tp_price >= entry_price or sl_price <= entry_price:
            logger.warning(f"Invalid TP/SL for SELL: TP={tp_price}, Entry={entry_price}, SL={sl_price}")
            return None

    try:
        # ✅ Use MARKET order for immediate execution - BUT with safety
        logger.info(f"Placing MARKET {side} for {symbol} | Qty: {qty} | Leverage: {leverage}x | Notional: ${notional:.2f}")
        
        # First, check if position already exists (safety check)
        try:
            position_info = client.futures_position_information(symbol=symbol)
            for pos in position_info:
                if float(pos['positionAmt']) != 0:
                    logger.warning(f"Position already exists for {symbol}, skipping order")
                    return None
        except:
            pass  # Continue if we can't check existing positions

        # Place entry order (MARKET for immediate execution)
        entry_order = client.futures_create_order(
            symbol=symbol,
            type=ORDER_TYPE_MARKET,
            side=order_side,
            quantity=qty,
            positionSide='BOTH'
        )
        entry_id = entry_order['orderId']
        logger.info(f"Entry market order filled: {entry_id} @ {entry_order['avgPrice']}")

        # Take Profit (MARKET)
        tp_order = client.futures_create_order(
            symbol=symbol,
            type='TAKE_PROFIT_MARKET',
            side=close_side,
            stopPrice=tp_price,
            positionSide='BOTH',
            workingType='MARK_PRICE',
            closePosition=True,
            priceProtect=True
        )
        tp_id = tp_order['orderId']
        logger.info(f"TP placed: {tp_id}")

        # Stop Loss (MARKET)
        sl_order = client.futures_create_order(
            symbol=symbol,
            type='STOP_MARKET',
            side=close_side,
            stopPrice=sl_price,
            positionSide='BOTH',
            workingType='MARK_PRICE',
            closePosition=True,
            priceProtect=True
        )
        sl_id = sl_order['orderId']
        logger.info(f"SL placed: {sl_id}")


        logger.info(f"✅ FULL POSITION OPENED: {symbol} | {side} | Qty: {qty} | Notional: ${notional:.2f}")

        # ✅ STORE ORDER IDs in Redis
        store_order_ids(symbol, tp_id, sl_id)

        # 🚨 CRITICAL: Verify position is still open (protects against instant SL/TP or API glitches)
        time.sleep(2)  # Let Binance settle
        try:
            pos_info = client.futures_position_information(symbol=symbol)
            current_amt = float(pos_info[0]['positionAmt'])
            if current_amt == 0:
                logger.warning(f"Position closed immediately after opening for {symbol}. Cleaning up TP/SL...")
                cleanup_specific_orders(symbol, tp_id, sl_id)  # ← you'll define this
                return None
        except Exception as e:
            logger.error(f"Failed to verify position: {e}")

        # Build the message for the bot
        confirmation_msg = (
            f"🚨 BINANCE - Scalp Signal Detected!\n"
            f"✅ POSITION OPENED\n"
            f"Symbol: {symbol}\n"
            f"Side: {signal['side'].upper()}\n"
            f"Qty: {qty:.6f}\n"
            f"Entry: {signal['entry']:.4f}\n"
            f"TP: {signal['take_profit']:.4f} (MARKET)\n"
            f"SL: {signal['stop_loss']:.4f} (MARKET)\n"
            f"Notional: ${notional:.2f}\n"
            f"Leverage: {leverage}x\n"
            f"Risk: {RISK_PER_TRADE_PCT}% of balance\n"
            f"⚠️ Auto-closing on TP/SL hit"
            f"ℹ️ If position closes but orders remain, cancel them manually in Binance"
        )
        send_bot_message(int(os.getenv("TELEGRAM_CHAT_ID")), confirmation_msg)


    except BinanceAPIException as e:
        err_msg = str(e)
        logger.error(f"Error placing orders for {symbol}: {err_msg}")
        if e.code == -2019:
            logger.warning(f"Margin insufficient. Balance=${available_balance:.2f}, Notional=${notional:.2f}, Leverage={leverage}x")
        elif e.code == -1111:
            logger.warning(f"Precision error for {symbol}")
        elif e.code == -2022:
            logger.warning(f"Reduce only error for {symbol}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing orders for {symbol}: {e}")
        return None

def cleanup_specific_orders(symbol: str, tp_id: str, sl_id: str):
    """Cancel specific TP/SL orders if they exist"""
    for order_id in [tp_id, sl_id]:
        try:
            client.futures_cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Cancelled orphaned order {order_id} for {symbol}")
        except Exception as e:
            # Order may not exist — that's OK
            pass    

def get_position_key(symbol: str) -> str:
    return f"binance_position:{symbol}"

def store_order_ids(symbol: str, tp_id: str, sl_id: str):
    """Store TP/SL order IDs in Redis with 48h TTL (safety)"""
    if redis_client:
        redis_client.hset(
            get_position_key(symbol),
            mapping={"tp_id": tp_id, "sl_id": sl_id}
        )
        redis_client.expire(get_position_key(symbol), 172800)  # 48h

def load_order_ids(symbol: str) -> dict:
    """Load TP/SL order IDs from Redis"""
    if redis_client:
        data = redis_client.hgetall(get_position_key(symbol))
        return {k.decode(): v.decode() for k, v in data.items()} if data else {}
    return {}

def clear_order_ids(symbol: str):
    """Remove TP/SL tracking from Redis"""
    if redis_client:
        redis_client.delete(get_position_key(symbol))

def cancel_orphaned_orders_for_symbol(symbol: str):
    """Cancel TP/SL if position is closed but orders remain"""
    try:
        # Check current position
        pos_info = client.futures_position_information(symbol=symbol)
        position_amt = float(pos_info[0]['positionAmt'])
        
        # If position is closed (zero size)
        if position_amt == 0:
            order_ids = load_order_ids(symbol)
            if not order_ids:
                return

            tp_id = order_ids.get("tp_id")
            sl_id = order_ids.get("sl_id")

            # Cancel any remaining TP/SL
            for order_type, order_id in [("TP", tp_id), ("SL", sl_id)]:
                if order_id:
                    try:
                        client.futures_cancel_order(symbol=symbol, orderId=order_id)
                        logger.info(f"✅ Cancelled orphaned {order_type} order {order_id} for {symbol}")
                    except Exception as e:
                        # Order may already be filled/cancelled — that's OK
                        logger.debug(f"Orphaned {order_type} order not found: {e}")

            # Clean up Redis
            clear_order_ids(symbol)

    except Exception as e:
        logger.error(f"Error checking/canceling orphans for {symbol}: {e}")        


def cleanup_all_orphaned_orders():
    """Check all tracked symbols for orphaned orders"""
    if not redis_client:
        return

    # Get all position keys
    keys = redis_client.keys("binance_position:*")
    for key in keys:
        try:
            symbol_bytes = key.split(b":")[-1]
            symbol = symbol_bytes.decode()
            cancel_orphaned_orders_for_symbol(symbol)
        except Exception as e:
            logger.error(f"Failed to process key {key}: {e}")


def recover_order_state_on_startup():
    """On startup, scan Binance for open positions + TP/SL orders, rebuild Redis"""
    try:
        positions = client.futures_position_information()
        for pos in positions:
            symbol = pos['symbol']
            amt = float(pos['positionAmt'])
            if amt != 0:  # Active position
                # Find associated TP/SL orders
                open_orders = client.futures_get_open_orders(symbol=symbol)
                tp_id = None
                sl_id = None
                for order in open_orders:
                    if order['type'] == 'TAKE_PROFIT_MARKET':
                        tp_id = order['orderId']
                    elif order['type'] == 'STOP_MARKET':
                        sl_id = order['orderId']
                
                # Only store if at least one TP/SL exists
                if tp_id or sl_id:
                    store_order_ids(symbol, tp_id or "", sl_id or "")
                    logger.info(f"🔁 Recovered state for {symbol}: TP={tp_id}, SL={sl_id}")
                else:
                    logger.warning(f"Active position for {symbol} but no TP/SL found")
    except Exception as e:
        logger.error(f"State recovery failed: {e}")