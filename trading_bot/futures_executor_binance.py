import os
import math
import time
import json
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
import redis
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logs.log_config import binance_trader_logger as logger

load_dotenv()

# Initialize Binance Futures client
client = Client(
    api_key=os.getenv("BINANCE_API_KEY"),
    api_secret=os.getenv("BINANCE_SECRET_KEY"),
    testnet=False
)

# Redis
try:
    redis_client = redis.StrictRedis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True
    )
    redis_client.ping()
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

# Risk parameters
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))  # ← Now defaults to 1%
LEVERAGE_MAP = {
    "🚀 VERY STRONG": int(os.getenv("MAX_LEVERAGE_VERY_STRONG", "20")),
    "💪 STRONG": int(os.getenv("MAX_LEVERAGE_STRONG", "15")),
    "👍 MODERATE": int(os.getenv("MAX_LEVERAGE_MODERATE", "10")),
    "⚠️ WEAK": 0,
    "❌ VERY WEAK": 0
}

def get_confidence_level(confidence: float) -> str:
    if confidence >= 2.5:
        return "🚀 VERY STRONG"
    elif confidence >= 1.8:
        return "💪 STRONG"
    elif confidence >= 1.3:
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
        for asset_info in account['assets']:
            if asset_info['asset'] == asset:
                return float(asset_info['availableBalance'])
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
    precision = int(round(-math.log(step_size, 10), 0))
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

    qty_by_risk = risk_amount / risk_per_unit

    # Margin-based cap
    max_notional = available_balance * leverage * 0.95
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
    symbol = signal['symbol']
    side = signal['side'].upper()
    confidence = signal['confidence']
    confidence_level = get_confidence_level(confidence)

    if confidence_level not in LEVERAGE_MAP or LEVERAGE_MAP[confidence_level] == 0:
        logger.info(f"Signal too weak ({confidence_level}), skipping: {symbol}")
        return None

    leverage = LEVERAGE_MAP[confidence_level]
    info = get_futures_exchange_info(symbol)
    if not info:
        logger.error(f"Could not get symbol info for {symbol}")
        return None

    available_balance = get_available_balance()
    if available_balance <= 5.0:
        logger.error("No available balance")
        return None

    # ✅ Use margin-aware position sizing
    qty = calculate_position_size_with_margin_cap(signal, available_balance, leverage, info)
    if qty <= 0:
        return None

    entry_price = float(signal['entry'])
    notional = qty * entry_price

    set_leverage(symbol, leverage)

    order_side = SIDE_BUY if side == 'BUY' else SIDE_SELL
    close_side = SIDE_SELL if side == 'BUY' else SIDE_BUY

    # Round prices
    entry_price = round(entry_price, info['pricePrecision'])
    tp_price = round(float(signal['take_profit']), info['pricePrecision'])
    sl_price = round(float(signal['stop_loss']), info['pricePrecision'])

    try:
        # ✅ Use MARKET order for immediate execution
        logger.info(f"Placing LIMIT {side} for {symbol} | Qty: {qty} | Leverage: {leverage}x")
        entry_order = client.futures_create_order(
            symbol=symbol,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,  # or TIME_IN_FORCE_IOC if you prefer immediate-or-cancel
            side=order_side,
            price=entry_price,
            quantity=qty,
            positionSide='BOTH'
        )
        entry_id = entry_order['orderId']
        logger.info(f"Entry order placed: {entry_id}")

        # Take Profit (MARKET)
        tp_order = client.futures_create_order(
            symbol=symbol,
            type='TAKE_PROFIT_MARKET',
            side=close_side,
            stopPrice=tp_price,
            quantity=qty,
            positionSide='BOTH',
            reduceOnly=True
        )
        tp_id = tp_order['orderId']
        logger.info(f"Take-profit market order placed: {tp_id} @ {tp_price}")

        # Stop Loss (MARKET)
        sl_order = client.futures_create_order(
            symbol=symbol,
            type='STOP_MARKET',
            side=close_side,
            stopPrice=sl_price,
            quantity=qty,
            positionSide='BOTH',
            reduceOnly=True
        )
        sl_id = sl_order['orderId']
        logger.info(f"Stop-loss market order placed: {sl_id} @ {sl_price}")

        logger.info(f"✅ FULL POSITION OPENED: {symbol} | {side} | Qty: {qty} | Notional: ${notional:.2f}")
        return {
            'entry_order_id': entry_id,
            'tp_order_id': tp_id,
            'sl_order_id': sl_id,
            'quantity': qty,
            'notional': notional,
            'symbol': symbol,
            'side': side
        }

    except BinanceAPIException as e:
        err_msg = str(e)
        logger.error(f"Error placing orders for {symbol}: {err_msg}")
        if e.code == -2019:
            logger.warning(f"Margin insufficient. Balance={available_balance:.4f}, Notional={notional:.2f}, Leverage={leverage}x")
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing orders for {symbol}: {e}")
        return None