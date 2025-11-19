import ccxt
import redis
import json
import time
import os
import sys
from datetime import datetime
from decimal import Decimal, getcontext
from dotenv import load_dotenv
import psycopg2  # Optional: comment out if not using DB

load_dotenv()

# Set precision for Decimal
getcontext().prec = 28

# -------------------------
# Resolve project root for imports: divergences -> add app/
# -------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -------------------------
# Logger
# -------------------------
from logs.log_config import scalping_logger as scalping_logger
from trading_bot.send_bot_message import send_bot_message

# === CONFIGURATION ===
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
MIN_SPREAD_PCT = 0.5  # Minimum spread to trigger signal (0.5%)
SCAN_INTERVAL = 5.0    # Seconds between scans

# Trading fees per exchange (taker fees for spot)
EXCHANGE_FEES = {
    'binance': 0.001,   # 0.1%
    'kucoin': 0.001,    # 0.1%
    'bitget': 0.001,    # 0.1%
    'bybit': 0.001,     # 0.1%
}

# Symbols to monitor (add more as needed)
SYMBOLS = [
    # Core large-caps
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LTC/USDT',
    
    # Mid-caps with good liquidity
    'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'NEAR/USDT', 'APT/USDT',
    'SUI/USDT', 'OP/USDT', 'ARB/USDT', 'ICP/USDT', 'FIL/USDT',
    
    # High-volatility (use cautiously)
    'PEPE/USDT', 'WIF/USDT', 'SHIB/USDT',
]

# === REDIS SETUP ===
try:
    redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=1, decode_responses=True)
    redis_client.ping()
    scalping_logger.info("Connected to Redis")
except Exception as e:
    scalping_logger.error(f"Redis connection failed: {e}")
    redis_client = None

# === POSTGRESQL SETUP (Optional) ===
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'mockba'),
            user=os.getenv('DB_USER', 'user'),
            password=os.getenv('DB_PASSWORD', 'password')
        )
        return conn
    except Exception as e:
        scalping_logger.warning(f"DB connection failed: {e}")
        return None

# === EXCHANGE SETUP ===
def create_exchanges():
    return {
        'binance': ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }),
        'kucoin': ccxt.kucoin({
            'enableRateLimit': True,
        }),
        'bitget': ccxt.bitget({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }),
        'bybit': ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }),
    }

# === PRICE FETCHING WITH REDIS CACHE ===
def get_cached_price(exchange, symbol, ttl_ms=500):
    """Fetch price with sub-second Redis caching"""
    cache_key = f"price:{exchange.id}:{symbol}"
    
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            return float(cached)
    
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        if redis_client:
            ttl_seconds = max(1, ttl_ms // 1000)  # Ensure at least 1 second TTL
            redis_client.setex(cache_key, ttl_seconds, str(price))
        return price
    except Exception as e:
        scalping_logger.warning(f"Error fetching {symbol} from {exchange.id}: {e}")
        return None

# === FEE-AWARE PROFIT CALCULATION ===
def calculate_net_profit(buy_ex, sell_ex, buy_price, sell_price, amount_usd=1000):
    """
    Calculate net profit after fees for cross-exchange arb
    Assumes user has pre-funded accounts (no withdrawal fees)
    """
    buy_fee = EXCHANGE_FEES[buy_ex]
    sell_fee = EXCHANGE_FEES[sell_ex]
    
    # Buy BTC on buy_ex
    btc_bought = (amount_usd / buy_price) * (1 - buy_fee)
    
    # Sell BTC on sell_ex
    usd_received = (btc_bought * sell_price) * (1 - sell_fee)
    
    net_profit = usd_received - amount_usd
    profit_pct = (net_profit / amount_usd) * 100
    
    return {
        'gross_spread_pct': ((sell_price - buy_price) / buy_price) * 100,
        'net_profit_usd': net_profit,
        'net_profit_pct': profit_pct,
        'btc_amount': btc_bought,
        'usd_invested': amount_usd,
        'usd_out': usd_received
    }

# === ARBITRAGE SCANNER ===
def scan_opportunities(exchanges):
    opportunities = []
    
    for symbol in SYMBOLS:
        scalping_logger.info(f"Scanning {symbol} across exchanges...")
        prices = {}
        
        # Fetch prices
        for name, ex in exchanges.items():
            price = get_cached_price(ex, symbol)
            if price:
                prices[name] = price
                scalping_logger.debug(f"  {name}: ${price:,.2f}")
        
        if len(prices) < 2:
            continue
            
        # Find best buy/sell pairs
        exchange_names = list(prices.keys())
        for buy_ex in exchange_names:
            for sell_ex in exchange_names:
                if buy_ex == sell_ex:
                    continue
                    
                buy_price = prices[buy_ex]
                sell_price = prices[sell_ex]
                
                if sell_price <= buy_price:
                    continue
                
                gross_spread = ((sell_price - buy_price) / buy_price) * 100
                if gross_spread < MIN_SPREAD_PCT:
                    continue
                
                # Calculate net profit (after fees)
                profit = calculate_net_profit(buy_ex, sell_ex, buy_price, sell_price)
                
                if profit['net_profit_pct'] > 0:
                    opp = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'symbol': symbol,
                        'buy_exchange': buy_ex,
                        'sell_exchange': sell_ex,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'gross_spread_pct': round(gross_spread, 3),
                        'net_profit_pct': round(profit['net_profit_pct'], 3),
                        'net_profit_usd': round(profit['net_profit_usd'], 2),
                        'usd_invested': profit['usd_invested']
                    }
                    opportunities.append(opp)
                    scalping_logger.info(f"ARB: Buy {symbol} on {buy_ex} @ ${buy_price:,.2f} → Sell on {sell_ex} @ ${sell_price:,.2f} | Net: {opp['net_profit_pct']}%")
    
    return opportunities

# === DATABASE LOGGING (Optional) ===
def log_to_db(opportunities):
    conn = get_db_connection()
    if not conn:
        return
        
    try:
        cur = conn.cursor()
        for opp in opportunities:
            cur.execute("""
                INSERT INTO arbitrage_signals (
                    timestamp, symbol, buy_exchange, sell_exchange,
                    buy_price, sell_price, gross_spread_pct,
                    net_profit_pct, net_profit_usd, usd_invested
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                opp['timestamp'], opp['symbol'], opp['buy_exchange'], opp['sell_exchange'],
                opp['buy_price'], opp['sell_price'], opp['gross_spread_pct'],
                opp['net_profit_pct'], opp['net_profit_usd'], opp['usd_invested']
            ))
        conn.commit()
        cur.close()
        scalping_logger.info(f"Logged {len(opportunities)} opportunities to DB")
    except Exception as e:
        scalping_logger.error(f"DB insert error: {e}")
    finally:
        conn.close()

# === MAIN LOOP ===
def main():
    scalping_logger.info("Starting Mockba Cross-Exchange Arbitrage Engine")
    scalping_logger.info("Exchanges: Binance, KuCoin, Bitget, Bybit")
    scalping_logger.info(f"Symbols: {', '.join(SYMBOLS)}")
    scalping_logger.info(f"Min Spread: {MIN_SPREAD_PCT}% | Scan Interval: {SCAN_INTERVAL}s")
    
    exchanges = create_exchanges()
    
    while True:
        try:
            opportunities = scan_opportunities(exchanges)
            
            if opportunities:
                scalping_logger.info(f"FOUND {len(opportunities)} ARBITRAGE OPPORTUNITIES!")
                # Optional: log to DB
                # log_to_db(opportunities)
                # send bot messages
                message = "🔺 *Triangular Arbitrage Opportunities Found!* 🔺\n\n"
                for opp in opportunities:
                    message += f"💰 {opp['symbol']}: {opp['net_profit_pct']}% | ${opp['net_profit_usd']} | Buy: {opp['buy_exchange']} @ {opp['buy_price']} | Sell: {opp['sell_exchange']} @ {opp['sell_price']}\n"
                send_bot_message(message)
            else:
                scalping_logger.info("No profitable opportunities found.")
                
            time.sleep(SCAN_INTERVAL)
            
        except KeyboardInterrupt:
            scalping_logger.info("Scanner stopped by user")
            break
        except Exception as e:
            scalping_logger.error(f"Main loop error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()