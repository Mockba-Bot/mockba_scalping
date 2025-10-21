import ccxt
import redis
import json
import time
import os
from datetime import datetime
from decimal import Decimal, getcontext
from dotenv import load_dotenv
import psycopg2  # Optional: comment out if not using DB

load_dotenv()

# Set precision for Decimal
getcontext().prec = 28

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
    print("✅ Connected to Redis")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
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
        print(f"⚠️ DB connection failed: {e}")
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
        print(f"⚠️ Error fetching {symbol} from {exchange.id}: {e}")
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
        print(f"\n🔍 Scanning {symbol} across exchanges...")
        prices = {}
        
        # Fetch prices
        for name, ex in exchanges.items():
            price = get_cached_price(ex, symbol)
            if price:
                prices[name] = price
                print(f"  {name}: ${price:,.2f}")
        
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
                    print(f"✅ ARB: Buy {symbol} on {buy_ex} @ ${buy_price:,.2f} → Sell on {sell_ex} @ ${sell_price:,.2f} | Net: {opp['net_profit_pct']}%")
    
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
        print(f"💾 Logged {len(opportunities)} opportunities to DB")
    except Exception as e:
        print(f"❌ DB insert error: {e}")
    finally:
        conn.close()

# === MAIN LOOP ===
def main():
    print("🚀 Starting Mockba Cross-Exchange Arbitrage Engine")
    print(f"Exchanges: Binance, KuCoin, Bitget, Bybit")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Min Spread: {MIN_SPREAD_PCT}% | Scan Interval: {SCAN_INTERVAL}s")
    
    exchanges = create_exchanges()
    
    while True:
        try:
            opportunities = scan_opportunities(exchanges)
            
            if opportunities:
                print(f"\n🎯 FOUND {len(opportunities)} ARBITRAGE OPPORTUNITIES!")
                # Optional: log to DB
                # log_to_db(opportunities)
            else:
                print("⏳ No profitable opportunities found.")
                
            time.sleep(SCAN_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n🛑 Scanner stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Main loop error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()