import json
import pandas as pd
import io
import requests
import os
from datetime import datetime
import threading
import time
import sys
import re
import redis
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from db.db_ops import get_open_positions, update_position_pnl, initialize_database_tables, get_bot_status
from logs.log_config import binance_trader_logger as logger
from binance.client import Client as BinanceClient
from trading_bot.send_bot_message import send_bot_message
from historical_data import get_historical_data_limit_binance, get_orderbook

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Redis connection
redis_url = os.getenv("REDIS_URL")
if redis_url:
    try:
        redis_client = redis.from_url(redis_url)
        redis_client.ping()
    except redis.ConnectionError as e:
        print(f"Redis connection error: {e}")
        redis_client = None
else:
    redis_client = None


# Import your executor
from trading_bot.futures_executor_binance import place_futures_order, get_confidence_level as executor_get_confidence_level

# Import your liquidity persistence monitor
import liquidity_persistence_monitor as lpm

RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.5"))
MAX_LEVERAGE_HIGH = int(os.getenv("MAX_LEVERAGE_HIGH", "5"))
MAX_LEVERAGE_MEDIUM = int(os.getenv("MAX_LEVERAGE_MEDIUM", "4"))
MAX_LEVERAGE_SMALL = int(os.getenv("MAX_LEVERAGE_SMALL", "3"))
MICRO_BACKTEST_MIN_EXPECTANCY = float(os.getenv("MICRO_BACKTEST_MIN_EXPECTANCY", "0.0025"))

# Request model for your signal
class TradingSignal(BaseModel):
    asset: str
    signal: str  # "LONG" or "SHORT"
    confidence: float  # 0-100%
    timeframe: str  # "4h", "1h", etc.
    current_price: float
    liquidity_score: float
    volume_1h: float
    volatility_1h: float

def get_confidence_level(confidence: float) -> str:
    """Map confidence score to human-readable level for ML signals (0-100 scale)"""
    if confidence >= 80:  # Updated for your 100% scale
        return "🚀 VERY STRONG"
    elif confidence >= 70:
        return "💪 STRONG"
    elif confidence >= 60:
        return "👍 MODERATE"
    else:
        return "❌ WEAK"

def get_leverage_by_confidence(confidence: float) -> int:
    """Get leverage based on confidence level"""
    if confidence >= 80:
        return MAX_LEVERAGE_HIGH  # High confidence = max leverage
    elif confidence >= 70:
        return MAX_LEVERAGE_MEDIUM   # Medium confidence = moderate leverage
    elif confidence >= 60:
        return MAX_LEVERAGE_SMALL   # Low confidence = low leverage
    else:
        return 1   # Very low confidence = minimal leverage

def load_prompt_template():
    """Load LLM prompt from file"""
    try:
        with open("llm_prompt_template.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError("llm_prompt_template.txt not found. Please create the prompt file.")

def get_current_balance():
    """Get current account balance from Binance"""
    from binance.client import Client as BinanceClient
    import os
    
    client = BinanceClient(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_SECRET_KEY"),
        testnet=False
    )
    
    try:
        account = client.futures_account()
        for asset_info in account['assets']:
            if asset_info['asset'] == 'USDT':
                return float(asset_info['marginBalance'])
    except Exception as e:
        # Default to 20 if API call fails
        return 20.0

# Helper: Format orderbook as text (not CSV!)
def format_orderbook_as_text(ob: dict) -> str:
    lines = ["Top Bids (price, quantity):"]
    for price, qty in ob.get('bids', [])[:15]:
        lines.append(f"{price},{qty}")
    
    lines.append("\nTop Asks (price, quantity):")
    for price, qty in ob.get('asks', [])[:15]:
        lines.append(f"{price},{qty}")
    
    return "\n".join(lines)


def analyze_with_llm(signal_dict: dict) -> dict:
    """Send to LLM for detailed analysis using fixed prompt structure."""
    
    # ✅ Get DataFrame with ALL indicators (your function handles timeframe logic)
    df = get_historical_data_limit_binance(
        symbol=signal_dict['asset'],
        interval=signal_dict['timeframe'],
        limit=80
    )
    csv_content = df.to_csv(index=False)  # ← Preserves all columns automatically

    # ✅ Get orderbook as TEXT (not CSV!)
    orderbook = get_orderbook(signal_dict['asset'], limit=20)
    orderbook_content = format_orderbook_as_text(orderbook)  # ← See helper below

    # --- Rest of your prompt logic (unchanged) ---
    intro = (
        "You are an experienced retail crypto trader with 10 years of experience.\n"
        "Analyze the attached CSV (80 candles) and orderbook for the given signal.\n\n"
        f"• Asset: {signal_dict['asset']}\n"
        f"• Signal: {signal_dict['signal']}\n"
        f"• Confidence: {signal_dict['confidence']}%\n"
        f"• Timeframe: {signal_dict['timeframe']}\n"
        f"• Current Price: ${signal_dict['current_price']}\n"
        f"• Liquidity Score: {signal_dict['liquidity_score']}\n"
        f"• Volume (1h): ${signal_dict['volume_1h']}\n"
        f"• Volatility (1h): {signal_dict['volatility_1h']}%\n\n"
    )

    analysis_logic = load_prompt_template()
    account_size = get_current_balance()
    max_leverage = int(os.getenv("MAX_LEVERAGE_SMALL", "3"))
    max_loss = account_size * 0.015

    response_format = (
        "\nRESPONSE FORMAT:\n"
        "• Entry: [price]\n"
        "• SL: [price]\n"
        "• TP: [price]\n"
        f"• Size: [quantity for ${account_size} account]\n"
        "• Risk: [percentage of account risked]\n"
        "• Reason: [1 sentence why this is a good/bad trade]\n"
        "• Trap Risk: [High/Medium/Low - based on orderbook imbalances]\n"
    )

    prompt = intro + analysis_logic + response_format
    prompt = prompt.format(
        account_size=account_size,
        max_leverage=max_leverage,
        max_loss=f"{max_loss:.2f}"
    )

    # --- Send to DeepSeek ---
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('DEEP_SEEK_API_KEY')}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "user", "content": f"Candles (CSV format):\n{csv_content}"},
                {"role": "user", "content": f"Orderbook:\n{orderbook_content}"}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
    )
    
    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        return {"analysis": content, "approved": True}
    return {"analysis": "LLM analysis failed", "approved": False}


def parse_llm_response(llm_analysis: str, original_signal: dict) -> dict:
    """Parse LLM response to extract entry/SL/TP"""
    # Extract prices from LLM response using regex or string parsing
    # Look for price patterns in the LLM response
    # Example: "entry zone (0.6320–0.6340)" or "SL above 0.6350"
    entry_match = re.search(r'entry.*?(\d+\.\d+)', llm_analysis.lower())
    sl_match = re.search(r'sl.*?(\d+\.\d+)', llm_analysis.lower())
    tp_match = re.search(r'tp.*?(\d+\.\d+)', llm_analysis.lower())
    
    if entry_match and sl_match and tp_match:
        return {
            "symbol": original_signal['asset'],
            "side": "LONG" if original_signal['signal'] == 1 else "SHORT",
            "entry": float(entry_match.group(1)),
            "stop_loss": float(sl_match.group(1)),
            "take_profit": float(tp_match.group(1)),
            "confidence": original_signal['confidence_percent']
        }
    return None


def process_signal():
    while True:
        """Process incoming signal from Api bot with combined CSV + orderbook file"""
        # Only proceed if bot is running
        if not get_bot_status():
            logger.info("Bot is paused. Waiting to resume...")
            time.sleep(10)
            continue

        # Get signal from Mockba ML
        URL = "https://signal.globaldv.net/api/v1/signals/active?venue=CEX"
        # The API is free, get no post
        response = requests.get(URL)
        if response.status_code != 200:
            logger.error(f"Failed to fetch signals: {response.status_code}")
            time.sleep(30)
            continue
        
        signals = response.json()  # List of signal dicts

        # Compare with Redis to avoid duplicates
        if redis_client:
            stored = redis_client.get("active_signals")
            if stored:
                stored_signals = json.loads(stored)
                if stored_signals == signals:
                    logger.info("Signals unchanged, skipping processing...")
                    time.sleep(30)
                    continue
            # Store new signals (no expiration)
            redis_client.set("active_signals", json.dumps(signals))
        else:
            logger.warning("Redis not available, skipping deduplication")

        # Process the single signal (API always returns one)
        if signals:
            signal = signals[0]
            
            # Get confidence level
            confidence_level = get_confidence_level(signal['confidence_percent'])
            
            # Only proceed if confidence is moderate or higher
            if confidence_level == "❌ WEAK":
                logger.info(f"Skipping weak signal for {signal['asset']}")
                time.sleep(30)
                continue
            
            # --- MICRO BACKTEST CHECK ---
            bt = signal.get('backtest', {})
            
            # Must have positive expectancy and enough trades
            if bt.get("trades", 0) < 15 or bt.get("exp", 0.0) <= MICRO_BACKTEST_MIN_EXPECTANCY:
                logger.info(f"❌ {signal['asset']} micro-backtest failed: {bt}")
                time.sleep(30)
                continue
            
            logger.info(f"✅ {signal['asset']} micro-backtest passed: {bt}")
            
            # Get leverage based on confidence
            leverage = get_leverage_by_confidence(signal['confidence_percent'])
            
            # --- LIQUIDITY PERSISTENCE CHECK ---
            cex_check = lpm.validate_cex_consensus_for_dex_asset(signal['asset'])
            if cex_check["consensus"] == "NO_CEX_PAIR":
                logger.info(f"🛑 {signal['asset']} CEX consensus check failed: {cex_check['reason']}")
                time.sleep(30)
                continue
            elif cex_check["consensus"] == "LOW":
                logger.info(f"❌ Skipping {signal['asset']}: LOW CEX consensus ({cex_check['reason']})")
                time.sleep(30)
                continue
            else:
                logger.info(f"✅ {signal['asset']} passed CEX consensus: {cex_check['reason']}")
            
            # Analyze with LLM
            llm_result = analyze_with_llm(signal)
            
            if not llm_result["approved"]:
                logger.info(f"LLM rejected signal for {signal['asset']}: {llm_result['analysis']}")
                time.sleep(30)
                continue
            
            # PARSE LLM RESULT to extract entry/SL/TP
            parsed_signal = parse_llm_response(llm_result["analysis"], signal)
            if not parsed_signal:
                logger.info(f"Could not parse LLM response for {signal['asset']}")
                time.sleep(30)
                continue

            # 👇 ADD LEVERAGE TO PARSED SIGNAL
            parsed_signal['leverage'] = leverage
            
            # Execute position using your existing executor
            execution_result = place_futures_order(parsed_signal)
            
            logger.info(f"Execution result for {signal['asset']}: {execution_result}")

        # Sleep for 30 seconds before next fetch
        time.sleep(30)


# Position monitoring thread (same as before)
def position_monitor_loop():
    """Continuously monitor all open positions."""
    logger.info("🚀 Starting position monitor...")
    
    while True:
        try:
            open_positions = get_open_positions()
            
            if open_positions:
                logger.info(f"Monitoring {len(open_positions)} open positions")
                
                for pos in open_positions:
                    closed_info = update_position_pnl(pos['id'], pos)
                    if closed_info:
                        emoji = "🟢" if closed_info['pnl_usd'] >= 0 else "🔴"
                        message = (
                            f"{emoji} POSITION UPDATE on BINANCE\n"
                            f"Symbol: {closed_info['symbol']}\n"
                            f"Side: {closed_info['side'].upper()}\n"
                            f"Fill Price: {closed_info['fill_price']:.4f}\n"
                            f"Current Price: {closed_info['current_price']:.4f}\n"
                            f"PnL: {closed_info['pnl_pct']:.4f}% | ${closed_info['pnl_usd']:.4f}"
                        )
                        send_bot_message(int(os.getenv("TELEGRAM_CHAT_ID")), message)

                    time.sleep(0.1)

            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Position monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Position monitor error: {e}")
            time.sleep(5)



if __name__ == "__main__":
    # Check for tables
    initialize_database_tables()

    # # start monitoring
    monitor_thread = threading.Thread(target=position_monitor_loop, daemon=True)
    monitor_thread.start()