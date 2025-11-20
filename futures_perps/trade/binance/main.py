from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import json
import io
import requests
import os
from datetime import datetime
import threading
import time
import sys
import re
import uvicorn
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from db.db_ops import get_open_positions, update_position_pnl, initialize_database_tables
from logs.log_config import binance_trader_logger as logger
from binance.client import Client as BinanceClient
from trading_bot.send_bot_message import send_bot_message


app = FastAPI()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


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

def analyze_with_llm(signal_dict: dict, csv_content: str, orderbook_content: str) -> dict:
    """Send to LLM for detailed analysis using fixed prompt structure."""
    
    # --- 1. Static mandatory intro ---
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

    # --- 2. Load middle section (TASKS + RULES) from .txt ---
    analysis_logic = load_prompt_template()  # reads only the TASKS+RULES part

    # --- 3. Static mandatory response format ---
    account_size = get_current_balance()
    max_leverage = int(os.getenv("MAX_LEVERAGE_SMALL", "3"))
    max_loss = account_size * 0.015  # 1.5% risk

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

    # --- Combine all three parts ---
    prompt = intro + analysis_logic + response_format

    # --- Inject dynamic values into the middle section ---
    # Since analysis_logic contains placeholders like {account_size}, we format the whole prompt
    prompt = prompt.format(
        account_size=account_size,
        max_leverage=max_leverage,
        max_loss=f"{max_loss:.2f}"
    )

    # --- Send to DeepSeek ---
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('DEEP_SEEK_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "user", "content": f"Candle \n{csv_content}"},
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


def parse_llm_response(llm_analysis: str, original_signal: TradingSignal) -> dict:
    """Parse LLM response to extract entry/SL/TP"""
    # Extract prices from LLM response using regex or string parsing
    import re
    
    # Look for price patterns in the LLM response
    # Example: "entry zone (0.6320–0.6340)" or "SL above 0.6350"
    entry_match = re.search(r'entry.*?(\d+\.\d+)', llm_analysis.lower())
    sl_match = re.search(r'sl.*?(\d+\.\d+)', llm_analysis.lower())
    tp_match = re.search(r'tp.*?(\d+\.\d+)', llm_analysis.lower())
    
    if entry_match and sl_match and tp_match:
        return {
            "symbol": original_signal.asset,
            "side": original_signal.signal,
            "entry": float(entry_match.group(1)),
            "stop_loss": float(sl_match.group(1)),
            "take_profit": float(tp_match.group(1)),
            "confidence": original_signal.confidence
        }
    return None

# Add this function to your main.py
def micro_backtest(
    df: pd.DataFrame,
    direction: int,
    tp_pct: float = 0.006,
    sl_pct: float = 0.008,
    max_bars: int = 8,
    fee_bps: float = 0.0,
    slip_bps: float = 0.0,
):
    """
    direction: 1 (long) or -1 (short).
    Entry at close[t]. Exit when TP/SL is touched using high/low of subsequent bars.
    Costs: fee_bps and slip_bps are round-trip costs, in basis points (e.g., 8 = 0.08%).
    Returns: dict(trades, winrate, avg_ret, exp, max_dd) using *net* returns after costs.
    """
    import pandas as pd
    import numpy as np
    
    if direction == 0 or df.shape[0] < (max_bars + 2):
        return {"trades": 0, "winrate": 0.0, "avg_ret": 0.0, "exp": 0.0, "max_dd": 0.0}

    closes = pd.to_numeric(df["close"], errors="coerce").values
    highs  = pd.to_numeric(df["high"],  errors="coerce").values
    lows   = pd.to_numeric(df["low"],   errors="coerce").values

    trades = 0
    wins = 0
    rets = []
    equity = 1.0
    peak = 1.0
    max_dd = 0.0

    # convert bps → decimal cost per round trip
    round_trip_cost = (float(fee_bps) + float(slip_bps)) * 1e-4

    stride = 3
    last_idx = len(closes) - max_bars - 1
    for t in range(0, last_idx, stride):
        entry = float(closes[t])

        if direction == 1:
            tp = entry * (1 + tp_pct)
            sl = entry * (1 - sl_pct)
        else:
            tp = entry * (1 - tp_pct)
            sl = entry * (1 + sl_pct)

        exit_px = None

        for k in range(1, max_bars + 1):
            i = t + k
            if i >= len(closes):
                break

            if direction == 1:
                if highs[i] >= tp:
                    exit_px = tp; break
                if lows[i]  <= sl:
                    exit_px = sl; break
            else:
                if lows[i]  <= tp:
                    exit_px = tp; break
                if highs[i] >= sl:
                    exit_px = sl; break

        if exit_px is None:
            exit_px = float(closes[min(t + max_bars, len(closes) - 1)])

        gross_ret = ((exit_px - entry) / entry) * direction
        net_ret = gross_ret - round_trip_cost

        trades += 1
        wins += int(net_ret > 0)
        rets.append(net_ret)

        equity *= (1 + net_ret)
        peak = max(peak, equity)
        max_dd = min(max_dd, (equity / peak - 1.0))

    winrate = wins / trades if trades else 0.0
    avg_ret = float(np.mean(rets)) if rets else 0.0
    exp = winrate * tp_pct - (1 - winrate) * sl_pct  # expectancy proxy (unchanged)

    return {
        "trades": trades,
        "winrate": round(winrate, 3),
        "avg_ret": round(avg_ret, 4),
        "exp": round(exp, 4),
        "max_dd": round(max_dd, 4),
    }

#helper
def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Create a unified 'ts' (UTC) column and sort ascending by it."""
    d = df.copy()

    # Try columns in priority order
    ts = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")

    if "timestamp" in d.columns:
        ts = ts.fillna(pd.to_datetime(d["timestamp"], utc=True, errors="coerce"))

    if "time" in d.columns:
        # seconds or ms → datetime
        t = pd.to_numeric(d["time"], errors="coerce")
        if t.notna().any():
            unit = "ms" if t.dropna().median() > 1e12/2 else "s"
            ts = ts.fillna(pd.to_datetime(t, unit=unit, utc=True, errors="coerce"))

    if "start_time" in d.columns:
        t = pd.to_numeric(d["start_time"], errors="coerce")
        if t.notna().any():
            unit = "ms" if t.dropna().median() > 1e12/2 else "s"
            ts = ts.fillna(pd.to_datetime(t, unit=unit, utc=True, errors="coerce"))

    # DatetimeIndex as fallback
    if isinstance(d.index, pd.DatetimeIndex):
        index_ts = d.index.tz_convert("UTC") if d.index.tz is not None else d.index.tz_localize("UTC")
        ts[ts.isna()] = index_ts[ts.isna()]

    if ts.isna().all():
        raise ValueError("No usable time column found for klines")

    d["ts"] = ts
    d = d.dropna(subset=["ts"])
    # keep last if duplicates in the same ts
    d = d.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    return d

def take_recent(df: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    d = _normalize_ts(df)
    return d.tail(n)

# Then modify your endpoint to include the backtest:
@app.post("/process_binance_signal/")
async def process_signal(
    signal: TradingSignal,
    combined_file: UploadFile = File(...)
):
    """Process incoming signal from Telegram bot with combined CSV + orderbook file"""
    
    # Read the combined file
    combined_content = await combined_file.read()
    combined_str = combined_content.decode('utf-8')
    
    # Split the content at the separator
    if "# ORDERBOOK_SNAPSHOT" in combined_str:
        csv_part, orderbook_part = combined_str.split("# ORDERBOOK_SNAPSHOT", 1)
        csv_content = csv_part.strip()
        orderbook_content = orderbook_part.strip()
    else:
        return {"status": "error", "reason": "No orderbook separator found in file"}
    
    # Parse CSV to DataFrame
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        if df.empty or len(df) < 20:  # Minimum candles needed
            return {"status": "error", "reason": "Insufficient candle data in CSV"}
    except Exception as e:
        return {"status": "error", "reason": f"Invalid CSV format: {str(e)}"}
    
    # Get confidence level
    confidence_level = get_confidence_level(signal.confidence)
    
    # Only proceed if confidence is moderate or higher
    if confidence_level == "❌ WEAK":
        return {"status": "rejected", "reason": "Low confidence signal"}
    
    # --- MICRO-BACKTEST VALIDATION ---
    # Convert signal direction to 1/-1
    direction = 1 if signal.signal.lower() == "long" else -1
    
    bt = micro_backtest(
        take_recent(df, 80), 
        direction=direction,
        tp_pct=0.006,  # 0.6% target
        sl_pct=0.008,  # 0.8% stop
        max_bars=8,    # max 8 bars to close
        fee_bps=10,    # 0.1% fees
        slip_bps=6     # 0.06% slippage
    )
    
    # Must have positive expectancy and enough trades
    if bt.get("trades", 0) < 15 or bt.get("exp", 0.0) <= MICRO_BACKTEST_MIN_EXPECTANCY:
        print(f"❌ {signal.asset} micro-backtest failed: {bt}")
        return {
            "status": "rejected", 
            "reason": f"Micro-backtest failed - trades: {bt.get('trades', 0)}, exp: {bt.get('exp', 0.0)}"
        }
    
    print(f"✅ {signal.asset} micro-backtest passed: {bt}")
    
    # Get leverage based on confidence
    leverage = get_leverage_by_confidence(signal.confidence)
    
    # --- LIQUIDITY PERSISTENCE CHECK ---
    cex_check = lpm.validate_cex_consensus_for_dex_asset(signal.asset)
    if cex_check["consensus"] == "NO_CEX_PAIR":
        print(f"🛑 {signal.asset} CEX consensus check failed: {cex_check['reason']}")
        return {"status": "rejected", "reason": f"CEX consensus failed: {cex_check['reason']}"}
    elif cex_check["consensus"] == "LOW":
        print(f"❌ Skipping {signal.asset}: LOW CEX consensus ({cex_check['reason']})")
        return {"status": "rejected", "reason": f"Low CEX consensus: {cex_check['reason']}"}
    else:
        print(f"✅ {signal.asset} passed CEX consensus: {cex_check['reason']}")
    
    # Analyze with LLM
    llm_result = analyze_with_llm(signal.model_dump(), csv_content, orderbook_content)
    
    if not llm_result["approved"]:
        return {"status": "rejected", "reason": "LLM analysis failed"}
    
    # PARSE LLM RESULT to extract entry/SL/TP
    parsed_signal = parse_llm_response(llm_result["analysis"], signal)
    if not parsed_signal:
        return {"status": "rejected", "reason": "Could not parse LLM response for entry/SL/TP"}
    
    # Execute position using your existing executor
    execution_result = place_futures_order(parsed_signal)
    
    return {
        "status": "executed" if execution_result else "failed",
        "confidence_level": confidence_level,
        "leverage": leverage,
        "execution": execution_result,
        "llm_analysis": llm_result["analysis"],
        "parsed_signal": parsed_signal,
        "cex_check": cex_check,
        "micro_backtest": bt
    }


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

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("APP_PORT", 8000)))