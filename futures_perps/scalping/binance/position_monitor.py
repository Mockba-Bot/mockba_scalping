# position_monitor.py
import time
import logging
from binance.client import Client
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from db.db_ops import get_open_positions, update_position_pnl

# Setup Binance client (same as your scanner)
load_dotenv()
client = Client(
    api_key=os.getenv("BINANCE_API_KEY"),
    api_secret=os.getenv("BINANCE_SECRET_KEY"),
    testnet=False
)


def monitor_positions():
    
    while True:
        try:
            # Get all OPEN positions from DB
            open_positions = get_open_positions()
            
            if not open_positions:
                time.sleep(2)  # No positions, check less frequently
                continue
                 
            for pos in open_positions:
                try:
                    update_position_from_binance(pos['id'], pos)
                except Exception as e:
                   print(f"Error updating position {pos['id']}: {e}")
                
                # Respect Binance rate limits (1200 requests/minute)
                time.sleep(0.1)  # 100ms between position checks
                
        except KeyboardInterrupt:
            print("Position monitor stopped by user")
            break
        except Exception as e:
            print(f"Monitor loop error: {e}")
            time.sleep(5)  # Recover from errors

def update_position_from_binance(position_id: int, db_row: dict):
    """Your existing function - keep it here"""
    try:
        # Get actual fill price from entry order
        order_info = client.futures_get_order(
            symbol=db_row['symbol'],
            orderId=db_row['entry_order_id']
        )
        
        if order_info['status'] == 'FILLED':
            fill_price = float(order_info['avgPrice'])
            qty = float(order_info['executedQty'])
            
            # Calculate current PnL
            current_price = float(client.futures_symbol_ticker(symbol=db_row['symbol'])['price'])
            if db_row['side'] == 'BUY':
                pnl_pct = (current_price - fill_price) / fill_price * 100
            else:
                pnl_pct = (fill_price - current_price) / fill_price * 100
                
            pnl_usd = (pnl_pct / 100) * db_row['notional_usd']
            
            # Update DB
            update_position_pnl(
                position_id,
                pnl_pct=pnl_pct,
                pnl_usd=pnl_usd,
                fill_price=fill_price
            )
            
            # Check if TP/SL hit
            tp_info = client.futures_get_order(symbol=db_row['symbol'], orderId=db_row['tp_order_id'])
            sl_info = client.futures_get_order(symbol=db_row['symbol'], orderId=db_row['sl_order_id'])
            
            if tp_info['status'] == 'FILLED':
                update_position_pnl(position_id, pnl_pct, pnl_usd, status='TP')
            elif sl_info['status'] == 'FILLED':
                update_position_pnl(position_id, pnl_pct, pnl_usd, status='SL')
                
    except Exception as e:
        print(f"Error updating position {position_id}: {e}")