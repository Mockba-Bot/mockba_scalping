import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# DB Config from .env
DB_CONFIG = {
    'host': os.getenv('HOST'),
    'database': os.getenv('DATABASE'),
    'user': os.getenv('DATABASE_USR'),
    'password': os.getenv('PASSWD'),
    'port': 5432
}


def get_db_connection():
    """Establish a new DB connection."""
    return psycopg2.connect(**DB_CONFIG)

# ────────────────────────────────────────
# POSITION OPERATIONS
# ────────────────────────────────────────

def insert_position_with_orders(chat_id: int, signal: dict, order_result: dict, exchange: str = "BINANCE"):
    """Insert position with Binance order IDs."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
            INSERT INTO scalp_positions (
                chat_id, symbol, side, entry_price, stop_loss, take_profit,
                quantity, notional_usd,
                entry_order_id, tp_order_id, sl_order_id, exchange
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id;
        """
        cur.execute(query, (
            chat_id,
            signal['symbol'],
            signal['side'].upper(),
            float(signal['entry']),  # initial estimate
            float(signal['stop_loss']),
            float(signal['take_profit']),
            float(order_result['quantity']),
            float(order_result['notional']),
            order_result['entry_order_id'],
            order_result['tp_order_id'],
            order_result['sl_order_id']
        ))
        pos_id = cur.fetchone()[0]
        conn.commit()
        print(f"Inserted position ID {pos_id} with order IDs")
        return pos_id
    except Exception as e:
        print(f"DB insert failed: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()

def update_position_pnl(position_id: int, pnl_pct: float, pnl_usd: float, 
                       status: str = None, fill_price: float = None):
    """Update PnL and optionally fill_price/status."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        if status or fill_price:
            # Update multiple fields
            fields = []
            values = []
            
            if fill_price is not None:
                fields.append("fill_price = %s")
                values.append(fill_price)
                
            if status:
                fields.append("status = %s")
                values.append(status)
                
            fields.append("current_pnl_pct = %s")
            fields.append("current_pnl_usd = %s")
            fields.append("updated_at = NOW()")
            values.extend([pnl_pct, pnl_usd, position_id])
            
            query = f"""
                UPDATE scalp_positions 
                SET {', '.join(fields)}
                WHERE id = %s;
            """
            cur.execute(query, values)
        else:
            # Only update PnL
            cur.execute("""
                UPDATE scalp_positions
                SET current_pnl_pct = %s, current_pnl_usd = %s, updated_at = NOW()
                WHERE id = %s;
            """, (pnl_pct, pnl_usd, position_id))
            
        conn.commit()
        print(f"Updated PnL for position {position_id}")
    except Exception as e:
        print(f"Failed to update PnL for position {position_id}: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# In db_ops.py
def get_open_positions():
    """Get all OPEN positions from DB."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, chat_id, symbol, side, entry_price, stop_loss, take_profit,
                   quantity, notional_usd, entry_order_id, tp_order_id, sl_order_id
            FROM scalp_positions 
            WHERE status = 'OPEN'
            ORDER BY created_at DESC;
        """)
        return cur.fetchall()
    except Exception as e:
        print(f"Failed to get open positions: {e}")
        return []
    finally:
        if conn:
            conn.close()
# ────────────────────────────────────────
# SETTINGS OPERATIONS
# ────────────────────────────────────────

def ensure_chat_settings(chat_id: int):
    """Ensure a row exists for chat_id (idempotent)."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
            INSERT INTO scalp_settings (chat_id)
            VALUES (%s)
            ON CONFLICT (chat_id) DO NOTHING;
        """
        cur.execute(query, (chat_id,))
        conn.commit()
    except Exception as e:
        print(f"Failed to ensure settings for chat {chat_id}: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def set_signals_enabled(chat_id: int, enabled: bool):
    """Enable or disable signals for a chat."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
            INSERT INTO scalp_settings (chat_id, signals_enabled)
            VALUES (%s, %s)
            ON CONFLICT (chat_id)
            DO UPDATE SET signals_enabled = %s, updated_at = NOW();
        """
        cur.execute(query, (chat_id, enabled, enabled))
        conn.commit()
        print(f"Set signals_enabled={enabled} for chat {chat_id}")
    except Exception as e:
        print(f"Failed to update settings for chat {chat_id}: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def get_all_signal_statuses():
    """
    Retrieve all chat IDs and their signal enabled status.
    Returns a list of dicts: [{'chat_id': 123, 'signals_enabled': True}, ...]
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)  # Returns dict-like rows
        cur.execute("SELECT chat_id, signals_enabled FROM scalp_settings ORDER BY chat_id;")
        rows = cur.fetchall()
        # Convert to list of dicts
        result = [
            {
                'chat_id': row['chat_id'],
                'signals_enabled': row['signals_enabled']
            }
            for row in rows
        ]
        print(f"Fetched signal status for {len(result)} chats")
        return result
    except Exception as e:
        print(f"Failed to fetch all signal statuses: {e}")
        return []
    finally:
        if conn:
            conn.close()