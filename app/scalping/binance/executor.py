from binance import Client  # pip install python-binance
from .utils import calc_spread_pct
from typing import Dict
import time

class ScalpExecutor:
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)
        self.active_orders = set()

    def place_scalp_order(self, signal: Dict, max_wait_sec: int = 3):
        symbol = signal['symbol'].replace('/', '')
        side = signal['side'].upper()
        price = signal['entry']
        qty = self._calc_qty(symbol, price)  # e.g., $2 worth

        # Place aggressive limit (for buy: at ask or slightly above)
        if side == "BUY":
            ob = self.client.get_order_book(symbol=symbol, limit=1)
            limit_price = float(ob['asks'][0][0])  # hit the ask
        else:
            ob = self.client.get_order_book(symbol=symbol, limit=1)
            limit_price = float(ob['bids'][0][0])  # hit the bid

        order = self.client.create_order(
            symbol=symbol,
            side=side,
            type='LIMIT',
            timeInForce='GTC',
            quantity=qty,
            price='{:.8f}'.format(limit_price)
        )

        # TODO: monitor fill, place TP/SL as OCO or separate limit+stop
        # For now: cancel if not filled in max_wait_sec
        time.sleep(max_wait_sec)
        if order['status'] != 'FILLED':
            self.client.cancel_order(symbol=symbol, orderId=order['orderId'])

    def _calc_qty(self, symbol: str, price: float) -> float:
        # Allocate $2 per trade from $20 capital
        usdt_value = 2.0
        qty = usdt_value / price
        # Round to step size (you’ll need exchange info for this)
        return round(qty, 5)  # crude — improve with exchange info