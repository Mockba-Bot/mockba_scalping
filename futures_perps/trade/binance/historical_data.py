import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from time import time
from typing import Dict, Optional
import requests
import pandas as pd
import numpy as np
from logs.log_config import binance_trader_logger as logger

# === Binance Futures (USDT-M) Config ===
BASE_URL = "https://fapi.binance.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

base_features = ["close", "high", "low", "volume"]

strategy_features = {
    "5m": {
        "Trend-Following": {"features": base_features + ["ema_12", "ema_26", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
        "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
        "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Hybrid": {"features": base_features + ["ema_12", "ema_26", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
        "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
        "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
    },
    "15m": {
        "Trend-Following": {"features": base_features + ["ema_20", "ema_40", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
        "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
        "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_14", "momentum_14", "vwap"], "force_features": True},
        "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_14", "momentum_14", "vwap"], "force_features": True},
        "Hybrid": {"features": base_features + ["ema_20", "ema_40", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
        "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
        "Router": {"features": ["ema_20", "ema_40", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_14", "momentum_14", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
    },
    "30m": {
        "Trend-Following": {"features": base_features + ["ema_30", "ema_60", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
        "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
        "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_20", "momentum_20", "vwap"], "force_features": True},
        "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_20", "momentum_20", "vwap"], "force_features": True},
        "Hybrid": {"features": base_features + ["ema_30", "ema_60", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
        "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
        "Router": {"features": ["ema_30", "ema_60", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_20", "momentum_20", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
    },
    "1h": {
        "Trend-Following": {"features": base_features + ["ema_20", "ema_50", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
        "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
        "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Hybrid": {"features": base_features + ["ema_20", "ema_50", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
        "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
        "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
    },
    "4h": {
        "Trend-Following": {"features": base_features + ["ema_50", "ema_200", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
        "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
        "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Hybrid": {"features": base_features + ["ema_50", "ema_200", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
        "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
        "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
    },
    "1d": {
        "Trend-Following": {"features": base_features + ["ema_50", "ema_200", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
        "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
        "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Hybrid": {"features": base_features + ["ema_50", "ema_200", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
        "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
        "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
    }
}

def add_indicators(data, required_features):
    """
    Add only the necessary indicators to the data based on the requested features.
    """
    data[['close', 'high', 'low', 'volume']] = data[['close', 'high', 'low', 'volume']].apply(pd.to_numeric)

    # --- EMA ---
    for feature in required_features:
        if feature.startswith("ema_"):
            try:
                window = int(feature.split("_")[1])
                data[feature] = data['close'].ewm(span=window, adjust=False).mean()
            except (IndexError, ValueError):
                logger.warning(f"⚠️ Could not extract window for feature: {feature}")

    # --- MACD ---
    if any(x in required_features for x in ['macd', 'macd_signal']):
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

    # --- ATR ---
    for feature in required_features:
        if feature.startswith("atr_"):
            try:
                window = int(feature.split("_")[1])
                data['tr'] = pd.concat([
                    data['high'] - data['low'],
                    (data['high'] - data['close'].shift()).abs(),
                    (data['low'] - data['close'].shift()).abs()
                ], axis=1).max(axis=1)
                data[feature] = data['tr'].rolling(window=window).mean()
            except (IndexError, ValueError):
                logger.warning(f"⚠️ Could not extract window for feature: {feature}")

    # --- Bollinger Bands ---
    if any(f in required_features for f in ['bollinger_hband', 'bollinger_lband']):
        window = 20
        data['bollinger_mavg'] = data['close'].rolling(window=window).mean()
        data['bollinger_std'] = data['close'].rolling(window=window).std()
        data['bollinger_hband'] = data['bollinger_mavg'] + (data['bollinger_std'] * 2)
        data['bollinger_lband'] = data['bollinger_mavg'] - (data['bollinger_std'] * 2)

    # --- Standard Deviation ---
    for feature in required_features:
        if feature.startswith("std_"):
            try:
                window = int(feature.split("_")[1])
                data[feature] = data['close'].rolling(window=window).std()
            except (IndexError, ValueError):
                logger.warning(f"⚠️ Could not extract window for feature: {feature}")

    # --- RSI ---
    for feature in required_features:
        if feature.startswith("rsi_"):
            try:
                window = int(feature.split("_")[1])
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=window).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
                rs = gain / loss
                data[feature] = 100 - (100 / (1 + rs))
            except (IndexError, ValueError):
                logger.warning(f"⚠️ Could not extract window for feature: {feature}")

    # --- Stochastic Oscillator ---
    stoch_done = set()
    for feature in required_features:
        if feature.startswith("stoch_"):
            try:
                window = int(feature.split("_")[-1])
                if window in stoch_done:
                    continue
                stoch_k = ((data['close'] - data['low'].rolling(window).min()) /
                           (data['high'].rolling(window).max() - data['low'].rolling(window).min())) * 100
                stoch_d = stoch_k.rolling(3).mean()
                data[f'stoch_k_{window}'] = stoch_k
                data[f'stoch_d_{window}'] = stoch_d
                stoch_done.add(window)
            except (IndexError, ValueError):
                logger.warning(f"⚠️ Could not extract window for feature: {feature}")

    # --- Momentum ---
    for feature in required_features:
        if feature.startswith("momentum_"):
            try:
                window = int(feature.split("_")[1])
                data[feature] = data['close'].diff(periods=window)
            except (IndexError, ValueError):
                logger.warning(f"⚠️ Could not extract window for feature: {feature}")

    # --- Rate of Change (ROC) ---
    for feature in required_features:
        if feature.startswith("roc_"):
            try:
                window = int(feature.split("_")[1])
                data[feature] = data['close'].pct_change(periods=window) * 100
            except (IndexError, ValueError):
                logger.warning(f"⚠️ Could not extract window for feature: {feature}")

    # --- ADX ---
    for feature in required_features:
        if feature.startswith("adx"):
            try:
                window = int(feature.split("_")[1]) if "_" in feature else 14
                data['plus_dm'] = data['high'].diff().where(lambda x: x > 0, 0)
                data['minus_dm'] = -data['low'].diff().where(lambda x: x < 0, 0)
                data['tr'] = pd.concat([
                    data['high'] - data['low'],
                    (data['high'] - data['close'].shift()).abs(),
                    (data['low'] - data['close'].shift()).abs()
                ], axis=1).max(axis=1)
                data['plus_di'] = 100 * (data['plus_dm'].rolling(window=window).mean() / data['tr'].rolling(window=window).mean())
                data['minus_di'] = 100 * (data['minus_dm'].rolling(window=window).mean() / data['tr'].rolling(window=window).mean())
                data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
                data[feature] = data['dx'].rolling(window=window).mean()
            except (IndexError, ValueError) as e:
                logger.warning(f"⚠️ ADX error for {feature}: {e}")

    # --- Ichimoku Cloud ---
    for feature in required_features:
        if feature.startswith("tenkan_sen_"):
            try:
                window = int(feature.split("_")[-1])
                data[feature] = (data['high'].rolling(window=window).max() + data['low'].rolling(window=window).min()) / 2
            except (IndexError, ValueError):
                logger.warning(f"⚠️ Could not extract window for feature: {feature}")
        if feature.startswith("kijun_sen_"):
            try:
                window = int(feature.split("_")[-1])
                data[feature] = (data['high'].rolling(window=window).max() + data['low'].rolling(window=window).min()) / 2
            except (IndexError, ValueError):
                logger.warning(f"⚠️ Could not extract window for feature: {feature}")
        if feature.startswith("senkou_span_a"):
            data[feature] = ((data['tenkan_sen_9'] + data['kijun_sen_26']) / 2).shift(26)
        if feature.startswith("senkou_span_b"):
            data[feature] = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)

    # --- Parabolic SAR ---
    if 'sar' in required_features:
        data['sar'] = np.nan
        af = 0.02
        max_af = 0.2
        ep = data['high'].iloc[0]
        sar = data['low'].iloc[0]
        trend = 1
        for i in range(1, len(data)):
            prev_sar = sar
            sar = prev_sar + af * (ep - prev_sar)
            if trend == 1:
                if data['low'].iloc[i] < sar:
                    trend = -1
                    sar = ep
                    ep = data['low'].iloc[i]
                    af = 0.02
            else:
                if data['high'].iloc[i] > sar:
                    trend = 1
                    sar = ep
                    ep = data['high'].iloc[i]
                    af = 0.02
            if af < max_af:
                af += 0.02
            data.loc[data.index[i], 'sar'] = sar

    # --- VWAP ---
    if 'vwap' in required_features:
        data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()

    # Clean NaNs safely (NO backfill to avoid leakage)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.ffill(inplace=True)
    data.dropna(inplace=True)

    # --- RSI 14 (needed for entropy) ---
    if 'rsi_14' not in data.columns:
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        data['rsi_14'] = 100 - (100 / (1 + rs))


    return data


def get_features_for_strategy(interval, strategy):
    strategy_info = strategy_features.get(interval, {}).get(strategy, {})
    return {
        "interval": interval,
        "strategy": strategy,
        "features": strategy_info.get("features", []),
        "force_features": strategy_info.get("force_features", False)
    }

def get_funding_rate_history(symbol: str, limit: int = 1000):
    """
    Get funding rate history from Binance Futures
    """
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {
        "symbol": symbol,
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Binance returns list of funding rate records
        # Format: [{'symbol': 'BTCUSDT', 'fundingRate': '0.00010000', 'fundingTime': 1631232000000}, ...]
        return data if isinstance(data, list) else []
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Binance funding rate for {symbol}: {e}")
        return []

def get_funding_rate_current(symbol: str):
    """
    Get current funding rate from Binance Futures
    """
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    params = {"symbol": symbol}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Binance returns current funding rate in this endpoint
        # Format: {'symbol': 'BTCUSDT', 'markPrice': '50000.00', 'indexPrice': '50000.00', 
        #          'estimatedSettlePrice': '50000.00', 'lastFundingRate': '0.00010000', ...}
        return {
            'current_rate': float(data.get('lastFundingRate', 0)),
            'next_funding_time': data.get('nextFundingTime'),
            'mark_price': float(data.get('markPrice', 0))
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching current Binance funding rate for {symbol}: {e}")
        return {'current_rate': 0, 'next_funding_time': None, 'mark_price': 0}

def get_historical_data_limit_binance(pair: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
    """
    Fetch historical klines from Binance Futures (USDT-M) API.
    Mimics the interface and output of `get_historical_data_limit`, but uses Binance API instead of DB.
    
    Args:
        pair (str): e.g., "BTCUSDT"
        timeframe (str): Binance interval (e.g., "1h", "15m", "4h")
        limit (int): Number of klines to fetch (max 1000)
    
    Returns:
        pd.DataFrame with columns: start_timestamp, open, high, low, close, volume
        sorted chronologically (oldest → newest), or None if failed.
    """
    # Map common timeframe aliases if needed (optional)
    # Binance uses: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": pair,
        "interval": timeframe,
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return None

        # Parse klines: select only needed columns
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "_", "_", "_", "_", "_", "_"
        ])
        
        # Keep only required columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        
        # Convert timestamp (in milliseconds) to datetime
        df['start_timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert price/volume columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # ✅ SORT CHRONOLOGICALLY (OLDEST → NEWEST)
        df = df.sort_values('start_timestamp').reset_index(drop=True)

        features_dict = get_features_for_strategy(timeframe, "Hybrid")
        features = features_dict["features"]
        
        if not features:
            print(f"⚠️ Warning: No features defined for interval: {timeframe} and strategy: Hybrid")
            raise ValueError(f"No features defined for interval: {timeframe} and strategy: Hybrid")
        
        df = add_indicators(df, features)
        
        return df

    except Exception as e:
        print(f"❌ Failed to fetch Binance klines for {pair} ({timeframe}): {e}")
        return None

def get_orderbook(symbol: str, limit: int = 5) -> Optional[Dict]:
    url = f"{BASE_URL}/fapi/v1/depth"
    try:
        resp = requests.get(url, headers=HEADERS, params={"symbol": symbol, "limit": limit}, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_public_liquidations(symbol: str = None, lookback_hours: int = 24):
    """
    Get liquidation data from Binance Futures
    Note: Binance doesn't have direct liquidation history API, 
    so we use the open interest and force orders as alternative
    """
    # Method 1: Get forced orders (partial liquidation data)
    url = "https://fapi.binance.com/fapi/v1/forceOrders"
    
    end_time = int(time() * 1000)
    start_time = end_time - (lookback_hours * 3600 * 1000)
    
    params = {
        "startTime": start_time,
        "endTime": end_time,
    }
    if symbol:
        params["symbol"] = symbol
        
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Binance force orders include liquidations
        # Format: [{"orderId": 123, "symbol": "BTCUSDT", "status": "FILLED", 
        #           "clientOrderId": "abc", "price": "50000", "avgPrice": "50000",
        #           "origQty": "1", "executedQty": "1", "cumQuote": "50000",
        #           "timeInForce": "GTC", "type": "LIMIT", "reduceOnly": False,
        #           "closePosition": False, "side": "SELL", "positionSide": "LONG",
        #           "stopPrice": "0", "workingType": "CONTRACT_PRICE", 
        #           "priceProtect": False, "origType": "LIMIT", "time": 1631232000000,
        #           "updateTime": 1631232000000}]
        
        liquidations = []
        for order in data:
            # Filter for liquidation-like patterns (you may need to adjust this logic)
            if order.get('reduceOnly') or order.get('closePosition'):
                liquidations.append({
                    'symbol': order.get('symbol'),
                    'side': order.get('side'),
                    'quantity': float(order.get('executedQty', 0)),
                    'price': float(order.get('avgPrice', 0)),
                    'timestamp': order.get('time'),
                    'type': 'liquidation'
                })
        
        return liquidations
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Binance liquidations: {e}")
        return []

def get_binance_liquidation_levels(symbol: str):
    """
    Alternative: Get liquidation levels using open interest and mark price
    This estimates where liquidations might occur based on current OI
    """
    url = "https://fapi.binance.com/fapi/v1/openInterest"
    params = {"symbol": symbol}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        oi_data = response.json()
        
        # Get current mark price
        premium_url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        premium_response = requests.get(premium_url, params={"symbol": symbol})
        premium_data = premium_response.json()
        
        return {
            'symbol': symbol,
            'open_interest': float(oi_data.get('openInterest', 0)),
            'mark_price': float(premium_data.get('markPrice', 0)),
            'timestamp': int(time() * 1000)
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Binance liquidation levels for {symbol}: {e}")
        return {}
    
# if __name__ == "__main__":
# #   data = get_historical_data_limit_binance("AVAXUSDT", "30m", limit=80)
# #   current_price = float(data["close"].iloc[-1])
# #   print(data)
#     # orderbook = get_orderbook("BTCUSDT", limit=5)
#     # print(orderbook)
#     data = get_public_liquidations("BTCUSDT", lookback_hours=24)
#     print(data)
#     data1 = get_binance_liquidation_levels("BTCUSDT")
#     print(data1)
#     funding_history = get_funding_rate_history("BTCUSDT", limit=50)
#     print(funding_history)