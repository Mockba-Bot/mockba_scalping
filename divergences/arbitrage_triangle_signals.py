#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binance triangular arbitrage scanner (spot) with dynamic HUB selection.
- Uses project logger from logs/log_config.py (trader_logger).
- Executable pricing (bid/ask VWAP), per-leg taker fees.
- Enforces PRICE_FILTER, LOT_SIZE, MIN_NOTIONAL/NOTIONAL.
- Pre-filters by spread and top-of-book depth.
- Deduplicated triangles with directionality.
- Threaded depth checks; conservative request pacing.
- Optional Redis cache for metadata and orderbooks.
"""

import os
import sys
import time
import json
import math
import logging
import multiprocessing as mp
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import redis
from dotenv import load_dotenv

# -------------------------
# Resolve project root for imports: app/exchanges/binance -> add app/
# -------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------
# Logger
# -------------------------
from logs.log_config import trader_logger as _trader_logger
logger = _trader_logger

# Honor LOG_LEVEL from .env
load_dotenv()
_level = os.getenv("LOG_LEVEL", "INFO").upper()
if hasattr(logging, _level):
    logger.setLevel(getattr(logging, _level))

# -------------------------
# Configuration
# -------------------------
BINANCE_REST = "https://api.binance.com"
DEPTH_LIMIT = int(os.getenv("DEPTH_LIMIT", "10"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.025"))
TIMEOUT = int(os.getenv("TIMEOUT", "8"))
MAX_THREADS = int(os.getenv("MAX_THREADS", "32"))

# Fees: adjust for your account tier / BNB discount
TAKER_FEE = float(os.getenv("TAKER_FEE", "0.001"))  # 0.1%

# Pre-filters
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "5.0"))
MIN_TOP5_QUOTE_DEPTH = float(os.getenv("MIN_TOP5_QUOTE_DEPTH", "500.0"))
START_NOTIONAL = float(os.getenv("START_NOTIONAL", "200"))
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT", "0.30"))

# Dynamic HUBs
MAX_HUBS = int(os.getenv("MAX_HUBS", "50"))
HUB_REFRESH_SECS = int(os.getenv("HUB_REFRESH_SECS", "300"))
FORCED_HUBS = set(os.getenv("FORCED_HUBS", "USDT,FDUSD,BTC,ETH,BNB").split(","))

# Redis (optional)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Minutes between rounds
# Scan cadence
PERIOD_SECS = int(os.getenv("PERIOD_SECS", "600"))  # 10 minutes between rounds


# -------------------------
# HTTP session with retries
# -------------------------
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=256, pool_maxsize=256)
session.mount("http://", adapter)
session.mount("https://", adapter)

last_request_time = 0.0
request_lock = Lock()

def rate_limited_request():
    global last_request_time
    with request_lock:
        now = time.time()
        elapsed = now - last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        last_request_time = time.time()

# -------------------------
# Redis (optional)
# -------------------------
try:
    rds = redis.Redis(  # modern API; avoids deprecated retry_on_timeout kw
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        socket_connect_timeout=2,
        socket_timeout=2,
        decode_responses=True,
    )
    rds.ping()
    logger.info("Redis connection successful")
except Exception as e:
    logger.warning(f"Redis unavailable: {e}")
    rds = None

def redis_get(key):
    if not rds:
        return None
    try:
        return rds.get(key)
    except Exception:
        return None

def redis_setex(key, ttl, value):
    if not rds:
        return
    try:
        rds.setex(key, ttl, value)
    except Exception:
        pass

# -------------------------
# Exchange metadata
# -------------------------
def load_exchange_info():
    """Fetches and caches Binance exchangeInfo (symbol metadata + filters) for 1h in Redis."""
    cache_key = "binance:exchangeInfo:v1"
    cached = redis_get(cache_key)
    if cached:
        return json.loads(cached)

    rate_limited_request()
    resp = session.get(f"{BINANCE_REST}/api/v3/exchangeInfo", timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    redis_setex(cache_key, 3600, json.dumps(data))
    return data

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def parse_symbol_filters(symbol_obj):
    """Robustly parse PRICE_FILTER, LOT_SIZE, MIN_NOTIONAL/NOTIONAL."""
    fdict = {f.get("filterType"): f for f in symbol_obj.get("filters", [])}

    tick = _safe_float(fdict.get("PRICE_FILTER", {}).get("tickSize"), 1e-8)
    if tick <= 0: tick = 1e-8

    step = _safe_float(fdict.get("LOT_SIZE", {}).get("stepSize"), 1e-8)
    if step <= 0: step = 1e-8

    if "MIN_NOTIONAL" in fdict:
        min_notional = _safe_float(fdict["MIN_NOTIONAL"].get("minNotional"), 0.0)
    elif "NOTIONAL" in fdict:
        min_notional = _safe_float(fdict["NOTIONAL"].get("minNotional"), 0.0)
    else:
        min_notional = 0.0

    return tick, step, min_notional

def build_graph_and_filters():
    """Builds:
    - graph: symbols list, asset set, adjacency map, and base/quote->symbol map
    - filters: tickSize, stepSize, minNotional per symbol
    """
    data = load_exchange_info()
    symbols, symbol_map, assets, adjacency, filters = [], {}, set(), {}, {}

    for s in data.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        base = s.get("baseAsset"); quote = s.get("quoteAsset"); sym = s.get("symbol")
        if not base or not quote or not sym:
            continue

        symbols.append(sym)
        symbol_map[f"{base}/{quote}"] = sym
        assets.update([base, quote])
        adjacency.setdefault(base, set()).add(quote)
        adjacency.setdefault(quote, set()).add(base)

        tick, step, min_notional = parse_symbol_filters(s)
        filters[sym] = {
            "base": base,
            "quote": quote,
            "tickSize": tick,
            "stepSize": step,
            "minNotional": min_notional,
        }

    graph = {
        "symbols": symbols,
        "symbol_map": symbol_map,
        "assets": list(assets),
        "adjacency": {k: list(v) for k, v in adjacency.items()},
    }
    return graph, filters

# -------------------------
# Utilities
# -------------------------
def round_step(x, step):
    return math.floor(x / step) * step

def round_qty(q, step):
    return round_step(q, step)

# -------------------------
# Orderbook + bookTicker
# -------------------------
def get_book_ticker_all():
    """Returns all best bid/ask (bookTicker) with a 1s Redis TTL to limit REST pressure."""
    cache_key = "binance:bookTicker:all"
    cached = redis_get(cache_key)
    if cached:
        return json.loads(cached)

    rate_limited_request()
    resp = session.get(f"{BINANCE_REST}/api/v3/ticker/bookTicker", timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    redis_setex(cache_key, 1, json.dumps(data))
    return data

def get_depth(symbol, ttl_ms=300):
    """Fetches L2 depth up to DEPTH_LIMIT with short TTL caching to reduce duplicate calls."""
    key = f"binance:depth:{symbol}"
    cached = redis_get(key)
    if cached:
        return json.loads(cached)

    rate_limited_request()
    resp = session.get(f"{BINANCE_REST}/api/v3/depth",
                       params={"symbol": symbol, "limit": DEPTH_LIMIT},
                       timeout=TIMEOUT)
    if not resp.ok:
        return {"bids": [], "asks": []}
    data = resp.json()
    if data.get("bids") and data.get("asks"):
        redis_setex(key, max(1, ttl_ms // 1000), json.dumps(data))
    return data

# -------------------------
# Liquidity pre-filters
# -------------------------
def top5_ask_quote_depth(depth):
    total = 0.0
    for px, qty in depth.get("asks", [])[:5]:
        total += float(px) * float(qty)
    return total

def spread_bps_from_bookticker(bt):
    bid = float(bt["bidPrice"]); ask = float(bt["askPrice"])
    if ask <= 0 or bid <= 0 or ask <= bid:
        return float("inf")
    mid = 0.5 * (ask + bid)
    return (ask - bid) / mid * 1e4

def liquidity_ok(symbol, booktickers_map):
    """Fast pre-filter: spread in bps + top-5 ask quote depth threshold + non-empty book."""
    bt = booktickers_map.get(symbol)
    if not bt:
        return False
    if spread_bps_from_bookticker(bt) > MAX_SPREAD_BPS:
        return False
    d = get_depth(symbol, ttl_ms=300)
    if not d.get("bids") or not d.get("asks"):
        return False
    if top5_ask_quote_depth(d) < MIN_TOP5_QUOTE_DEPTH:
        return False
    return True

# -------------------------
# Triangle generation (dedup + directed)
# -------------------------
def find_triangles(graph, hubs):
    """Enumerates directed triangles A→B→C→A anchored on selected hubs; deduplicates by set {A,B,C}."""
    adj = {k: set(v) for k, v in graph["adjacency"].items()}
    asset_set = set(graph["assets"])
    hubs_in = [h for h in hubs if h in asset_set]

    seen = set()
    triangles = []
    for a in hubs_in:
        if a not in adj:
            continue
        nbrs = list(adj[a])
        n = len(nbrs)
        for i in range(n):
            b = nbrs[i]
            for j in range(i + 1, n):
                c = nbrs[j]
                if c in adj.get(b, set()):
                    t1 = (a, b, c, a)
                    t2 = (a, c, b, a)
                    key1 = tuple(sorted([a, b, c])) + ("t1",)
                    key2 = tuple(sorted([a, b, c])) + ("t2",)
                    if key1 not in seen:
                        triangles.append(t1); seen.add(key1)
                    if key2 not in seen:
                        triangles.append(t2); seen.add(key2)
    return triangles

# -------------------------
# Executable legs (VWAP)
# -------------------------
def vwap_fill(depth_side, target_qty_or_quote, by_base, max_levels=DEPTH_LIMIT):
    """Executes a single leg at executable VWAP with taker fee and filter checks.
    Handles both directions depending on (from_asset, to_asset) vs (base, quote).
    Returns dict(amount_to, spent_from, side) or None if not fillable/invalid.
    """
    filled_base = 0.0
    filled_quote = 0.0
    remaining = float(target_qty_or_quote)
    for px_str, qty_str in depth_side[:max_levels]:
        px = float(px_str); lvl_qty = float(qty_str)
        if by_base:
            take = min(lvl_qty, remaining)
            filled_base += take
            filled_quote += take * px
            remaining -= take
        else:
            lvl_quote = lvl_qty * px
            spend = min(lvl_quote, remaining)
            base_buy = spend / px
            filled_base += base_buy
            filled_quote += spend
            remaining -= spend
        if remaining <= 0:
            return filled_base, filled_quote, True
    return filled_base, filled_quote, False


def execute_leg(symbol, from_asset, to_asset, amount_from, filters):
    f = filters.get(symbol)
    if not f:
        return None
    base = f["base"]; quote = f["quote"]
    step = f["stepSize"]; min_notional = f["minNotional"]

    depth = get_depth(symbol, ttl_ms=300)
    if not depth.get("bids") or not depth.get("asks"):
        return None

    if from_asset == base and to_asset == quote:
        qty_base = round_qty(float(amount_from), step)
        if qty_base <= 0:
            return None
        got_base, got_quote, ok = vwap_fill(depth["bids"], qty_base, by_base=True)
        if not ok:
            return None
        gross_quote = float(got_quote)
        fee_quote = gross_quote * TAKER_FEE
        net_quote = gross_quote - fee_quote
        if net_quote < min_notional:
            return None
        return {"amount_to": float(net_quote), "spent_from": float(got_base), "side": "SELL"}

    if from_asset == quote and to_asset == base:
        spend_quote = float(amount_from)
        if spend_quote < min_notional:
            return None
        got_base, used_quote, ok = vwap_fill(depth["asks"], spend_quote, by_base=False)
        if not ok:
            return None
        fee_base = float(got_base) * TAKER_FEE
        net_base = float(got_base) - fee_base
        net_base = round_qty(net_base, step)
        if net_base <= 0:
            return None
        return {"amount_to": float(net_base), "spent_from": float(used_quote), "side": "BUY"}

    return None

def simulate_triangle(triangle, graph, filters, start_amount):
    """Runs A→B, B→C, C→A legs with per-leg sanity:
    - require spent_from ≈ input (±5%) to prevent unit flips/rounding blowups
    - propagate amount_to as next leg input
    Returns profitability metrics or None.
    """
    a, b, c, _ = triangle
    amt = float(start_amount)
    path = [(a, b), (b, c), (c, a)]

    for from_asset, to_asset in path:
        sym = graph["symbol_map"].get(f"{from_asset}/{to_asset}") \
              or graph["symbol_map"].get(f"{to_asset}/{from_asset}")
        if not sym:
            return None

        # NO invertir los argumentos: siempre pasar from_asset -> to_asset
        leg = execute_leg(sym, from_asset, to_asset, amt, filters)
        if not leg:
            return None

        # Sanidad: lo gastado debe ≈ al monto de entrada (tolerancia por redondeo)
        spent = float(leg.get("spent_from", 0.0))
        if spent <= 0 or abs(spent - amt) / max(1e-9, amt) > 0.05:
            return None

        amt = leg["amount_to"]

    profit = amt - start_amount
    return {
        "triangle": triangle,
        "start": start_amount,
        "end": amt,
        "profit_pct": profit / start_amount * 100.0,
        "absolute_profit": profit,
    }

# -------------------------
# Dynamic HUB builder
# -------------------------
def compute_asset_metrics(graph, filters):
    """Scores assets for hub selection by connectivity, quote presence, and pass-rate of liquidity filters."""
    degree = {a: len(neigh) for a, neigh in graph["adjacency"].items()}

    base_count = {a: 0 for a in graph["assets"]}
    quote_count = {a: 0 for a in graph["assets"]}

    bt_all = get_book_ticker_all()
    bt_map = {x["symbol"]: x for x in bt_all if x.get("symbol")}

    def symbol_pass(sym):
        try:
            bt = bt_map.get(sym)
            if not bt:
                return False
            if spread_bps_from_bookticker(bt) > MAX_SPREAD_BPS:
                return False
            d = get_depth(sym, ttl_ms=300)
            if not d.get("bids") or not d.get("asks"):
                return False
            if top5_ask_quote_depth(d) < MIN_TOP5_QUOTE_DEPTH:
                return False
            return True
        except Exception:
            return False

    pass_count = {a: 0 for a in graph["assets"]}
    total_count = {a: 0 for a in graph["assets"]}

    for sym, f in filters.items():
        a = f["base"]; q = f["quote"]
        base_count[a] += 1
        quote_count[q] += 1
        total_count[a] += 1
        total_count[q] += 1
        if symbol_pass(sym):
            pass_count[a] += 1
            pass_count[q] += 1

    metrics = {}
    for a in graph["assets"]:
        tot = max(1, total_count.get(a, 0))
        liq_rate = pass_count.get(a, 0) / tot
        metrics[a] = {
            "degree": degree.get(a, 0),
            "base_count": base_count.get(a, 0),
            "quote_count": quote_count.get(a, 0),
            "liq_rate": liq_rate,
        }
    return metrics

def build_hubs_dynamic(graph, filters, max_hubs=MAX_HUBS):
    """Ranks assets by a weighted score and returns a capped hub set merged with FORCED_HUBS."""
    metrics = compute_asset_metrics(graph, filters)
    if not metrics:
        return set(FORCED_HUBS)

    deg_vals = [v["degree"] for v in metrics.values()]
    qc_vals = [v["quote_count"] for v in metrics.values()]
    deg_lo, deg_hi = min(deg_vals or [0]), max(deg_vals or [1])
    qc_lo, qc_hi = min(qc_vals or [0]), max(qc_vals or [1])

    def ndeg(k):
        return 0.0 if deg_hi == deg_lo else (metrics[k]["degree"] - deg_lo) / (deg_hi - deg_lo)

    def nqc(k):
        return 0.0 if qc_hi == qc_lo else (metrics[k]["quote_count"] - qc_lo) / (qc_hi - qc_lo)

    w1, w2, w3 = 0.5, 0.3, 0.2
    scored = []
    for a in graph["assets"]:
        m = metrics[a]
        score = w1 * ndeg(a) + w2 * nqc(a) + w3 * m["liq_rate"]
        scored.append((score, a))

    scored.sort(reverse=True)
    hubs = set(FORCED_HUBS)
    for _, a in scored:
        if len(hubs) >= max_hubs:
            break
        hubs.add(a)

    return hubs

# -------------------------
# Scanning pipeline
# -------------------------
def build_liquidity_whitelist(symbols):
    """Parallel liquidity gate: keeps only symbols passing spread/depth tests."""
    bt_all = get_book_ticker_all()
    bt_map = {x["symbol"]: x for x in bt_all if x.get("symbol")}

    whitelist = set()

    def worker(sym):
        try:
            return sym if liquidity_ok(sym, bt_map) else None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as ex:
        futures = [ex.submit(worker, s) for s in symbols]
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                whitelist.add(res)

    return whitelist

def scan_with_hubs(graph, filters, hubs, start_notional=START_NOTIONAL, min_profit_pct=MIN_PROFIT_PCT):
    """Full round:
    - enumerate triangles from hubs
    - build whitelist and cull triangles whose legs fail it
    - simulate remaining triangles in parallel
    - return results sorted by profit_pct
    """
    triangles = find_triangles(graph, hubs=hubs)
    if not triangles:
        return []

    tri_symbols = set()
    for a, b, c, _ in triangles:
        for x, y in ((a, b), (b, c), (c, a)):
            s = graph["symbol_map"].get(f"{x}/{y}") or graph["symbol_map"].get(f"{y}/{x}")
            if s:
                tri_symbols.add(s)

    whitelist = build_liquidity_whitelist(list(tri_symbols))
    if not whitelist:
        return []

    filtered = []
    for t in triangles:
        ok = True
        for x, y in ((t[0], t[1]), (t[1], t[2]), (t[2], t[3])):
            s = graph["symbol_map"].get(f"{x}/{y}") or graph["symbol_map"].get(f"{y}/{x}")
            if not s or s not in whitelist:
                ok = False
                break
        if ok:
            filtered.append(t)

    if not filtered:
        return []

    results = []

    def sim_worker(tri):
        try:
            res = simulate_triangle(tri, graph, filters, start_notional)
            if res and res["profit_pct"] >= min_profit_pct:
                return res
        except Exception:
            return None
        return None

    with ThreadPoolExecutor(max_workers=min(16, MAX_THREADS)) as ex:
        futures = [ex.submit(sim_worker, t) for t in filtered]
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: x["profit_pct"], reverse=True)
    return results

# -------------------------
# Main
# -------------------------
def main():
    """
    Orchestrates the scanning lifecycle:
    - Builds static metadata (graph, filters).
    - Initializes/rebuilds dynamic hubs on schedule.
    - Runs a scan round (build whitelist -> filter triangles -> simulate).
    - Logs a clear end-of-round marker and sleeps to keep a fixed cadence.
    """
    logger.info("Starting Binance triangular arbitrage scanner")
    graph, filters = build_graph_and_filters()
    logger.info(f"Assets: {len(graph['assets'])}  Symbols: {len(graph['symbols'])}")

    hubs = build_hubs_dynamic(graph, filters, max_hubs=MAX_HUBS)
    logger.info(f"Dynamic hubs initialized ({len(hubs)}): sample={sorted(list(hubs))[:20]}")
    last_hub_refresh = time.time()

    consecutive_errors = 0
    max_errors = 5
    round_idx = 0

    while True:
        t0 = time.time()
        round_idx += 1
        try:
            # Refresh hubs on schedule
            if time.time() - last_hub_refresh >= HUB_REFRESH_SECS:
                hubs = build_hubs_dynamic(graph, filters, max_hubs=MAX_HUBS)
                last_hub_refresh = time.time()
                logger.info(f"Hubs refreshed ({len(hubs)})")

            # One scan round
            opps = scan_with_hubs(graph, filters, hubs, START_NOTIONAL, MIN_PROFIT_PCT)

            # Round results
            if opps:
                logger.info(f"Found {len(opps)} opportunities")
                for o in opps[:5]:
                    tri = " → ".join(o["triangle"])
                    logger.info(f"{tri} | {o['profit_pct']:.3f}% | +{o['absolute_profit']:.6f}")
            else:
                logger.info("No profitable opportunities")

            consecutive_errors = 0

        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Scan error {consecutive_errors}/{max_errors}: {e}")
            if consecutive_errors >= max_errors:
                logger.error("Too many consecutive errors. Exiting.")
                break

        # End-of-round marker + deterministic sleep to next round
        elapsed = time.time() - t0
        sleep_s = max(0.0, PERIOD_SECS - elapsed)
        logger.info(
            f"--- End of round #{round_idx} | duration={elapsed:.2f}s | next round in {sleep_s:.2f}s ---"
        )
        time.sleep(sleep_s)

# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    mp.freeze_support()
    main()
