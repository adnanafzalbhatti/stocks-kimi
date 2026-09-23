"""
STOCK SIGNAL AGENT v3.4 — LONG-ONLY + full universe (default) + Telegram + blacklist + ATR stops

Cash-account friendly: BUY only (no shorting). Clear enter / hold / exit plan.

USAGE:
    python stock_agent_v3.py backtest
    python stock_agent_v3.py scan
    python stock_agent_v3.py live          # Telegram auto-scan (market hours)
    python stock_agent_v3.py signal NVDA
    python stock_agent_v3.py --list

TELEGRAM:
    1) Put TELEGRAM_BOT_TOKEN and ADMIN_IDS in .env
    2) Message your bot once (/start)
    3) python stock_agent_v3.py live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time as time_mod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from scipy.signal import argrelextrema

# Optional third‑party sources for enriched signals
# Reddit (praw) – requires CLIENT_ID, CLIENT_SECRET, USER_AGENT in env vars
# Twitter (tweepy) – requires BEARER_TOKEN in env vars
try:
    import praw
except ImportError:
    praw = None  # Will be installed if needed
try:
    import tweepy
except ImportError:
    tweepy = None  # Will be installed if needed

# -------------------------------------------------------------------
# Configuration – external data sources
# -------------------------------------------------------------------
# Reddit subreddits to monitor for trade ideas / signals
REDDIT_SUBREDDITS = ["tradewithcongress", "wallstreetbets", "theraceto1million"]

# News source placeholders – yfinance provides ticker news; additional APIs could be added
NEWS_SOURCES = ["yfinance"]

# Twitter (X) handles to monitor – top investors, whales, analysts
TWITTER_HANDLES = ["elonmusk", "CathieDWood", "RayDalio"]

load_dotenv()

# Credentials – read from environment (or .env)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "stock_agent_v3_bot")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Ensure yfinance (curl) can locate trusted CA bundle on Windows
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes
    from telegram.request import HTTPXRequest
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    HTTPXRequest = object  # type: ignore
    Update = object  # type: ignore
    ContextTypes = object  # type: ignore

try:
    import factor_engine
    HAS_FACTORS = True
except ImportError:
    factor_engine = None  # type: ignore
    HAS_FACTORS = False

try:
    import genai_agents
    HAS_GENAI = True
except ImportError:
    genai_agents = None  # type: ignore
    HAS_GENAI = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# LONG ONLY — cash / non-margin accounts cannot short until shares are owned.
LONG_ONLY = True

# SPY is regime reference only (not traded by default).
# Expanded liquid universe: mega/large/mid-caps + sector ETFs (~90 names).
STOCKS: Dict[str, Dict[str, str]] = {
    # --- Mega-cap tech / growth (Mag7 + AI) ---
    "AAPL": {"yf": "AAPL", "sector": "Tech"},
    "MSFT": {"yf": "MSFT", "sector": "Tech"},
    "NVDA": {"yf": "NVDA", "sector": "Tech"},
    "GOOGL": {"yf": "GOOGL", "sector": "Tech"},
    "AMZN": {"yf": "AMZN", "sector": "Consumer"},
    "META": {"yf": "META", "sector": "Tech"},
    "TSLA": {"yf": "TSLA", "sector": "Auto"},
    "AVGO": {"yf": "AVGO", "sector": "Tech"},
    "AMD": {"yf": "AMD", "sector": "Tech"},
    "ORCL": {"yf": "ORCL", "sector": "Tech"},
    # --- Software / semis / hardware ---
    "CRM": {"yf": "CRM", "sector": "Tech"},
    "ADBE": {"yf": "ADBE", "sector": "Tech"},
    "CSCO": {"yf": "CSCO", "sector": "Tech"},
    "IBM": {"yf": "IBM", "sector": "Tech"},
    "INTC": {"yf": "INTC", "sector": "Tech"},
    "QCOM": {"yf": "QCOM", "sector": "Tech"},
    "TXN": {"yf": "TXN", "sector": "Tech"},
    "AMAT": {"yf": "AMAT", "sector": "Tech"},
    "MU": {"yf": "MU", "sector": "Tech"},
    "SNDK": {"yf": "SNDK", "sector": "Tech"},
    "NOW": {"yf": "NOW", "sector": "Tech"},
    "INTU": {"yf": "INTU", "sector": "Tech"},
    "PANW": {"yf": "PANW", "sector": "Tech"},
    "PLTR": {"yf": "PLTR", "sector": "Tech"},
    "SNPS": {"yf": "SNPS", "sector": "Tech"},
    "CDNS": {"yf": "CDNS", "sector": "Tech"},
    "KLAC": {"yf": "KLAC", "sector": "Tech"},
    "LRCX": {"yf": "LRCX", "sector": "Tech"},
    "ADI": {"yf": "ADI", "sector": "Tech"},
    "NXPI": {"yf": "NXPI", "sector": "Tech"},
    "MRVL": {"yf": "MRVL", "sector": "Tech"},
    "CRWD": {"yf": "CRWD", "sector": "Tech"},
    "SNOW": {"yf": "SNOW", "sector": "Tech"},
    "DDOG": {"yf": "DDOG", "sector": "Tech"},
    "APP": {"yf": "APP", "sector": "Tech"},
    "SHOP": {"yf": "SHOP", "sector": "Tech"},
    # --- Consumer / retail / media ---
    "NFLX": {"yf": "NFLX", "sector": "Media"},
    "COST": {"yf": "COST", "sector": "Consumer"},
    "WMT": {"yf": "WMT", "sector": "Consumer"},
    "HD": {"yf": "HD", "sector": "Consumer"},
    "LOW": {"yf": "LOW", "sector": "Consumer"},
    "MCD": {"yf": "MCD", "sector": "Consumer"},
    "NKE": {"yf": "NKE", "sector": "Consumer"},
    "SBUX": {"yf": "SBUX", "sector": "Consumer"},
    "DIS": {"yf": "DIS", "sector": "Media"},
    "BKNG": {"yf": "BKNG", "sector": "Consumer"},
    "UBER": {"yf": "UBER", "sector": "Transport"},
    "ABNB": {"yf": "ABNB", "sector": "Consumer"},
    "CMG": {"yf": "CMG", "sector": "Consumer"},
    "TGT": {"yf": "TGT", "sector": "Consumer"},
    "TJX": {"yf": "TJX", "sector": "Consumer"},
    "MAR": {"yf": "MAR", "sector": "Consumer"},
    # --- Financials ---
    "JPM": {"yf": "JPM", "sector": "Finance"},
    "BAC": {"yf": "BAC", "sector": "Finance"},
    "WFC": {"yf": "WFC", "sector": "Finance"},
    "C": {"yf": "C", "sector": "Finance"},
    "GS": {"yf": "GS", "sector": "Finance"},
    "MS": {"yf": "MS", "sector": "Finance"},
    "V": {"yf": "V", "sector": "Finance"},
    "MA": {"yf": "MA", "sector": "Finance"},
    "BRK-B": {"yf": "BRK-B", "sector": "Finance"},
    "AXP": {"yf": "AXP", "sector": "Finance"},
    "BLK": {"yf": "BLK", "sector": "Finance"},
    "SCHW": {"yf": "SCHW", "sector": "Finance"},
    "SPGI": {"yf": "SPGI", "sector": "Finance"},
    "CB": {"yf": "CB", "sector": "Finance"},
    "PGR": {"yf": "PGR", "sector": "Finance"},
    # --- Healthcare ---
    "UNH": {"yf": "UNH", "sector": "Health"},
    "JNJ": {"yf": "JNJ", "sector": "Health"},
    "LLY": {"yf": "LLY", "sector": "Health"},
    "ABBV": {"yf": "ABBV", "sector": "Health"},
    "MRK": {"yf": "MRK", "sector": "Health"},
    "PFE": {"yf": "PFE", "sector": "Health"},
    "TMO": {"yf": "TMO", "sector": "Health"},
    "ISRG": {"yf": "ISRG", "sector": "Health"},
    "ABT": {"yf": "ABT", "sector": "Health"},
    "DHR": {"yf": "DHR", "sector": "Health"},
    "SYK": {"yf": "SYK", "sector": "Health"},
    "BSX": {"yf": "BSX", "sector": "Health"},
    "AMGN": {"yf": "AMGN", "sector": "Health"},
    "GILD": {"yf": "GILD", "sector": "Health"},
    "VRTX": {"yf": "VRTX", "sector": "Health"},
    "REGN": {"yf": "REGN", "sector": "Health"},
    # --- Industrials / energy / materials ---
    "CAT": {"yf": "CAT", "sector": "Industrial"},
    "GE": {"yf": "GE", "sector": "Industrial"},
    "HON": {"yf": "HON", "sector": "Industrial"},
    "RTX": {"yf": "RTX", "sector": "Industrial"},
    "BA": {"yf": "BA", "sector": "Industrial"},
    "DE": {"yf": "DE", "sector": "Industrial"},
    "UNP": {"yf": "UNP", "sector": "Industrial"},
    "UPS": {"yf": "UPS", "sector": "Industrial"},
    "LMT": {"yf": "LMT", "sector": "Industrial"},
    "ETN": {"yf": "ETN", "sector": "Industrial"},
    "PH": {"yf": "PH", "sector": "Industrial"},
    "XOM": {"yf": "XOM", "sector": "Energy"},
    "CVX": {"yf": "CVX", "sector": "Energy"},
    "COP": {"yf": "COP", "sector": "Energy"},
    "SLB": {"yf": "SLB", "sector": "Energy"},
    "LIN": {"yf": "LIN", "sector": "Materials"},
    "NEE": {"yf": "NEE", "sector": "Utilities"},
    "SO": {"yf": "SO", "sector": "Utilities"},
    "DUK": {"yf": "DUK", "sector": "Utilities"},
    # --- Major ETFs (liquid proxies) ---
    "QQQ": {"yf": "QQQ", "sector": "ETF"},
    "IWM": {"yf": "IWM", "sector": "ETF"},
    "DIA": {"yf": "DIA", "sector": "ETF"},
    "XLK": {"yf": "XLK", "sector": "ETF"},
    "XLF": {"yf": "XLF", "sector": "ETF"},
    "XLV": {"yf": "XLV", "sector": "ETF"},
    "XLE": {"yf": "XLE", "sector": "ETF"},
    "XLI": {"yf": "XLI", "sector": "ETF"},
    "XLY": {"yf": "XLY", "sector": "ETF"},
    "SMH": {"yf": "SMH", "sector": "ETF"},
    "SPY": {"yf": "SPY", "sector": "ETF"},
    "PG": {"yf": "PG", "sector": "Consumer"},
    "KO": {"yf": "KO", "sector": "Consumer"},
    "PEP": {"yf": "PEP", "sector": "Consumer"},
    "AMT": {"yf": "AMT", "sector": "RealEstate"},
    "PLD": {"yf": "PLD", "sector": "RealEstate"},
    "GLD": {"yf": "GLD", "sector": "Commodity"},
}

# Regime reference (fetched separately; not required in trade list)
REGIME_SYMBOL = "SPY"

# Tickers with consistently negative backtest P&L (v3.2 audit, Aug 2026).
# Kept in STOCKS for reference / re-testing but excluded from live signals.
BLACKLIST: set = {
    "SHOP", "GS", "REGN", "TMO", "HD", "TXN", "PANW", "META",
    "ADBE", "UPS", "SNPS", "PFE", "INTU", "SBUX", "SLB", "MU",
    "CRM", "IBM", "INTC", "QCOM", "AVGO", "AMD", "ORCL", "NXPI",
    "LRCX", "ADI", "KLAC", "MRVL", "CRWD", "DDOG", "APP", "PLTR",
    "ATVI", "ZBRA", "WBA", "BKR", "CCL", "DAL", "FITB", "GPC",
    "HOG", "JBHT", "K", "KHC", "KMX", "MCK", "NEM",
    "NWL", "O", "POOL", "RCL", "ROKU", "SLM", "SJM", "TTWO",
    "VZ", "WDC", "WYNN", "ZTS"
}

# High-conviction core list rebuilt from top-15 by total backtest P&L (≥55% WR).
APPROVED_TICKERS: List[str] = [
    "AVGO", "AMD", "LRCX", "BKNG", "BA", "CAT", "NXPI", "LIN",
    "SCHW", "CDNS", "ABNB", "PLTR", "HON", "GE", "TGT",
]
# Default LIVE/scan = FULL universe. Quality comes from high conf + tight gates (not fewer names).
# Set USE_APPROVED_ONLY=1 only if you want the old 20-name core list.
USE_APPROVED_ONLY = os.getenv("USE_APPROVED_ONLY", "0").strip().lower() in ("1", "true", "yes")

# Risk / quality gates — Swing Investor (~20% target) & Core ETF Accumulation
MAX_RISK_PCT = 1.0          # risk per trade as % of account
POSITION_SIZE_PCT = 15.0    # hard cap on notional per swing position
# High confidence only. Override with MIN_CONFIDENCE= in .env if needed.
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "0"))
MIN_RR = float(os.getenv("MIN_RR", "2.0"))
MAX_STOP_PCT = float(os.getenv("MAX_STOP_PCT", "0.075"))  # max 7.5% structural risk
MIN_STOP_PCT = float(os.getenv("MIN_STOP_PCT", "0.035"))  # min 3.5% buffer to avoid noise shakeout
ATR_STOP_MULT = float(os.getenv("ATR_STOP_MULT", "2.2"))
# Allow both full stacked uptrends and value dip reversals on solid blue chips
REQUIRE_STACKED_UP = os.getenv("REQUIRE_STACKED_UP", "0").strip().lower() in ("1", "true", "yes")
# Allow neutral regime dip buying on quality names (bear regime still blocked)
BULL_REGIME_ONLY = os.getenv("BULL_REGIME_ONLY", "0").strip().lower() in ("1", "true", "yes")
# Pullback bands
MIN_PULLBACK = float(os.getenv("MIN_PULLBACK", "0.015"))
MAX_PULLBACK = float(os.getenv("MAX_PULLBACK", "0.065"))
# Deep value dip band (for AMZN / NVDA style reversals)
MIN_VALUE_DIP = float(os.getenv("MIN_VALUE_DIP", "0.05"))
MAX_VALUE_DIP = float(os.getenv("MAX_VALUE_DIP", "0.22"))
# RSI buy-the-dip band
RSI_MIN = float(os.getenv("RSI_MIN", "38"))
RSI_MAX = float(os.getenv("RSI_MAX", "58"))
# Minimum ADX
MIN_ADX = float(os.getenv("MIN_ADX", "18"))
# Reject climax / panic volume on entry bar
MAX_VOL_MULT = float(os.getenv("MAX_VOL_MULT", "2.2"))
MAX_ATR_EXPANSION = float(os.getenv("MAX_ATR_EXPANSION", "1.6"))

# --- Fundamental / analyst / ownership / social layer (v3.4) -----------------
USE_FACTORS = os.getenv("USE_FACTORS", "1").strip().lower() in ("1", "true", "yes")
MIN_FACTOR_SCORE = float(os.getenv("MIN_FACTOR_SCORE", "55"))
FACTOR_MAX_TILT = float(os.getenv("FACTOR_MAX_TILT", "10"))
FACTOR_PIVOT = float(os.getenv("FACTOR_PIVOT", "55"))
MAX_SIGNALS_PER_SCAN = int(os.getenv("MAX_SIGNALS_PER_SCAN", "3"))
ENABLE_EXTERNAL_DATA = os.getenv("ENABLE_EXTERNAL_DATA", "0").strip().lower() in ("1", "true", "yes")

# Medium-term holding horizon (~2.5 months of 1h bars)
HOLD_BARS = int(os.getenv("HOLD_BARS", "350"))
SCAN_STEP = 6               # scan every N bars (~1 trading day on 1h)
COOLDOWN_BARS = 36          # fewer re-entries on same ticker in backtest
SLIPPAGE = 0.0005

# Swing profit targets (~20% target on single stocks)
SWING_TP_PCT = float(os.getenv("SWING_TP_PCT", "0.20"))             # +20% take profit target
SWING_TRAIL_START_PCT = float(os.getenv("SWING_TRAIL_START_PCT", "0.10"))  # +10% activates trailing stop
SWING_TRAIL_LOCK_PCT = float(os.getenv("SWING_TRAIL_LOCK_PCT", "0.05"))   # +5% locked in
BE_TRIGGER_R = 1.0          # trailing or breakeven trigger
TP1_R = 1.8                 # scale-out or trailing trigger in R terms
TP2_R = 3.5                 # runner target in R terms
TP1_FRACTION = 0.50         # take 50% off at TP1

# Live / Telegram schedule (US/Eastern)
ET = ZoneInfo("America/New_York")
MARKET_OPEN = dt_time(10, 0)    # skip opening noise
MARKET_CLOSE = dt_time(15, 30)  # avoid last 30m
SCAN_TIMES_ET = [
    dt_time(10, 15),
    dt_time(12, 15),
    dt_time(14, 15),
]  # 3 quality scans / session (tight filters = less need for 4th)
EOD_RECAP_TIME = dt_time(16, 5)  # after cash close — daily Telegram scorecard
SIGNAL_COOLDOWN_HOURS = int(os.getenv("SIGNAL_COOLDOWN_HOURS", "36"))  # re-alert window
STATE_FILE = Path(__file__).with_name("live_signal_state.json")
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "25000"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit()]

# Quiet noisy HTTP libs so bot tokens never appear in live_stderr.log
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)
logging.getLogger("telegram.ext").setLevel(logging.INFO)

HOLIDAYS = {
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
}


def tradeable_tickers(use_approved: Optional[bool] = None) -> List[str]:
    """Default: full STOCKS universe minus BLACKLIST. Pass use_approved=True for core list only."""
    flag = USE_APPROVED_ONLY if use_approved is None else use_approved
    if flag:
        return [t for t in APPROVED_TICKERS if t in STOCKS and t not in BLACKLIST]
    return [t for t in STOCKS if t not in BLACKLIST]


# =============================================================================
# HELPERS
# =============================================================================

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or np.isnan(b) or np.isinf(b) or np.isnan(a):
        return default
    return float(a / b)


def is_market_open(dt: datetime) -> bool:
    if dt.weekday() >= 5:
        return False
    if dt.strftime("%Y-%m-%d") in HOLIDAYS:
        return False
    t = dt.time() if hasattr(dt, "time") else dt
    # yfinance hourly bars are often tz-aware; compare clock time only
    if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        t = dt.timetz().replace(tzinfo=None) if hasattr(dt, "timetz") else dt.time()
        # Prefer naive local-like hour from timestamp
        t = datetime(2000, 1, 1, dt.hour, dt.minute).time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


# =============================================================================
# DATA
# =============================================================================

def fetch_data(symbol: str, period: str = "730d", interval: str = "1h") -> pd.DataFrame:
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        rename_map = {
            "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Adj Close": "close", "Volume": "volume",
            "Datetime": "timestamp", "Date": "timestamp",
        }
        data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        for col in ["open", "high", "low", "close", "volume"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce").astype(float)
        data = data.dropna(subset=["open", "high", "low", "close"])
        data = data[(data["close"] > 0) & (data["high"] > 0) & (data["low"] > 0)].reset_index(drop=True)
        return data
    except Exception as e:
        logger.error("Error fetching %s: %s", symbol, e)
        return pd.DataFrame()


# =============================================================================
# EXTERNAL DATA HELPERS (Reddit / Twitter / News)
# =============================================================================

def fetch_reddit_posts(limit: int = 10) -> Dict[str, List[str]]:
    """Fetch top post titles from configured Reddit subreddits.
    Uses Reddit's public RSS/Atom feeds — no API credentials needed.
    Returns a dict {subreddit: [titles]}."""
    import requests as _req
    import re
    import time as _time

    # Fast 10-minute in-memory cache
    if not hasattr(fetch_reddit_posts, "_cache"):
        fetch_reddit_posts._cache = {}
        fetch_reddit_posts._last_fetch = 0

    now_ts = _time.time()
    if now_ts - fetch_reddit_posts._last_fetch < 600 and fetch_reddit_posts._cache:
        return fetch_reddit_posts._cache

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockSignalAgent/3.5 (Contact: trading_committee@quant.io)"
    }
    out: Dict[str, List[str]] = {}
    for sub in REDDIT_SUBREDDITS:
        try:
            url = f"https://www.reddit.com/r/{sub}/.rss?limit={limit}"
            resp = _req.get(url, headers=headers, timeout=5)
            if resp.status_code != 200:
                logger.debug(f"Reddit r/{sub} RSS returned HTTP {resp.status_code}")
                continue
            titles = re.findall(r"<title[^>]*>(.+?)</title>", resp.text)
            posts = [t for t in titles[1:] if t and not t.startswith("r/")]
            out[sub] = posts
        except Exception as e:
            logger.debug(f"Reddit fetch note for r/{sub}: {e}")

    if out:
        fetch_reddit_posts._cache = out
        fetch_reddit_posts._last_fetch = now_ts
    return out or fetch_reddit_posts._cache


def fetch_twitter_recent(handle: str, count: int = 10) -> List[str]:
    """Fetch recent tweets (text only) for a given X handle."""
    if not tweepy:
        logger.warning("tweepy not installed – Twitter data unavailable.")
        return []
    if not TWITTER_BEARER_TOKEN:
        logger.warning("Twitter bearer token missing – skipping Twitter fetch.")
        return []
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    try:
        user = client.get_user(username=handle)
        if not user.data:
            return []
        uid = user.data.id
        tweets = client.get_users_tweets(id=uid, max_results=count)
        return [t.text for t in tweets.data] if tweets.data else []
    except Exception as e:
        logger.error(f"Twitter fetch error for @{handle}: {e}")
        return []


def fetch_all_twitter() -> Dict[str, List[str]]:
    """Aggregate recent tweets from all configured handles."""
    return {h: fetch_twitter_recent(h) for h in TWITTER_HANDLES}


def fetch_company_news(ticker: str) -> List[Dict]:
    """Retrieve recent news items for a ticker via yfinance (fallback to empty list)."""
    try:
        tk = yf.Ticker(ticker)
        return tk.news if tk.news else []
    except Exception as e:
        logger.error(f"News fetch error for {ticker}: {e}")
        return []


def enrich_signal_with_external_data(ticker: str, factors: List[str]) -> List[str]:
    """Append external source info (Reddit / Twitter / News) to the rationale list.
    Only called when ENABLE_EXTERNAL_DATA is True."""
    # 1) Reddit – first relevant headline per subreddit
    try:
        reddit_data = fetch_reddit_posts(limit=5)
        for sub, titles in reddit_data.items():
            # Look for posts that mention this ticker
            relevant = [t for t in titles if ticker.upper() in t.upper()]
            if relevant:
                factors.append(f"Reddit r/{sub}: {relevant[0][:80]}")
            elif titles:
                factors.append(f"Reddit r/{sub} (top): {titles[0][:80]}")
    except Exception as e:
        logger.error(f"Reddit enrichment error: {e}")

    # 2) Twitter – first tweet per handle
    try:
        twitter_data = fetch_all_twitter()
        for handle, tweets in twitter_data.items():
            if tweets:
                factors.append(f"X @{handle}: {tweets[0][:80]}")
    except Exception as e:
        logger.error(f"Twitter enrichment error: {e}")

    # 3) News – top 3 headlines from yfinance
    try:
        news_items = fetch_company_news(ticker)
        for item in news_items[:3]:
            title = ""
            if isinstance(item, dict):
                title = item.get("title") or ""
                if not title and "content" in item and isinstance(item["content"], dict):
                    title = item["content"].get("title", "")
            else:
                title = str(item)
            if title:
                factors.append(f"News: {title[:80]}")
    except Exception as e:
        logger.error(f"News enrichment error: {e}")

    return factors


# =============================================================================
# INDICATORS
# =============================================================================

def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period).mean() / tr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, min_periods=period).mean()


def macd_hist(closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = ema(closes, fast) - ema(closes, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line - sig


def bollinger(closes: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    return mid + n_std * std, mid, mid - n_std * std


def swing_points(series: pd.Series, order: int = 5) -> Tuple[pd.Series, pd.Series]:
    highs_idx = argrelextrema(series.values, np.greater, order=order)[0]
    lows_idx = argrelextrema(series.values, np.less, order=order)[0]
    highs = pd.Series(False, index=series.index)
    lows = pd.Series(False, index=series.index)
    if len(highs_idx):
        highs.iloc[highs_idx] = True
    if len(lows_idx):
        lows.iloc[lows_idx] = True
    return highs, lows


def detect_divergence(price: pd.Series, indicator: pd.Series, kind: str = "bullish") -> bool:
    p = price.tail(60)
    ind = indicator.tail(60)
    if kind == "bullish":
        p_idx = argrelextrema(p.values, np.less, order=5)[0]
        i_idx = argrelextrema(ind.values, np.less, order=5)[0]
        if len(p_idx) < 2 or len(i_idx) < 2:
            return False
        return (
            p.iloc[p_idx[-1]] < p.iloc[p_idx[-2]]
            and ind.iloc[i_idx[-1]] > ind.iloc[i_idx[-2]]
            and ind.iloc[i_idx[-1]] < 40
        )
    p_idx = argrelextrema(p.values, np.greater, order=5)[0]
    i_idx = argrelextrema(ind.values, np.greater, order=5)[0]
    if len(p_idx) < 2 or len(i_idx) < 2:
        return False
    return (
        p.iloc[p_idx[-1]] > p.iloc[p_idx[-2]]
        and ind.iloc[i_idx[-1]] < ind.iloc[i_idx[-2]]
        and ind.iloc[i_idx[-1]] > 60
    )


# =============================================================================
# SIGNAL
# =============================================================================

@dataclass
class StockSignal:
    ticker: str
    direction: str
    entry: float
    stop_loss: float
    target_1: float
    target_2: float
    confidence: int
    rationale: List[str]
    r_r: float
    risk_pct: float
    timestamp: datetime
    action_plan: str
    regime: str
    tech_confidence: int = 0
    factor_score: Optional[float] = None
    factor_grade: Optional[str] = None
    factor_summary: Optional[str] = None
    days_to_earnings: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["rationale"] = list(self.rationale)
        return d


def _regime_from_spy(spy_df: Optional[pd.DataFrame], ts: datetime) -> str:
    """Bull / bear / neutral from SPY trend (long-only: avoid new buys in bear)."""
    if spy_df is None or spy_df.empty:
        return "unknown"
    hist = spy_df[spy_df["timestamp"] <= ts]
    if len(hist) < 220:
        return "unknown"
    close = hist["close"]
    e50 = ema(close, 50).iloc[-1]
    e200 = ema(close, 200).iloc[-1]
    px = float(close.iloc[-1])
    if np.isnan(e50) or np.isnan(e200):
        return "unknown"
    slope = safe_div(e50 - ema(close, 50).iloc[-6], abs(e50), 0.0)
    if px > e200 and e50 > e200 and slope >= 0:
        return "bull"
    if px < e200 and e50 < e200 and slope <= 0:
        return "bear"
    return "neutral"


def generate_signal(
    ticker: str,
    df: pd.DataFrame,
    current_price: float,
    timestamp: datetime,
    regime: str = "unknown",
    factor: Optional[Any] = None,
) -> Optional[StockSignal]:
    """
    LONG-ONLY quality signal:
      1) Uptrend alignment (EMA stack)
      2) Controlled pullback into value (not FOMO chase)
      3) RSI in healthy buy zone + MACD not accelerating down
      4) Trend strength (ADX)
      5) Location (BB lower half preferred)
      6) Optional bullish divergence bonus
      7) SPY regime: no new longs in bear; A+ only in neutral
      8) Business quality gate: analysts / growth / ownership / earnings timing
    """
    if df is None or df.empty or current_price <= 0 or len(df) < 220:
        return None
    if np.isnan(current_price):
        return None

    # Hard long-only gate
    if LONG_ONLY is False:
        pass  # reserved; this build is long-only

    close = df["close"]
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e200 = ema(close, 200)
    if any(np.isnan(x.iloc[-1]) for x in (e20, e50, e200)):
        return None

    e20v, e50v, e200v = float(e20.iloc[-1]), float(e50.iloc[-1]), float(e200.iloc[-1])
    stacked_up = current_price > e20v > e50v > e200v
    soft_up = current_price > e50v > e200v and e20v >= e50v * 0.99

    is_etf = STOCKS.get(ticker, {}).get("sector") == "ETF" or ticker in (
        "SPY", "VOO", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "SMH"
    )

    # --- Market regime: cash is a position in severe bear market ---
    if regime == "bear":
        return None

    # Structure & Pullback metrics
    sh, sl_pts = swing_points(close, order=5)
    recent_highs = close[sh].tail(4)
    recent_lows = close[sl_pts].tail(4)

    atr_s = atr(df)
    atr_v = float(atr_s.iloc[-1])
    if np.isnan(atr_v) or atr_v <= 0:
        return None

    roll_high_25 = float(df["high"].tail(25).max()) if len(df) >= 25 else float(close.max())
    pullback = safe_div(roll_high_25 - current_price, roll_high_25, 0.0)

    rsi_s = rsi(close)
    rsi_v = float(rsi_s.iloc[-1])
    if np.isnan(rsi_v):
        return None
    min_rsi_recent = float(rsi_s.tail(18).min())

    hist = macd_hist(close)
    h0 = float(hist.iloc[-1]) if len(hist) > 0 else 0.0
    h1 = float(hist.iloc[-2]) if len(hist) > 1 else h0

    score = 0
    factors: List[str] = []
    setup_name = ""

    # Evaluate Setups: Core ETF vs Single Stock (Momentum Pullback or Value Dip Reversal)
    if is_etf:
        if pullback >= 0.018 or rsi_v <= 50 or current_price <= e50v * 1.01:
            setup_name = "ETF_DIP_ACCUMULATION"
            score += 35
            factors.append("Core ETF accumulation zone (Long-term hold)")
        else:
            return None
    else:
        # Setup A: Momentum Pullback in Uptrend
        momentum_eligible = (
            (stacked_up or (soft_up and not REQUIRE_STACKED_UP))
            and (MIN_PULLBACK <= pullback <= MAX_PULLBACK)
            and (RSI_MIN <= rsi_v <= RSI_MAX)
            and (current_price > e20v * 0.99)
        )

        # Setup B: Value Dip Reversal (Captures AMZN/NVDA deep pullbacks)
        macro_ok = current_price >= e200v * 0.92
        deep_dip = (MIN_VALUE_DIP <= pullback <= MAX_VALUE_DIP)
        oversold = min_rsi_recent <= 38
        reversal_candle = (
            (current_price > float(df["open"].iloc[-1]))
            and (current_price > e20v)
            and (float(close.iloc[-2]) <= float(e20.iloc[-2]))
        )
        value_dip_eligible = macro_ok and deep_dip and oversold and reversal_candle

        if value_dip_eligible:
            setup_name = "VALUE_DIP_REVERSAL"
            score += 38
            factors.append(f"Value dip reversal ({pullback*100:.1f}% dip reclaimed 20 EMA)")
            factors.append(f"Oversold washed (RSI hit {min_rsi_recent:.0f})")
        elif momentum_eligible:
            setup_name = "MOMENTUM_PULLBACK"
            score += 32
            factors.append(f"Momentum pullback ({pullback*100:.1f}% into EMA support)")
            factors.append(f"RSI healthy buy zone ({rsi_v:.0f})")
        else:
            return None

    direction = "LONG"

    # Regime scoring
    if regime == "bull":
        score += 12
        factors.append("SPY bull regime")
    elif regime == "neutral":
        if BULL_REGIME_ONLY:
            return None
        score += 4
        factors.append("Neutral regime — selective entry")

    # ADX trend strength
    adx_s = adx(df)
    adx_v = float(adx_s.iloc[-1]) if len(adx_s) else np.nan
    if not np.isnan(adx_v) and adx_v >= MIN_ADX:
        score += 8
        factors.append(f"Trend strength confirmed (ADX {adx_v:.0f})")

    # MACD momentum turning up
    if h0 > h1:
        score += 8
        factors.append("MACD momentum turning up")

    # Divergence
    if detect_divergence(close, rsi_s, "bullish"):
        score += 8
        factors.append("Bullish RSI divergence")

    tech_score = int(max(0, min(score, 100)))

    # Factors / Fundamentals (Live only)
    f_score = f_grade = f_summary = None
    days_to_earnings = None
    if factor is not None:
        days_to_earnings = getattr(factor, "days_to_earnings", None)
        if getattr(factor, "earnings_blackout", False):
            return None
        f_score = float(getattr(factor, "score", 50.0))
        if f_score < MIN_FACTOR_SCORE:
            return None
        f_grade = getattr(factor, "grade", None)
        f_summary = factor.summary() if hasattr(factor, "summary") else None
        tilt = (f_score - FACTOR_PIVOT) * 0.35
        tilt = max(-FACTOR_MAX_TILT, min(FACTOR_MAX_TILT, tilt))
        score = tech_score + tilt
        factors.append(f"Fundamentals {f_score:.0f}/100 ({f_grade}) — {f_summary}")
    else:
        score = tech_score

    score = int(round(max(0, min(score, 100))))
    if score < MIN_CONFIDENCE:
        return None

    # === Stops & Swing Targets (~20% target for stocks, long-term ETF) ===
    if is_etf:
        t1 = round(current_price * 1.08, 2)
        t2 = round(current_price * 1.15, 2)
        sl = round(current_price * 0.94, 2)
        factors.append("Strategy: Core ETF (Long-Term Compounding)")
    else:
        t1 = round(current_price * (1 + SWING_TRAIL_START_PCT), 2)  # +10% trail trigger
        t2 = round(current_price * (1 + SWING_TP_PCT), 2)           # +20% profit target!
        struct_sl = float(recent_lows.iloc[-1]) * 0.993 if len(recent_lows) else current_price * 0.95
        # Clamp stop risk between MIN_STOP_PCT and MAX_STOP_PCT
        sl = round(max(current_price * (1 - MAX_STOP_PCT), min(struct_sl, current_price * (1 - MIN_STOP_PCT))), 2)
        factors.append(f"Strategy: Medium-Term Swing (Target +{SWING_TP_PCT*100:.0f}%, Setup: {setup_name})")

    risk = current_price - sl
    if risk <= 0:
        return None
    stop_pct = risk / current_price
    rr = safe_div(abs(t2 - current_price), risk, 0.0)
    if rr < MIN_RR:
        return None

    plan = _build_action_plan(current_price, sl, t1, t2, score, is_etf=is_etf)

    if ENABLE_EXTERNAL_DATA:
        factors = enrich_signal_with_external_data(ticker, factors)

    return StockSignal(
        ticker=ticker,
        direction=direction,
        entry=float(current_price),
        stop_loss=float(sl),
        target_1=float(t1),
        target_2=float(t2),
        confidence=score,
        rationale=factors,
        r_r=float(rr),
        risk_pct=float(stop_pct * 100),
        timestamp=timestamp,
        action_plan=plan,
        regime=regime,
        tech_confidence=tech_score,
        factor_score=f_score,
        factor_grade=f_grade,
        factor_summary=f_summary,
        days_to_earnings=days_to_earnings,
    )


def _build_action_plan(entry: float, sl: float, t1: float, t2: float, conf: int, is_etf: bool = False) -> str:
    if is_etf:
        return (
            f"1) ALLOCATION: Core ETF long-term accumulation near {entry:.2f}\n"
            f"2) HORIZON: Multi-quarter / multi-year compounding — hold through minor dips\n"
            f"3) ACCUMULATE: Add tranche on any 3–5% pullback into support\n"
            f"4) HARD STOP: Only exit if major multi-month bear regime confirmed ({sl:.2f})"
        )
    return (
        f"1) ENTER: BUY near {entry:.2f} (Target: +{SWING_TP_PCT*100:.0f}% at {t2:.2f})\n"
        f"2) TRAILING LOCK: At +{SWING_TRAIL_START_PCT*100:.0f}% ({t1:.2f}), raise stop to lock in profit\n"
        f"3) FULL TAKE PROFIT: SELL at +{SWING_TP_PCT*100:.0f}% target ({t2:.2f})\n"
        f"4) HARD STOP: Cut loss at {sl:.2f} (max risk {(entry-sl)/entry*100:.1f}%) — never average down\n"
        f"5) POSITION-AWARE: Telegram bot tracks this position and will alert you to SELL when target is hit!"
    )


# =============================================================================
# TRADE SIMULATION (Swing Investor realistic management)
# =============================================================================

def simulate_trade(signal: StockSignal, future: pd.DataFrame) -> Optional[dict]:
    """Long-only swing trade management: +20% target, +10% trailing stop trigger, structural SL."""
    if future is None or future.empty:
        return None
    if signal.direction != "LONG":
        return None

    entry = signal.entry * (1 + SLIPPAGE)
    sl = signal.stop_loss
    t1 = signal.target_1  # +10% trail trigger
    t2 = signal.target_2  # +20% profit target
    risk = abs(entry - sl)
    if risk <= 0:
        return None

    trail_armed = False
    trail_stop = sl
    realized = 0.0
    exit_price = None
    exit_reason = None
    bars_held = 0

    for _, row in future.iterrows():
        bars_held += 1
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        if np.isnan(high) or np.isnan(low) or np.isnan(close):
            continue

        # 1. Take Profit at +20% Target
        if high >= t2:
            px = t2 * (1 - SLIPPAGE)
            realized = safe_div(px - entry, entry, 0.0) * 100
            exit_price = px
            exit_reason = "TARGET_20PCT_HIT"
            break

        # 2. Activate Trailing Stop once up +10% (lock in +5%)
        if not trail_armed and high >= t1:
            trail_armed = True
            trail_stop = entry * (1 + SWING_TRAIL_LOCK_PCT)

        # 3. Stop loss / Trailing stop check
        cur_stop = max(sl, trail_stop if trail_armed else 0.0)
        if low <= cur_stop:
            px = cur_stop * (1 - SLIPPAGE)
            realized = safe_div(px - entry, entry, 0.0) * 100
            exit_price = px
            exit_reason = "TRAIL_STOP_LOCK" if trail_armed else "STOP_LOSS"
            break

    if exit_price is None:
        last = float(future.iloc[-1]["close"])
        realized = safe_div(last - entry, entry, 0.0) * 100
        exit_price = last
        exit_reason = "TIMEOUT_TIME_EXIT"

    return {
        "ticker": signal.ticker,
        "direction": "LONG",
        "entry": entry,
        "exit": exit_price,
        "stop_loss": sl,
        "target_1": t1,
        "target_2": t2,
        "pnl_pct": realized,
        "exit_reason": exit_reason,
        "confidence": signal.confidence,
        "r_r": signal.r_r,
        "risk_pct": signal.risk_pct,
        "regime": signal.regime,
        "bars_held": bars_held,
        "date": signal.timestamp.strftime("%Y-%m-%d"),
        "rationale": " | ".join(signal.rationale),
    }


# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest(
    tickers: Optional[List[str]] = None,
    start: str = "2024-01-01",
    end: str = "2026-07-01",
    use_approved: Optional[bool] = None,
) -> pd.DataFrame:
    if tickers is None:
        tickers = tradeable_tickers(use_approved=use_approved)

    mode_label = "APPROVED" if (USE_APPROVED_ONLY if use_approved is None else use_approved) else "FULL"
    print("\n" + "=" * 64)
    print(f"STOCK SIGNAL BACKTEST v3.4 — LONG ONLY + {mode_label} universe + BLACKLIST + ATR stops")
    print("=" * 64)
    print(f"Period: {start} -> {end}")
    print(f"Tickers ({len(tickers)}): {', '.join(tickers)}")
    print(
        f"Mode: LONG ONLY | Min conf: {MIN_CONFIDENCE} | Min RR: {MIN_RR} | "
        f"Stop {MIN_STOP_PCT*100:.1f}-{MAX_STOP_PCT*100:.1f}% | "
        f"StackOnly={REQUIRE_STACKED_UP} BullOnly={BULL_REGIME_ONLY}"
    )
    print(f"Hold bars: {HOLD_BARS} | Cooldown bars: {COOLDOWN_BARS}")
    print(f"TP1={TP1_R}R ({int(TP1_FRACTION*100)}%) | TP2={TP2_R}R | BE after {BE_TRIGGER_R}R")
    print("Factors: OFF in backtest (yfinance fundamentals are not point-in-time)")
    print("=" * 64 + "\n")

    print(f"Loading {REGIME_SYMBOL} for market regime...")
    spy = fetch_data(REGIME_SYMBOL, period="730d", interval="1h")
    if spy.empty:
        spy = fetch_data(REGIME_SYMBOL, period="max", interval="1h")

    all_trades: List[dict] = []

    for ticker in tickers:
        yf_sym = STOCKS.get(ticker, {}).get("yf", ticker)
        print(f"Backtesting {ticker}...")
        df = fetch_data(yf_sym, period="730d", interval="1h")
        if df.empty or len(df) < 300:
            df = fetch_data(yf_sym, period="max", interval="1h")
        if df.empty or len(df) < 300:
            print("  insufficient data")
            continue

        # Filter date range (tz-safe)
        ts = pd.to_datetime(df["timestamp"])
        if ts.dt.tz is not None:
            start_ts = pd.Timestamp(start, tz=ts.dt.tz)
            end_ts = pd.Timestamp(end, tz=ts.dt.tz)
        else:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
        df = df[(ts >= start_ts) & (ts <= end_ts)].reset_index(drop=True)
        if len(df) < 300:
            print("  no data in range")
            continue

        trades = []
        last_entry_i = -10_000

        for i in range(220, len(df) - 5, SCAN_STEP):
            if i - last_entry_i < COOLDOWN_BARS:
                continue
            timestamp = df.iloc[i]["timestamp"]
            ts_dt = pd.Timestamp(timestamp).to_pydatetime()
            # Session filter using hour
            hour = ts_dt.hour
            # yfinance US equity 1h bars often in US/Eastern-like hours 9-16
            if hour < 10 or hour > 15:
                continue

            price = float(df.iloc[i]["close"])
            if price <= 0 or np.isnan(price):
                continue

            hist = df.iloc[: i + 1]
            future = df.iloc[i + 1 : i + 1 + HOLD_BARS]
            regime = _regime_from_spy(spy, ts_dt)

            try:
                sig = generate_signal(ticker, hist, price, ts_dt, regime=regime)
            except Exception as e:
                logger.error("Signal error %s %s: %s", ticker, timestamp, e)
                continue

            if not sig:
                continue

            result = simulate_trade(sig, future)
            if not result:
                continue

            trades.append(result)
            last_entry_i = i

        if trades:
            w = sum(1 for t in trades if t["pnl_pct"] > 0)
            print(f"  trades={len(trades)} wins={w} wr={100 * w / len(trades):.1f}% "
                  f"avg={np.mean([t['pnl_pct'] for t in trades]):+.2f}%")
        else:
            print("  trades=0")
        all_trades.extend(trades)

    if not all_trades:
        print("\nNo trades generated. Try widening date range or lowering MIN_CONFIDENCE.")
        return pd.DataFrame()

    out = pd.DataFrame(all_trades)
    mode_label = "APPROVED" if (USE_APPROVED_ONLY if use_approved is None else use_approved) else "FULL"
    _print_report(out, mode_label=mode_label)
    out_name = f"backtest_v33_{mode_label.lower()}_tight_results.csv"
    out.to_csv(out_name, index=False)
    # keep legacy filename too for older workflows
    out.to_csv("backtest_v32_approved_results.csv", index=False)
    print(f"\nSaved: {out_name}")
    print("Also wrote: backtest_v32_approved_results.csv")
    return out


def _print_report(df: pd.DataFrame, mode_label: str = "FULL") -> None:
    total = len(df)
    wins = int((df["pnl_pct"] > 0).sum())
    losses = total - wins
    wr = 100.0 * wins / total
    avg_win = df.loc[df["pnl_pct"] > 0, "pnl_pct"].mean() if wins else 0.0
    avg_loss = df.loc[df["pnl_pct"] <= 0, "pnl_pct"].mean() if losses else 0.0
    gp = df.loc[df["pnl_pct"] > 0, "pnl_pct"].sum()
    gl = abs(df.loc[df["pnl_pct"] <= 0, "pnl_pct"].sum())
    pf = safe_div(gp, gl, 0.0)
    exp = df["pnl_pct"].mean()

    print("\n" + "=" * 64)
    print(f"BACKTEST RESULTS v3.3 — LONG ONLY + {mode_label} + TIGHT")
    print(f"Min conf {MIN_CONFIDENCE} | RR>={MIN_RR} | stack_only={REQUIRE_STACKED_UP} | bull_only={BULL_REGIME_ONLY}")
    print("=" * 64)
    print(f"Total Trades:   {total}")
    print(f"Win Rate:       {wr:.1f}% ({wins}W / {losses}L)")
    print(f"Profit Factor:  {pf:.2f}")
    print(f"Expectancy:     {exp:+.2f}% per trade")
    print(f"Avg Win:        {avg_win:+.2f}%")
    print(f"Avg Loss:       {avg_loss:.2f}%")
    print(f"Median PnL:     {df['pnl_pct'].median():+.2f}%")
    print("\nExit reasons:")
    print(df["exit_reason"].value_counts().to_string())
    print("\nBy regime:")
    if "regime" in df.columns:
        print(
            df.groupby("regime")["pnl_pct"]
            .agg(n="count", wr=lambda s: (s > 0).mean() * 100, avg="mean", total="sum")
            .round(2)
            .to_string()
        )
    print("\nBy ticker (top/bottom by win rate):")
    by_t = (
        df.groupby("ticker")["pnl_pct"]
        .agg(n="count", wr=lambda s: (s > 0).mean() * 100, avg="mean", total="sum")
        .round(2)
        .sort_values("wr", ascending=False)
    )
    print(by_t.to_string())
    print("\nBy confidence bucket:")
    df = df.copy()
    df["conf_bucket"] = pd.cut(
        df["confidence"],
        bins=[0, 84, 89, 94, 100],
        labels=["<=84", "85-89", "90-94", "95+"],
    )
    print(
        df.groupby("conf_bucket", observed=False)["pnl_pct"]
        .agg(n="count", wr=lambda s: (s > 0).mean() * 100, avg="mean")
        .round(2)
        .to_string()
    )
    print("=" * 64)

    if wr >= 55 and exp > 0 and pf >= 1.3:
        print("\nQuality check: PASS — edge looks usable with strict risk rules.")
    elif exp > 0:
        print("\nQuality check: MIXED — positive expectancy but win rate still modest; size small.")
    else:
        print("\nQuality check: FAIL — do not trade live; tighten filters further.")


# =============================================================================
# LIVE SCAN + TELEGRAM
# =============================================================================

def _now_et() -> datetime:
    return datetime.now(tz=ET)


def is_trading_session(dt: Optional[datetime] = None) -> bool:
    """True during Mon–Fri regular hours (ET), excluding holidays, 10:00–15:30."""
    dt = dt or _now_et()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    else:
        dt = dt.astimezone(ET)
    if dt.weekday() >= 5:
        return False
    if dt.strftime("%Y-%m-%d") in HOLIDAYS:
        return False
    return MARKET_OPEN <= dt.time() <= MARKET_CLOSE


def next_scan_eta(dt: Optional[datetime] = None) -> datetime:
    """Next scheduled scan datetime in ET."""
    dt = dt or _now_et()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    else:
        dt = dt.astimezone(ET)
    cursor = dt
    for _ in range(14):  # search up to 2 weeks
        if cursor.weekday() < 5 and cursor.strftime("%Y-%m-%d") not in HOLIDAYS:
            for st in SCAN_TIMES_ET:
                candidate = cursor.replace(hour=st.hour, minute=st.minute, second=0, microsecond=0)
                if candidate > dt:
                    return candidate
        cursor = (cursor + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return dt + timedelta(days=1)


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            data.setdefault("last_alerts", {})
            data.setdefault("last_scan", None)
            data.setdefault("subscribers", [])
            data.setdefault("open_signals", [])   # snapshots for EOD / multi-day tracking
            data.setdefault("last_eod_recap", None)
            return data
        except Exception:
            pass
    return {
        "last_alerts": {},
        "last_scan": None,
        "subscribers": [],
        "open_signals": [],
        "last_eod_recap": None,
    }


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _signal_snapshot(sig: StockSignal) -> dict:
    return {
        "ticker": sig.ticker,
        "direction": sig.direction,
        "entry": float(sig.entry),
        "stop_loss": float(sig.stop_loss),
        "target_1": float(sig.target_1),
        "target_2": float(sig.target_2),
        "confidence": int(sig.confidence),
        "risk_pct": float(sig.risk_pct),
        "r_r": float(sig.r_r),
        "regime": sig.regime,
        "rationale": list(sig.rationale),
        "alerted_at": _now_et().isoformat(),
        "status": "open",
    }


def _align_alert_ts(ts: pd.Series, alert: datetime) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    alert_ts = pd.Timestamp(alert)
    if getattr(ts.dt, "tz", None) is not None:
        if alert_ts.tzinfo is None:
            alert_ts = alert_ts.tz_localize(ET)
        return alert_ts.tz_convert(ts.dt.tz)
    if alert_ts.tzinfo is not None:
        return pd.Timestamp(alert_ts.tz_convert(ET).replace(tzinfo=None))
    return alert_ts


def _score_open_signal(snap: dict) -> dict:
    """Mark-to-market + level hits for one stored alert snapshot."""
    t = snap["ticker"]
    entry = float(snap["entry"])
    sl = float(snap["stop_loss"])
    t1 = float(snap["target_1"])
    t2 = float(snap["target_2"])
    alerted_at = datetime.fromisoformat(snap["alerted_at"])
    meta = STOCKS.get(t, {"yf": t})
    df = fetch_data(meta["yf"], period="30d", interval="1h")
    out = {
        "ticker": t,
        "confidence": int(snap.get("confidence", 0)),
        "entry": entry,
        "stop_loss": sl,
        "target_1": t1,
        "target_2": t2,
        "alerted_at": snap["alerted_at"],
        "last_px": None,
        "mtm_pct": None,
        "hit_sl": False,
        "hit_tp1": False,
        "hit_tp2": False,
        "hit_be": False,
        "status": "NO_DATA",
        "note": "",
    }
    if df.empty:
        return out

    ts = pd.to_datetime(df["timestamp"])
    alert_ts = _align_alert_ts(ts, alerted_at)
    # Prefer bars strictly after alert; if empty, include alert bar close path
    future = df.loc[ts > alert_ts]
    if future.empty:
        future = df.loc[ts >= alert_ts]
    if future.empty:
        out["status"] = "NO_BARS"
        return out

    last_px = float(future.iloc[-1]["close"])
    out["last_px"] = last_px
    out["mtm_pct"] = (last_px - entry) / entry * 100.0 if entry else 0.0

    risk = abs(entry - sl)
    be_lvl = entry + BE_TRIGGER_R * risk if risk > 0 else entry
    hit_sl = hit_tp1 = hit_tp2 = hit_be = False
    exit_status = "OPEN"
    # Walk forward chronologically — first decisive event for simple status
    for _, row in future.iterrows():
        hi, lo = float(row["high"]), float(row["low"])
        if lo <= sl:
            hit_sl = True
        if hi >= t1:
            hit_tp1 = True
        if hi >= t2:
            hit_tp2 = True
        if hi >= be_lvl:
            hit_be = True

    out["hit_sl"] = hit_sl
    out["hit_tp1"] = hit_tp1
    out["hit_tp2"] = hit_tp2
    out["hit_be"] = hit_be

    # Managed status (same priority idea as simulator, simplified for message)
    if hit_tp2 and hit_tp1:
        exit_status = "TP2"
    elif hit_tp2:
        exit_status = "TP2"
    elif hit_tp1 and hit_sl:
        # both touched same window — ambiguous; prefer MTM sign
        exit_status = "TP1" if out["mtm_pct"] >= 0 else "SL"
    elif hit_tp1:
        exit_status = "TP1_PARTIAL"
    elif hit_sl and hit_be:
        exit_status = "BE"
    elif hit_sl:
        exit_status = "SL"
    elif hit_be:
        exit_status = "OPEN_BE"
    else:
        exit_status = "OPEN"

    out["status"] = exit_status
    return out


def build_eod_recap(
    state: Optional[dict] = None,
    day: Optional[datetime] = None,
) -> Tuple[str, List[dict]]:
    """
    Build end-of-day HTML recap for alerts from `day` (ET calendar date).
    Falls back to reconstructing from last_alerts if open_signals empty.
    """
    state = state or _load_state()
    day = day or _now_et()
    day_key = day.astimezone(ET).strftime("%Y-%m-%d") if day.tzinfo else day.strftime("%Y-%m-%d")

    snaps: List[dict] = list(state.get("open_signals") or [])
    # Include today's alerts even if only last_alerts exists (legacy state)
    if not snaps and state.get("last_alerts"):
        snaps = _reconstruct_snaps_from_last_alerts(state["last_alerts"])

    # Filter to alerts from this ET day OR still open from prior days (multi-day book)
    todays: List[dict] = []
    open_book: List[dict] = []
    for s in snaps:
        if s.get("status") not in (None, "open", "OPEN", "OPEN_BE", "TP1_PARTIAL"):
            # still show closed today if alerted today
            pass
        try:
            a = datetime.fromisoformat(s["alerted_at"])
            if a.tzinfo is None:
                a = a.replace(tzinfo=ET)
            a_day = a.astimezone(ET).strftime("%Y-%m-%d")
        except Exception:
            a_day = ""
        if a_day == day_key:
            todays.append(s)
        else:
            # prior open positions still tracked
            if s.get("status", "open") in ("open", "OPEN", "OPEN_BE", "TP1_PARTIAL", None):
                open_book.append(s)

    scored_today = [_score_open_signal(s) for s in todays]
    scored_book = [_score_open_signal(s) for s in open_book]

    lines = [
        f"📋 <b>EOD RECAP</b> — {day_key} ET",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if not scored_today and not scored_book:
        lines.append("No alerts on the book today.")
        lines.append("Cash / wait was the trade.")
        return "\n".join(lines), []

    if scored_today:
        lines.append(f"<b>Today's new alerts ({len(scored_today)})</b>")
        mtms = [r["mtm_pct"] for r in scored_today if r["mtm_pct"] is not None]
        avg = sum(mtms) / len(mtms) if mtms else 0.0
        green = sum(1 for x in mtms if x > 0)
        lines.append(f"Green now: {green}/{len(mtms)} · avg MTM {avg:+.2f}%")
        lines.append("")
        for r in sorted(scored_today, key=lambda x: (x["mtm_pct"] is None, -(x["mtm_pct"] or 0))):
            mtm = f"{r['mtm_pct']:+.2f}%" if r["mtm_pct"] is not None else "n/a"
            last = f"${r['last_px']:.2f}" if r["last_px"] is not None else "?"
            flags = []
            if r["hit_sl"]:
                flags.append("SL")
            if r["hit_tp1"]:
                flags.append("TP1")
            if r["hit_tp2"]:
                flags.append("TP2")
            if r["hit_be"]:
                flags.append("BE+")
            flag_s = f" [{','.join(flags)}]" if flags else ""
            lines.append(
                f"• <b>{r['ticker']}</b> conf {r['confidence']} · {r['status']}{flag_s}\n"
                f"  entry ${r['entry']:.2f} → {last} ({mtm})\n"
                f"  SL ${r['stop_loss']:.2f} · TP1 ${r['target_1']:.2f} · TP2 ${r['target_2']:.2f}"
            )
    else:
        lines.append("<b>Today's new alerts:</b> none")

    if scored_book:
        lines.append("")
        lines.append(f"<b>Still open from prior days ({len(scored_book)})</b>")
        for r in scored_book:
            mtm = f"{r['mtm_pct']:+.2f}%" if r["mtm_pct"] is not None else "n/a"
            last = f"${r['last_px']:.2f}" if r["last_px"] is not None else "?"
            lines.append(f"• <b>{r['ticker']}</b> {r['status']} · {last} ({mtm})")

    lines.append("")
    lines.append("<i>Swing holds can take days. MTM ≠ final P&amp;L until TP/SL/time exit.</i>")
    lines.append("Commands: /recap · /scan · /status")
    return "\n".join(lines), scored_today + scored_book


def _reconstruct_snaps_from_last_alerts(last_alerts: dict) -> List[dict]:
    """Rebuild snapshots when state only has ticker→timestamp (older live runs)."""
    if not last_alerts:
        return []
    spy = fetch_data(REGIME_SYMBOL, period="120d", interval="1h")
    snaps = []
    for t, iso in last_alerts.items():
        try:
            alert_time = datetime.fromisoformat(iso)
        except Exception:
            continue
        meta = STOCKS.get(t, {"yf": t})
        df = fetch_data(meta["yf"], period="120d", interval="1h")
        if df.empty:
            continue
        ts = pd.to_datetime(df["timestamp"])
        alert_ts = _align_alert_ts(ts, alert_time)
        hist = df.loc[ts <= alert_ts].reset_index(drop=True)
        if len(hist) < 100:
            continue
        px = float(hist.iloc[-1]["close"])
        bar_ts = pd.Timestamp(hist.iloc[-1]["timestamp"]).to_pydatetime()
        regime = _regime_from_spy(spy, bar_ts)
        sig = generate_signal(t, hist, px, bar_ts, regime=regime)
        if not sig:
            # store bare levels from price if rebuild fails
            continue
        snap = _signal_snapshot(sig)
        snap["alerted_at"] = iso
        snaps.append(snap)
    return snaps


def _position_size(entry: float, stop: float, balance: float = ACCOUNT_BALANCE) -> Tuple[int, float, float]:
    risk_amt = balance * (MAX_RISK_PCT / 100.0)
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return 0, 0.0, 0.0
    shares = int(risk_amt / stop_dist)
    pos_val = shares * entry
    max_pos = balance * (POSITION_SIZE_PCT / 100.0)
    if pos_val > max_pos and entry > 0:
        shares = int(max_pos / entry)
        pos_val = shares * entry
        risk_amt = shares * stop_dist
    return max(shares, 0), pos_val, risk_amt


def format_signal_html(sig: StockSignal, balance: float = ACCOUNT_BALANCE) -> str:
    shares, pos_val, risk_amt = _position_size(sig.entry, sig.stop_loss, balance)
    why = "\n".join(f"• {r}" for r in sig.rationale)
    if sig.factor_score is not None:
        earn = (f" | earnings in {sig.days_to_earnings}d"
                if sig.days_to_earnings is not None else "")
        fundamentals = (
            f"🏢 Fundamentals: <b>{sig.factor_score:.0f}/100 ({sig.factor_grade})</b>{earn}\n"
            f"📊 Technical: <b>{sig.tech_confidence}</b>\n"
        )
    else:
        fundamentals = ""
    return (
        f"🟢 <b>BUY {sig.ticker}</b>  |  conf <b>{sig.confidence}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Entry: <b>${sig.entry:.2f}</b>\n"
        f"⛔ Stop: <b>${sig.stop_loss:.2f}</b> ({sig.risk_pct:.2f}%)\n"
        f"1️⃣ TP1 (55%): <b>${sig.target_1:.2f}</b>\n"
        f"2️⃣ TP2 (45%): <b>${sig.target_2:.2f}</b>\n"
        f"📈 R:R to TP2: <b>1:{sig.r_r:.1f}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{fundamentals}"
        f"📦 Shares (on ${balance:,.0f}): <b>{shares}</b>\n"
        f"💵 Position ≈ ${pos_val:,.0f} | Risk ≈ ${risk_amt:,.0f}\n"
        f"🌐 Regime: {sig.regime.upper()}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Why</b>\n{why}\n\n"
        f"<b>Plan</b>\n"
        f"1) BUY near ${sig.entry:.2f}\n"
        f"2) Hard stop ${sig.stop_loss:.2f} — full exit\n"
        f"3) Sell ~55% at TP1 ${sig.target_1:.2f}, stop → breakeven\n"
        f"4) Sell rest at TP2 ${sig.target_2:.2f}\n"
        f"5) Time stop ~10 sessions if nothing hits\n"
        f"6) No shorting · no averaging down\n\n"
        f"🕐 {sig.timestamp.strftime('%Y-%m-%d %H:%M')} | LONG ONLY v3.4"
    )


def scan_live(
    tickers: Optional[List[str]] = None,
    use_approved: Optional[bool] = None,
    quiet: bool = False,
) -> List[StockSignal]:
    if tickers is None:
        tickers = tradeable_tickers(use_approved=use_approved)
    spy = fetch_data(REGIME_SYMBOL, period="730d", interval="1h")
    now = _now_et()
    # Align regime bar time with latest SPY bar when possible
    if spy is not None and not spy.empty:
        ts_reg = pd.Timestamp(spy.iloc[-1]["timestamp"]).to_pydatetime()
    else:
        ts_reg = now
    regime = _regime_from_spy(spy, ts_reg)
    if not quiet:
        print(f"\nMarket regime ({REGIME_SYMBOL}): {regime.upper()} | LONG ONLY | min conf {MIN_CONFIDENCE}")
        print(f"Scanning {len(tickers)} symbols: {', '.join(tickers)}")
        if not is_trading_session(now):
            nxt = next_scan_eta(now)
            print(f"Note: outside scan window now ({now.strftime('%Y-%m-%d %H:%M %Z')}). "
                  f"Next scheduled scan: {nxt.strftime('%Y-%m-%d %H:%M %Z')}")

    signals: List[StockSignal] = []
    candidates: List[Tuple[str, pd.DataFrame, float, datetime]] = []
    for t in tickers:
        meta = STOCKS.get(t, {"yf": t})
        df = fetch_data(meta["yf"], period="730d", interval="1h")
        if df.empty:
            continue
        px = float(df.iloc[-1]["close"])
        ts = pd.Timestamp(df.iloc[-1]["timestamp"]).to_pydatetime()
        # Cheap technical pre-pass so we only pull fundamentals for real candidates
        if generate_signal(t, df, px, ts, regime=regime) is None:
            continue
        candidates.append((t, df, px, ts))

    factors = _factor_map([c[0] for c in candidates], quiet=quiet)

    for t, df, px, ts in candidates:
        sig = generate_signal(t, df, px, ts, regime=regime, factor=factors.get(t))
        if sig:
            signals.append(sig)
            if not quiet:
                print(f"\n*** BUY {t} conf={sig.confidence} ***")
                print(f"Entry {sig.entry:.2f} | SL {sig.stop_loss:.2f} | TP1 {sig.target_1:.2f} | TP2 {sig.target_2:.2f}")
                print(f"Risk {sig.risk_pct:.2f}% | R:R to TP2 {sig.r_r:.2f}")
                print("Why:", "; ".join(sig.rationale))
                print(sig.action_plan)

    signals = _rank_and_cap(signals)

    if not quiet:
        if not signals:
            print("\nNo high-quality BUY signals right now. Waiting (cash) is valid.")
        else:
            print(f"\n--- {len(signals)} BUY signal(s) ---")
            for s in sorted(signals, key=lambda x: x.confidence, reverse=True):
                print(f"  {s.ticker:6s} conf={s.confidence} entry={s.entry:.2f} sl={s.stop_loss:.2f} tp1={s.target_1:.2f}")
    return signals


def _factor_map(tickers: List[str], quiet: bool = True) -> Dict[str, Any]:
    """Fundamental/analyst scores for shortlisted tickers (empty when disabled)."""
    if not tickers or not USE_FACTORS or not HAS_FACTORS:
        return {}
    if not quiet:
        print(f"\nScoring fundamentals for {len(tickers)} candidate(s)...")
    try:
        return factor_engine.fetch_factor_scores(
            tickers,
            yf_map={t: STOCKS.get(t, {}).get("yf", t) for t in tickers},
            progress=not quiet,
        )
    except Exception as exc:
        logger.error("factor scoring failed: %s", exc)
        return {}


def _rank_and_cap(signals: List[StockSignal]) -> List[StockSignal]:
    """Keep only the best MAX_SIGNALS_PER_SCAN setups (conviction + fundamentals)."""
    if len(signals) <= MAX_SIGNALS_PER_SCAN:
        return signals

    def rank(s: StockSignal) -> float:
        base = float(s.confidence)
        if s.factor_score is not None:
            base = 0.6 * base + 0.4 * float(s.factor_score)
        return base + s.r_r

    return sorted(signals, key=rank, reverse=True)[:MAX_SIGNALS_PER_SCAN]


def _filter_fresh_signals(signals: List[StockSignal], state: dict) -> List[StockSignal]:
    """Drop tickers already alerted within SIGNAL_COOLDOWN_HOURS."""
    now = _now_et()
    last = state.get("last_alerts", {})
    fresh = []
    for s in signals:
        prev = last.get(s.ticker)
        if prev:
            try:
                prev_dt = datetime.fromisoformat(prev)
                if prev_dt.tzinfo is None:
                    prev_dt = prev_dt.replace(tzinfo=ET)
                if now - prev_dt < timedelta(hours=SIGNAL_COOLDOWN_HOURS):
                    continue
            except Exception:
                pass
        fresh.append(s)
    return fresh


def _mark_alerted(signals: List[StockSignal], state: dict) -> dict:
    now = _now_et().isoformat()
    last = state.setdefault("last_alerts", {})
    open_sigs: List[dict] = state.setdefault("open_signals", [])
    for s in signals:
        last[s.ticker] = now
        snap = _signal_snapshot(s)
        snap["alerted_at"] = now
        # replace existing open row for same ticker
        open_sigs = [x for x in open_sigs if x.get("ticker") != s.ticker]
        open_sigs.append(snap)
    state["open_signals"] = open_sigs
    state["last_scan"] = now
    return state


def set_windows_keep_awake(enable: bool = True) -> None:
    """Request Windows OS to stay awake and avoid system standby during live trading."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        if enable:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
            logger.info("Windows OS keep-awake activated for market trading hours.")
        else:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception as e:
        logger.debug("Failed to set Windows thread execution state: %s", e)


class TelegramSignalService:
    """Telegram bot + scheduled market scans."""

    def __init__(self, token: str, admin_ids: List[int]):
        if not HAS_TELEGRAM:
            raise RuntimeError("python-telegram-bot not installed. pip install python-telegram-bot")
        if not token or token == "YOUR_BOT_TOKEN_HERE":
            raise RuntimeError("Set TELEGRAM_BOT_TOKEN in .env")
        self.token = token
        self.admin_ids = set(admin_ids)
        self.state = _load_state()
        # Merge admins + any prior /start subscribers
        subs = set(self.state.get("subscribers", []))
        subs |= self.admin_ids
        self.subscribers: Set[int] = {int(x) for x in subs}
        self.state["subscribers"] = sorted(self.subscribers)
        _save_state(self.state)

        # Robust connection pool with resilient read/connect timeouts
        request_client = HTTPXRequest(
            connection_pool_size=8,
            connect_timeout=15.0,
            read_timeout=30.0,
            write_timeout=15.0,
        )
        self.app = Application.builder().token(token).request(request_client).build()
        self._setup_handlers()
        self._scan_lock = asyncio.Lock()

    def _setup_handlers(self) -> None:
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("scan", self.cmd_scan))
        self.app.add_handler(CommandHandler("watchlist", self.cmd_watchlist))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("risk", self.cmd_risk))
        self.app.add_handler(CommandHandler("recap", self.cmd_recap))
        self.app.add_handler(CommandHandler("committee", self.cmd_committee))
        self.app.add_handler(CommandHandler("debate", self.cmd_committee))
        self.app.add_handler(CommandHandler("bought", self.cmd_bought))
        self.app.add_handler(CommandHandler("buy", self.cmd_bought))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("portfolio", self.cmd_positions))
        self.app.add_handler(CommandHandler("sold", self.cmd_sold))
        self.app.add_handler(CommandHandler("close", self.cmd_sold))
        self.app.add_handler(CallbackQueryHandler(self.cb_button))
        self.app.add_error_handler(self.error_handler)

    def _is_allowed(self, user_id: int) -> bool:
        # If ADMIN_IDS set, restrict; else allow anyone who /start'ed
        if self.admin_ids:
            return user_id in self.admin_ids or user_id in self.subscribers
        return True

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if self.admin_ids and uid not in self.admin_ids:
            await update.message.reply_text("Unauthorized. Add your Telegram user id to ADMIN_IDS in .env")
            return
        self.subscribers.add(uid)
        self.state["subscribers"] = sorted(self.subscribers)
        _save_state(self.state)
        nxt = next_scan_eta()
        times = ", ".join(t.strftime("%H:%M") for t in SCAN_TIMES_ET)
        msg = (
            "📈 <b>Swing Investor Agent v3.5</b>\n\n"
            "Targeting ~20% gains on single stocks & Core ETF long-term compounding.\n\n"
            f"<b>Scanned universe:</b> {len(tradeable_tickers())} names\n"
            f"<b>Scan times (US/Eastern):</b> {times}\n"
            f"<b>Session:</b> {MARKET_OPEN.strftime('%H:%M')}–{MARKET_CLOSE.strftime('%H:%M')} ET, Mon–Fri\n"
            f"<b>Next scan:</b> {nxt.strftime('%Y-%m-%d %H:%M %Z')}\n\n"
            "💼 <b>Portfolio & Position Tracking:</b>\n"
            "• <code>/bought TICKER PRICE [SHARES]</code> — track a bought position\n"
            "• <code>/positions</code> — view live holdings & progress to +20%\n"
            "• <code>/sold TICKER PRICE</code> — record exit & archive\n\n"
            "🏛️ <b>Multi-Agent Deliberation:</b>\n"
            "• <code>/committee &lt;ticker&gt;</code> — convene debate on any symbol\n"
            "• <code>/committee all</code> — scan 86 stocks and return high-conviction buys\n"
            "• <code>/scan</code> — force immediate universe scan\n\n"
            "🛡️ <b>Position-Aware Sells:</b> The bot will ONLY alert to sell stocks you actually own!"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📚 <b>Agent Commands</b>\n\n"
            "<b>Multi-Agent Committee:</b>\n"
            "• /committee &lt;ticker&gt; — convene 4-agent debate on a ticker\n"
            "• /committee all — screen all 86 tickers for approved setups\n"
            "• /scan — run full universe scan right now\n"
            "• /watchlist — list all scanned tickers\n\n"
            "<b>Position-Aware Portfolio:</b>\n"
            "• /bought &lt;ticker&gt; &lt;price&gt; [shares] — register a bought stock\n"
            "• /positions — view active holdings, live PnL & distance to +20%\n"
            "• /sold &lt;ticker&gt; &lt;price&gt; — close position & record to journal\n\n"
            "<b>System & Utilities:</b>\n"
            "• /status — bot health, scan schedule & subscribers\n"
            "• /recap — EOD review & performance scorecard\n"
            "• /risk balance entry stop — calculate shares for 1% risk\n"
            "• /start — subscribe to live Telegram alerts\n",
            parse_mode="HTML",
        )

    async def cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        names = tradeable_tickers()
        lines = []
        for i in range(0, len(names), 5):
            lines.append(" · ".join(names[i:i + 5]))
        mode = "CORE approved" if USE_APPROVED_ONLY else "FULL universe"
        await update.message.reply_text(
            f"📋 <b>LONG watchlist</b> ({mode}, {len(names)})\n\n" + "\n".join(lines)
            + f"\n\nMin conf {MIN_CONFIDENCE} · scans {len(SCAN_TIMES_ET)}x/session ET",
            parse_mode="HTML",
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        now = _now_et()
        nxt = next_scan_eta(now)
        last = self.state.get("last_scan") or "never"
        open_flag = "OPEN" if is_trading_session(now) else "CLOSED"
        await update.message.reply_text(
            "🤖 <b>Status</b>\n"
            f"Now: {now.strftime('%Y-%m-%d %H:%M %Z')} ({open_flag})\n"
            f"Next scan: {nxt.strftime('%Y-%m-%d %H:%M %Z')}\n"
            f"Last scan: {last}\n"
            f"Subscribers: {len(self.subscribers)}\n"
            f"Watchlist: {len(tradeable_tickers())} | conf≥{MIN_CONFIDENCE}\n"
            f"Cooldown-alert cooldown: {SIGNAL_COOLDOWN_HOURS}h\n",
            parse_mode="HTML",
        )

    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args or []
        if len(args) < 3:
            await update.message.reply_text("Usage: /risk 25000 225.50 218.00")
            return
        try:
            bal, entry, stop = float(args[0]), float(args[1]), float(args[2])
            shares, pos, risk = _position_size(entry, stop, bal)
            await update.message.reply_text(
                f"💰 Shares: <b>{shares}</b>\nPosition ≈ ${pos:,.0f}\nRisk ≈ ${risk:,.0f} "
                f"({MAX_RISK_PCT}% of ${bal:,.0f})",
                parse_mode="HTML",
            )
        except Exception:
            await update.message.reply_text("Invalid numbers. Example: /risk 25000 225.50 218.00")

    async def cmd_committee(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self._is_allowed(uid):
            await update.message.reply_text("Unauthorized.")
            return
        args = context.args or []
        target = (args[0] if args else "ALL").upper()

        if target in ("ALL", "SCAN"):
            await update.message.reply_text(
                "🏛️ <b>Screening Entire Universe (86 stocks)…</b>\n"
                "The Committee is evaluating all symbols and will report <b>only actionable BUY setups</b> worth investing.\n"
                "<i>Please allow 60–90 seconds…</i>",
                parse_mode="HTML",
            )
            try:
                loop = asyncio.get_running_loop()
                approved = await loop.run_in_executor(None, lambda: scan_committee_universe(only_approved_buys=True))
                if not approved:
                    await update.message.reply_text(
                        "🟡 <b>Committee Universe Scan Complete</b>\n\n"
                        "<b>Verdict: 100% CASH</b>\n"
                        "The 4 agents screened all 86 symbols. Due to market regime / lack of high-probability confluence, "
                        "<b>0 stocks</b> currently meet our strict investment threshold.\n\n"
                        "<i>Preserving capital is the highest priority.</i>",
                        parse_mode="HTML",
                    )
                    return

                await update.message.reply_text(
                    f"🟢 <b>Committee Found {len(approved)} Approved BUY Setup(s)!</b>",
                    parse_mode="HTML",
                )
                for rep in approved:
                    lines = [
                        f"🎯 <b>APPROVED BUY: {rep.ticker}</b>",
                        f"• <b>Price:</b> ${rep.current_price:.2f}",
                        f"• <b>Entry:</b> ${rep.entry_price:.2f} | <b>Stop:</b> ${rep.stop_loss:.2f} | <b>TP1:</b> ${rep.take_profit_1:.2f}",
                        f"• <b>Sizing:</b> {rep.shares} shares (Risk: {rep.deliberation.get('risk', {}).get('portfolio_risk_pct', 0)}%)",
                        f"• <b>Tech Score:</b> {rep.tech_score}/100 | <b>Sentiment:</b> {rep.sent_stance}",
                        f"• <b>Executive Summary:</b> {rep.executive_summary}",
                    ]
                    await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            except Exception as e:
                logger.error("Committee all scan failed: %s", e)
                await update.message.reply_text(f"Committee universe scan failed: {e}")
            return

        ticker = target
        await update.message.reply_text(f"🏛️ <i>Convening Gen AI Multi-Agent Committee for <b>{ticker}</b>… (~15s)</i>", parse_mode="HTML")
        try:
            loop = asyncio.get_running_loop()
            is_etf = STOCKS.get(ticker, {}).get("sector") == "ETF" or ticker in (
                "SPY", "VOO", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "SMH"
            )

            # Convene Gen AI Multi-Agent System
            if HAS_GENAI and genai_agents:
                bb = await loop.run_in_executor(
                    None,
                    lambda: genai_agents.supervisor.convene_committee(
                        ticker=ticker,
                        regime="BULL",
                        account_balance=ACCOUNT_BALANCE,
                        is_etf=is_etf
                    )
                )
                verdict_icon = "🟢" if bb.verdict == "EXECUTE_BUY" else ("🔵" if bb.verdict == "CORE_ACCUMULATE" else ("🟡" if bb.verdict == "HOLD_CASH" else "🔴"))
                risk_info = bb.risk
                risk_approved = risk_info.get("approved", False)
                risk_badge = "APPROVED ✅" if risk_approved else "VETOED ❌"
                tech_info = bb.technicals
                sent_news = bb.sentiment.get("news", {})

                lines = [
                    f"🏛️ <b>GEN AI INVESTMENT COMMITTEE MEMO</b>",
                    f"Ticker: <b>{bb.ticker}</b> | Regime: <b>{bb.macro_regime.upper()}</b>\n",
                    f"<b>Verdict:</b> {verdict_icon} <code>{bb.verdict}</code>",
                    f"<b>Executive Memorandum:</b>\n{bb.investment_memo}\n",
                    f"<b>Specialized Agent Deliberations:</b>",
                    f"• <b>Technicals Worker:</b> {tech_info.get('trend', 'NEUTRAL')} (Confidence: {bb.confidence}/100)",
                ]
                if bb.entry_price and bb.stop_loss and bb.take_profit_2:
                    lines.append(f"  Entry: ${bb.entry_price:.2f} | Stop: ${bb.stop_loss:.2f} | Target (+20%): <b>${bb.take_profit_2:.2f}</b>")
                elif bb.entry_price:
                    lines.append(f"  Reference Price: ${bb.entry_price:.2f}")

                lines.append(f"• <b>Sentiment Worker:</b> {sent_news.get('sentiment_stance', 'NEUTRAL')} (Catalyst Score: {sent_news.get('catalyst_sentiment_score', 0)})")
                sample_news = sent_news.get("headlines", [])
                if sample_news:
                    lines.append(f"  Catalyst: <i>{sample_news[0].get('title', '')[:60]}…</i>")

                lines.append(f"• <b>Chief Risk Officer:</b> {risk_badge}")
                if risk_approved:
                    lines.append(f"  Allocation: <b>{bb.shares} shares</b> (${risk_info.get('capital_required', 0):,.0f}) | Max Risk: ${risk_info.get('dollar_risk', 0):,.0f} ({risk_info.get('portfolio_risk_pct', 0)}%)")
                else:
                    lines.append(f"  Veto: {risk_info.get('veto_reason')}")

                # Human-in-the-Loop Interactive Keyboard
                reply_markup = None
                if bb.approval_status == "PENDING_APPROVAL" and bb.human_approval_id:
                    reply_markup = InlineKeyboardMarkup([
                        [
                            InlineKeyboardButton(f"✅ Approve & Buy ({bb.shares} sh)", callback_data=f"hitl_approve:{bb.human_approval_id}"),
                            InlineKeyboardButton("❌ Reject / Pass", callback_data=f"hitl_reject:{bb.human_approval_id}")
                        ],
                        [
                            InlineKeyboardButton("🔍 View Deliberation Trace", callback_data=f"hitl_trace:{bb.human_approval_id}")
                        ]
                    ])

                await update.message.reply_text("\n".join(lines), parse_mode="HTML", reply_markup=reply_markup)
            else:
                report = await loop.run_in_executor(None, lambda: run_committee_analysis(ticker, account_balance=ACCOUNT_BALANCE))
                verdict_icon = "🟢" if report.verdict == "EXECUTE_BUY" else ("🟡" if report.verdict == "HOLD_CASH" else "🔴")
                risk_badge = "APPROVED ✅" if report.risk_approved else "VETOED ❌"
                lines = [
                    f"🏛️ <b>INVESTMENT COMMITTEE MEMORANDUM</b>",
                    f"Ticker: <b>{report.ticker}</b> | Regime: <b>{report.regime.upper()}</b>\n",
                    f"<b>Verdict:</b> {verdict_icon} <code>{report.verdict}</code>",
                    f"<b>Executive Summary:</b> {report.executive_summary}\n",
                ]
                await update.message.reply_text("\n".join(lines), parse_mode="HTML")
        except Exception as e:
            logger.error("Committee error for %s: %s", ticker, e)
            await update.message.reply_text(f"Committee analysis failed for {ticker}: {e}")

    async def cmd_recap(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self._is_allowed(uid):
            await update.message.reply_text("Unauthorized.")
            return
        await update.message.reply_text("📋 Building recap… (~30–90s)")
        try:
            loop = asyncio.get_running_loop()
            self.state = _load_state()
            html, _ = await loop.run_in_executor(None, lambda: build_eod_recap(self.state, _now_et()))
            # Telegram hard limit ~4096; split if needed
            if len(html) <= 4000:
                await update.message.reply_text(html, parse_mode="HTML")
            else:
                chunk = ""
                for line in html.split("\n"):
                    if len(chunk) + len(line) + 1 > 3900:
                        await update.message.reply_text(chunk, parse_mode="HTML")
                        chunk = line + "\n"
                    else:
                        chunk += line + "\n"
                if chunk.strip():
                    await update.message.reply_text(chunk, parse_mode="HTML")
        except Exception as e:
            logger.error("Recap failed: %s", e)
            await update.message.reply_text(f"Recap failed: {e}")

    async def maybe_send_eod_recap(self, now: Optional[datetime] = None) -> None:
        """Send once-per-day EOD scorecard after EOD_RECAP_TIME ET on session days."""
        now = now or _now_et()
        if now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        else:
            now = now.astimezone(ET)
        if now.weekday() >= 5 or now.strftime("%Y-%m-%d") in HOLIDAYS:
            return
        if now.time() < EOD_RECAP_TIME:
            return
        day_key = now.strftime("%Y-%m-%d")
        self.state = _load_state()
        if self.state.get("last_eod_recap") == day_key:
            return
        try:
            loop = asyncio.get_running_loop()
            html, _ = await loop.run_in_executor(None, lambda: build_eod_recap(self.state, now))
            await self.broadcast(html)
            self.state["last_eod_recap"] = day_key
            _save_state(self.state)
            logger.info("EOD recap sent for %s", day_key)
        except Exception as e:
            logger.error("EOD recap failed: %s", e)

    async def cmd_bought(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Register an owned position to monitor for the ~20% target & position-aware exit alerts."""
        uid = update.effective_user.id
        if not self._is_allowed(uid):
            await update.message.reply_text("Unauthorized.")
            return
        args = context.args or []
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: <code>/bought TICKER PRICE [SHARES]</code>\n"
                "Example: <code>/bought NVDA 113.25 25</code>\n"
                "Example: <code>/bought AMZN 172.00</code>",
                parse_mode="HTML",
            )
            return

        ticker = args[0].upper().strip()
        try:
            entry_price = float(args[1])
            shares = int(args[2]) if len(args) > 2 else 10
        except ValueError:
            await update.message.reply_text("Invalid price or share count number.")
            return

        is_etf = STOCKS.get(ticker, {}).get("sector") == "ETF" or ticker in (
            "SPY", "VOO", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "SMH"
        )
        if is_etf:
            tp1 = round(entry_price * 1.08, 2)
            tp2 = round(entry_price * 1.15, 2)
            sl = round(entry_price * 0.94, 2)
            strat_desc = "Core ETF (Long-Term Compounding)"
        else:
            tp1 = round(entry_price * (1 + SWING_TRAIL_START_PCT), 2)  # +10%
            tp2 = round(entry_price * (1 + SWING_TP_PCT), 2)           # +20%
            sl = round(entry_price * (1 - MAX_STOP_PCT), 2)           # -7.5%
            strat_desc = f"Medium-Term Swing (Target: +{SWING_TP_PCT*100:.0f}%)"

        pos_id = open_committee_position(
            ticker=ticker,
            entry_price=entry_price,
            shares=shares,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
        )

        msg = (
            f"📌 <b>Position Registered: #{pos_id} {ticker}</b>\n\n"
            f"• <b>Entry:</b> ${entry_price:.2f} ({shares} shares ≈ ${entry_price * shares:,.2f})\n"
            f"• <b>Target (+{SWING_TP_PCT*100:.0f}%):</b> <b>${tp2:.2f}</b>\n"
            f"• <b>Trailing Trigger (+{SWING_TRAIL_START_PCT*100:.0f}%):</b> ${tp1:.2f}\n"
            f"• <b>Stop Loss:</b> ${sl:.2f} (-{(entry_price - sl)/entry_price*100:.1f}%)\n"
            f"• <b>Strategy:</b> {strat_desc}\n\n"
            f"🤖 <i>The Committee is actively tracking this holding. You will receive an immediate SELL alert when target is reached or trailing stop is raised!</i>\n"
            f"View all open holdings anytime with <code>/positions</code>."
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View all active open positions and live performance toward the 20% target."""
        uid = update.effective_user.id
        if not self._is_allowed(uid):
            await update.message.reply_text("Unauthorized.")
            return

        open_pos = get_open_positions()
        if not open_pos:
            await update.message.reply_text(
                "💼 <b>Portfolio Status</b>\n\n"
                "No open positions currently tracked.\n\n"
                "When you buy a stock, log it with:\n"
                "<code>/bought TICKER PRICE [SHARES]</code>\n"
                "<i>Example: /bought NVDA 113.25 50</i>\n\n"
                "The bot will automatically monitor it and alert you when to SELL at +20%!",
                parse_mode="HTML",
            )
            return

        lines = ["💼 <b>Active Portfolio Holdings</b>\n"]
        total_val = 0.0
        total_pnl_dlr = 0.0

        for p in open_pos:
            ticker = p["ticker"]
            entry_px = float(p["entry_price"])
            shares = int(p["shares"])
            sl = float(p["stop_loss"])
            tp2 = float(p["take_profit_2"])
            trail = "🛡️ TRAIL ACTIVE" if p.get("trailing_stop_active") else "NORMAL"

            cur_px = entry_px
            try:
                df = fetch_data(ticker, period="5d", interval="1h")
                if df is not None and not df.empty:
                    cur_px = float(df.iloc[-1]["close"])
            except Exception:
                pass

            pnl_pct = safe_div(cur_px - entry_px, entry_px, 0.0) * 100
            pnl_dlr = (cur_px - entry_px) * shares
            val = cur_px * shares
            dist_tp = safe_div(tp2 - cur_px, cur_px, 0.0) * 100
            total_val += val
            total_pnl_dlr += pnl_dlr

            pnl_icon = "🟢" if pnl_pct >= 0 else "🔴"
            lines.append(
                f"{pnl_icon} <b>{ticker}</b> (#{p['id']}): {shares} sh @ ${entry_px:.2f}\n"
                f"   • Current: <b>${cur_px:.2f}</b> | PnL: <b>{pnl_pct:+.1f}%</b> (${pnl_dlr:+,.2f})\n"
                f"   • Target (+20%): ${tp2:.2f} ({dist_tp:+.1f}% to go)\n"
                f"   • Stop: ${sl:.2f} ({trail})\n"
            )

        lines.append(f"<b>Total Open Value:</b> ${total_val:,.2f} | <b>Total PnL:</b> ${total_pnl_dlr:+,.2f}")
        lines.append("\n<i>To close a sold holding: <code>/sold TICKER PRICE</code></i>")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def cmd_sold(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close an owned holding and record final realized gain to committee journal."""
        uid = update.effective_user.id
        if not self._is_allowed(uid):
            await update.message.reply_text("Unauthorized.")
            return
        args = context.args or []
        if len(args) < 2:
            await update.message.reply_text("Usage: <code>/sold TICKER EXIT_PRICE</code>\nExample: <code>/sold NVDA 136.00</code>", parse_mode="HTML")
            return

        ticker = args[0].upper().strip()
        try:
            exit_px = float(args[1])
        except ValueError:
            await update.message.reply_text("Invalid exit price number.")
            return

        open_pos = [p for p in get_open_positions() if p["ticker"] == ticker]
        if not open_pos:
            await update.message.reply_text(f"No open position found for <b>{ticker}</b>. Check <code>/positions</code>", parse_mode="HTML")
            return

        target_pos = open_pos[0]
        res = close_committee_position(target_pos["id"], exit_px, exit_reason="MANUAL_BROKER_EXIT")
        if not res:
            await update.message.reply_text("Error closing position.")
            return

        pnl_pct = res["pnl_pct"]
        pnl_dlr = res["pnl_dollars"]
        icon = "🎉" if pnl_pct > 0 else "🛑"
        msg = (
            f"{icon} <b>Position Closed: {ticker}</b>\n\n"
            f"• <b>Entry:</b> ${res['entry_price']:.2f} ➔ <b>Exit:</b> ${exit_px:.2f}\n"
            f"• <b>Shares:</b> {res['shares']}\n"
            f"• <b>Realized Gain:</b> <b>{pnl_pct:+.1f}%</b> (${pnl_dlr:+,.2f})\n"
            f"• <b>Archived:</b> Saved to Committee Journal memory for post-trade analysis."
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self._is_allowed(uid):
            await update.message.reply_text("Unauthorized.")
            return
        n = len(tradeable_tickers())
        await update.message.reply_text(f"Scanning {n} tickers... (~2-4 min on full list)")
        await self.run_scan_and_notify(force=True, reply_chat_id=update.effective_chat.id)

    async def cb_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        await q.answer()
        data = q.data or ""
        uid = update.effective_user.id
        if not self._is_allowed(uid):
            await q.edit_message_text("Unauthorized.")
            return

        if data.startswith("hitl_approve:"):
            approval_id = int(data.split(":")[1])
            res = genai_agents.handle_hitl_action(approval_id, "APPROVE", user_id=uid)
            if res.get("status") == "APPROVED":
                t = res["ticker"]
                await q.edit_message_text(
                    q.message.text_html + f"\n\n🟢 <b>HUMAN APPROVED & EXECUTED!</b>\n"
                    f"Position added to portfolio: <b>{res['shares']} shares of {t}</b> at ${res['entry_price']:.2f}.\n"
                    f"🎯 Target (+20%): <b>${res['target_20pct']:.2f}</b> | Stop Loss: ${res['stop_loss']:.2f}.\n"
                    f"Autonomous exit engine is now actively monitoring this holding!",
                    parse_mode="HTML",
                )
            elif res.get("status") == "ALREADY_APPROVED":
                await q.edit_message_text(q.message.text_html + f"\n\nℹ️ {res.get('message')}", parse_mode="HTML")
            else:
                await q.edit_message_text(q.message.text_html + f"\n\n⚠️ Error: {res.get('message', 'Failed to approve')}", parse_mode="HTML")

        elif data.startswith("hitl_reject:"):
            approval_id = int(data.split(":")[1])
            res = genai_agents.handle_hitl_action(approval_id, "REJECT", user_id=uid)
            await q.edit_message_text(
                q.message.text_html + f"\n\n🔴 <b>HUMAN REJECTED / PASSED</b>\n"
                f"Recommendation for {res.get('ticker')} was passed. Capital preserved in cash. Logged to journal.",
                parse_mode="HTML",
            )

        elif data.startswith("hitl_trace:"):
            approval_id = int(data.split(":")[1])
            res = genai_agents.handle_hitl_action(approval_id, "GET_TRACE", user_id=uid)
            trace_text = (
                f"🧠 <b>MULTI-AGENT REASONING TRACE: {res.get('ticker')}</b>\n"
                f"<b>Confidence:</b> {res.get('confidence')}/100\n"
                f"<b>Timestamp:</b> {res.get('created_at')}\n\n"
                f"<b>Executive Synthesis:</b>\n{res.get('thesis')}"
            )
            await q.message.reply_text(trace_text, parse_mode="HTML")

        elif q.data == "take":
            await q.edit_message_text(q.message.text_html + "\n\n✅ <b>Marked TAKEN</b>", parse_mode="HTML")
        elif q.data == "skip":
            await q.edit_message_text(q.message.text_html + "\n\n❌ <b>Skipped</b>", parse_mode="HTML")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error("Telegram error: %s", context.error)

    async def broadcast(self, text: str, reply_markup=None, chat_id: Optional[int] = None):
        targets = [chat_id] if chat_id else list(self.subscribers)
        for uid in targets:
            if uid is None:
                continue
            try:
                await self.app.bot.send_message(uid, text, parse_mode="HTML", reply_markup=reply_markup)
            except Exception as e:
                logger.error("Send failed to %s: %s", uid, e)

    async def run_scan_and_notify(self, force: bool = False, reply_chat_id: Optional[int] = None):
        async with self._scan_lock:
            now = _now_et()
            if not force and not is_trading_session(now):
                logger.info("Skip scan — session closed (%s)", now.isoformat())
                return

            loop = asyncio.get_running_loop()

            # STEP 1: Position-Aware Exit Checks (Only checks tickers the user owns!)
            exit_alerts = await loop.run_in_executor(None, check_open_positions_exits)
            for alert in exit_alerts:
                pos = alert["position"]
                t = pos["ticker"]
                if alert["type"] == "TAKE_PROFIT_20PCT":
                    msg = (
                        f"🎯 <b>PORTFOLIO SELL ALERT: +20% TARGET HIT!</b> 🎯\n\n"
                        f"<b>{t}</b> has reached your profit target!\n"
                        f"• <b>Entry:</b> ${pos['entry_price']:.2f}\n"
                        f"• <b>Current Price:</b> <b>${alert['current_price']:.2f}</b>\n"
                        f"• <b>Gain:</b> <b>+{alert['gain_pct']:.1f}%</b>\n"
                        f"• <b>Target:</b> ${alert['target_price']:.2f}\n\n"
                        f"💡 <b>Recommendation:</b> SELL / TAKE PROFIT now. 20% swing target achieved!\n"
                        f"Log exit when filled: <code>/sold {t} {alert['current_price']:.2f}</code>"
                    )
                    await self.broadcast(msg, chat_id=reply_chat_id)
                elif alert["type"] == "TRAIL_STOP_ARMED":
                    msg = (
                        f"🛡️ <b>PROFIT PROTECTION LOCKED: {t}</b>\n\n"
                        f"<b>{t}</b> is up <b>+{alert['gain_pct']:.1f}%</b> at ${alert['current_price']:.2f}!\n"
                        f"Trailing stop has been automatically raised to <b>${alert['new_stop']:.2f}</b> to guarantee a winning trade."
                    )
                    await self.broadcast(msg, chat_id=reply_chat_id)
                elif alert["type"] == "STOP_LOSS_HIT":
                    msg = (
                        f"⚠️ <b>STOP LOSS ALERT: {t}</b> ⚠️\n\n"
                        f"<b>{t}</b> dropped to ${alert['current_price']:.2f}, breaching stop loss at ${alert['stop_price']:.2f} ({alert['loss_pct']:.1f}%).\n"
                        f"💡 <b>Recommendation:</b> SELL to cut loss and preserve capital.\n"
                        f"Log exit when filled: <code>/sold {t} {alert['current_price']:.2f}</code>"
                    )
                    await self.broadcast(msg, chat_id=reply_chat_id)

            # STEP 2: Scan for New Long Setups across universe
            signals = await loop.run_in_executor(None, lambda: scan_live(quiet=True))
            self.state = _load_state()
            self.state["subscribers"] = sorted(self.subscribers)
            fresh = _filter_fresh_signals(signals, self.state) if not force else signals
            if force:
                fresh = _filter_fresh_signals(signals, self.state)

            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Take", callback_data="take"),
                    InlineKeyboardButton("⏭ Skip", callback_data="skip"),
                ]
            ])

            # Always stamp last_scan so we can detect a dead bot
            self.state["last_scan"] = now.isoformat()
            self.state["subscribers"] = sorted(self.subscribers)
            nxt = next_scan_eta(now)

            if not signals:
                msg = (
                    f"🔍 <b>Scan done</b> ({now.strftime('%Y-%m-%d %H:%M %Z')})\n"
                    f"Watchlist: {len(tradeable_tickers())} · conf≥{MIN_CONFIDENCE} · LONG only\n"
                    f"Result: <b>no BUY setups</b> — stay in cash.\n"
                    f"Next scan: {nxt.strftime('%Y-%m-%d %H:%M %Z')}\n"
                    f"<i>Bot is alive. Quiet days are normal with a high bar.</i>"
                )
                # Heartbeat on every scheduled/forced scan so silence ≠ crash
                await self.broadcast(msg, chat_id=reply_chat_id)
                _save_state(self.state)
                logger.info("Scan complete: 0 signals (heartbeat sent)")
            else:
                if not fresh:
                    msg = (
                        f"🔍 <b>Scan done</b> ({now.strftime('%Y-%m-%d %H:%M %Z')})\n"
                        f"Found {len(signals)} setup(s) but all still in "
                        f"{SIGNAL_COOLDOWN_HOURS}h re-alert cooldown.\n"
                        f"Next scan: {nxt.strftime('%Y-%m-%d %H:%M %Z')}"
                    )
                    await self.broadcast(msg, chat_id=reply_chat_id)
                for sig in sorted(fresh, key=lambda s: s.confidence, reverse=True):
                    await self.broadcast(format_signal_html(sig), reply_markup=keyboard)
                    await asyncio.sleep(0.4)
                if fresh:
                    self.state = _mark_alerted(fresh, self.state)
                _save_state(self.state)
                logger.info("Scan complete: %d signals, %d fresh alerts", len(signals), len(fresh))

    async def scheduler_loop(self):
        """Fire scans at SCAN_TIMES_ET on session days with missed-slot catch-up & heartbeat."""
        logger.info("Scheduler started. Times ET: %s", [t.strftime("%H:%M") for t in SCAN_TIMES_ET])
        completed_slots_today: Set[str] = set()
        current_date = None
        last_heartbeat = 0

        # Immediate first scan if market open when starting
        if is_trading_session():
            logger.info("Market open at startup — running initial scan")
            try:
                await self.run_scan_and_notify(force=True)
            except Exception as e:
                logger.error("Initial scan failed: %s", e)

        while True:
            try:
                now = _now_et()
                today_str = str(now.date())

                # Reset daily slot tracker at midnight
                if today_str != current_date:
                    current_date = today_str
                    completed_slots_today.clear()

                if is_trading_session(now):
                    for st in SCAN_TIMES_ET:
                        scheduled = now.replace(hour=st.hour, minute=st.minute, second=0, microsecond=0)
                        slot_key = f"{today_str}-{st.strftime('%H%M')}"

                        # If scheduled time has arrived or passed today, and hasn't run yet
                        if now >= scheduled and slot_key not in completed_slots_today:
                            completed_slots_today.add(slot_key)
                            delay_seconds = (now - scheduled).total_seconds()
                            if delay_seconds > 180:
                                logger.info(
                                    "Catching up missed scan slot %s (delayed by %.0f minutes, system likely resumed from sleep)",
                                    slot_key, delay_seconds / 60
                                )
                            else:
                                logger.info("Executing scheduled scan slot %s", slot_key)
                            await self.run_scan_and_notify(force=False)
                            break

                # Periodic heartbeat logging every 5 minutes
                if time_mod.time() - last_heartbeat >= 300:
                    last_heartbeat = time_mod.time()
                    nxt = next_scan_eta(now)
                    logger.info(
                        "Scheduler heartbeat: Market session=%s | Next scan ETA: %s",
                        is_trading_session(now), nxt.strftime("%H:%M %Z")
                    )

                # Once-daily EOD scorecard after cash close
                await self.maybe_send_eod_recap(now)
            except Exception as e:
                logger.error("Scheduler error: %s", e)
            await asyncio.sleep(15)

    async def run(self):
        set_windows_keep_awake(True)
        sched = None
        try:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling(drop_pending_updates=True)
            # Notify admins that live mode is up
            nxt = next_scan_eta()
            boot = (
                f"✅ <b>Live agent online</b> (v3.5 Gen AI Multi-Agent)\n"
                f"Watchlist: {len(tradeable_tickers())} names (full universe)\n"
                f"Scans: {', '.join(t.strftime('%H:%M') for t in SCAN_TIMES_ET)} ET\n"
                f"Session: Mon–Fri, 10:00–15:30 ET\n"
                f"Next scan: {nxt.strftime('%Y-%m-%d %H:%M %Z')}\n"
                f"Windows keep-awake active during market session."
            )
            for uid in self.admin_ids or self.subscribers:
                try:
                    await self.app.bot.send_message(uid, boot, parse_mode="HTML")
                except Exception as e:
                    logger.warning("Boot notify %s failed: %s", uid, e)
            sched = asyncio.create_task(self.scheduler_loop())
            logger.info("Telegram live mode running. Ctrl+C to stop.")
            await asyncio.Event().wait()
        finally:
            set_windows_keep_awake(False)
            if sched:
                sched.cancel()
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()


def run_live() -> None:
    if not HAS_TELEGRAM:
        print("Install: pip install python-telegram-bot python-dotenv")
        raise SystemExit(1)
    if not TELEGRAM_TOKEN:
        print("Missing TELEGRAM_BOT_TOKEN in .env")
        raise SystemExit(1)
    if not ADMIN_IDS:
        print("Warning: ADMIN_IDS empty — anyone who /start can use the bot. Set ADMIN_IDS in .env.")
    print("Starting Telegram live mode…")
    print(f"  Watchlist size: {len(tradeable_tickers())} ({'approved-only' if USE_APPROVED_ONLY else 'full universe'})")
    print(f"  Min confidence: {MIN_CONFIDENCE}")
    print(f"  Scan times ET: {[t.strftime('%H:%M') for t in SCAN_TIMES_ET]}")
    print(f"  Session: {MARKET_OPEN.strftime('%H:%M')}–{MARKET_CLOSE.strftime('%H:%M')} ET Mon–Fri")
    print(f"  Next scan: {next_scan_eta().strftime('%Y-%m-%d %H:%M %Z')}")
    svc = TelegramSignalService(TELEGRAM_TOKEN, ADMIN_IDS)
    asyncio.run(svc.run())


def print_schedule() -> None:
    now = _now_et()
    nxt = next_scan_eta(now)
    print("TELEGRAM / LIVE SCHEDULE (US Eastern)")
    print(f"  Now:              {now.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"  Session window:   {MARKET_OPEN.strftime('%H:%M')}–{MARKET_CLOSE.strftime('%H:%M')} ET, Mon–Fri")
    print(f"  Scan clock times: {', '.join(t.strftime('%H:%M') for t in SCAN_TIMES_ET)} ET")
    print(f"  Next scan:        {nxt.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"  Starts when:      you run  python stock_agent_v3.py live")
    print(f"  First alert:      at next scan slot (or immediately if market is open at startup)")
    print(f"  How often:        up to {len(SCAN_TIMES_ET)} scans per trading day")
    print(f"  Re-alert:         same ticker at most once per {SIGNAL_COOLDOWN_HOURS}h")
    print(f"  Min confidence:   {MIN_CONFIDENCE}")
    print(f"  Fundamentals:     {'ON' if (USE_FACTORS and HAS_FACTORS) else 'OFF'} "
          f"(min factor {MIN_FACTOR_SCORE:.0f}, max {MAX_SIGNALS_PER_SCAN} alerts/scan)")
    print(f"  Watchlist:        {', '.join(tradeable_tickers())}")
    print("  Setup:")
    print("    1. .env has TELEGRAM_BOT_TOKEN and ADMIN_IDS=<your telegram user id>")
    print("    2. Open Telegram, find your bot, send /start")
    print("    3. Keep this PC/terminal running: python stock_agent_v3.py live")
    print("  IMPORTANT: if live is not running, you get ZERO messages (bot is offline).")


# =============================================================================
# MULTI-AGENT COMMITTEE STORAGE & MEMORY (SQLite)
# =============================================================================

COMMITTEE_DB_FILE = Path(__file__).resolve().parent / "trading_committee.db"

def init_committee_db(db_path: Path = COMMITTEE_DB_FILE) -> None:
    """Initialize the SQLite database for Committee deliberations, positions, and journal."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS committee_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        ticker TEXT NOT NULL,
        regime TEXT NOT NULL,
        current_price REAL NOT NULL,
        verdict TEXT NOT NULL,
        shares INTEGER NOT NULL,
        entry_price REAL,
        stop_loss REAL,
        take_profit_1 REAL,
        take_profit_2 REAL,
        technical_score INTEGER,
        technical_stance TEXT,
        sentiment_score INTEGER,
        sentiment_stance TEXT,
        risk_approved INTEGER,
        executive_summary TEXT,
        deliberation_json TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS committee_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        entry_time TEXT NOT NULL,
        entry_price REAL NOT NULL,
        shares INTEGER NOT NULL,
        stop_loss REAL NOT NULL,
        take_profit_1 REAL NOT NULL,
        take_profit_2 REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'OPEN',
        exit_time TEXT,
        exit_price REAL,
        pnl_dollars REAL,
        pnl_pct REAL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS committee_journal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        position_id INTEGER,
        ticker TEXT NOT NULL,
        outcome TEXT NOT NULL,
        pnl_pct REAL NOT NULL,
        reflection_notes TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (position_id) REFERENCES committee_positions (id)
    );
    """)
    conn.commit()
    conn.close()

def log_committee_signal(
    ticker: str,
    regime: str,
    current_price: float,
    verdict: str,
    shares: int,
    entry_price: Optional[float],
    stop_loss: Optional[float],
    tp1: Optional[float],
    tp2: Optional[float],
    tech_score: int,
    tech_stance: str,
    sent_score: int,
    sent_stance: str,
    risk_approved: bool,
    summary: str,
    deliberation: Dict[str, Any],
    db_path: Path = COMMITTEE_DB_FILE,
) -> int:
    """Save full multi-agent deliberations and verdict into committee_signals table."""
    import sqlite3
    init_committee_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    now_str = datetime.now().isoformat()
    cur.execute("""
    INSERT INTO committee_signals (
        timestamp, ticker, regime, current_price, verdict, shares,
        entry_price, stop_loss, take_profit_1, take_profit_2,
        technical_score, technical_stance, sentiment_score, sentiment_stance,
        risk_approved, executive_summary, deliberation_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str, ticker, regime, current_price, verdict, shares,
        entry_price, stop_loss, tp1, tp2,
        tech_score, tech_stance, sent_score, sent_stance,
        1 if risk_approved else 0, summary, json.dumps(deliberation)
    ))
    sig_id = cur.lastrowid
    conn.commit()
    conn.close()
    return sig_id

def get_recent_committee_signals(limit: int = 10, db_path: Path = COMMITTEE_DB_FILE) -> List[Dict[str, Any]]:
    """Retrieve recent multi-agent committee decisions."""
    import sqlite3
    init_committee_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM committee_signals ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def open_committee_position(
    ticker: str,
    entry_price: float,
    shares: int = 10,
    stop_loss: Optional[float] = None,
    tp1: Optional[float] = None,
    tp2: Optional[float] = None,
    db_path: Path = COMMITTEE_DB_FILE,
) -> int:
    """Register an active stock position owned by user."""
    import sqlite3
    init_committee_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(committee_positions)")
    cols = [c[1] for c in cur.fetchall()]
    if "trailing_stop_active" not in cols:
        try:
            cur.execute("ALTER TABLE committee_positions ADD COLUMN trailing_stop_active INTEGER DEFAULT 0")
        except Exception:
            pass

    now_str = datetime.now().isoformat()
    sl = stop_loss if stop_loss is not None else round(entry_price * (1 - MAX_STOP_PCT), 2)
    t1 = tp1 if tp1 is not None else round(entry_price * (1 + SWING_TRAIL_START_PCT), 2)
    t2 = tp2 if tp2 is not None else round(entry_price * (1 + SWING_TP_PCT), 2)

    cur.execute("""
    INSERT INTO committee_positions (
        ticker, entry_time, entry_price, shares, stop_loss,
        take_profit_1, take_profit_2, status, trailing_stop_active
    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', 0)
    """, (ticker.upper(), now_str, float(entry_price), int(shares), float(sl), float(t1), float(t2)))
    pos_id = cur.lastrowid
    conn.commit()
    conn.close()
    return pos_id


def get_open_positions(db_path: Path = COMMITTEE_DB_FILE) -> List[Dict[str, Any]]:
    """Retrieve all open positions currently tracked."""
    import sqlite3
    init_committee_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM committee_positions WHERE status = 'OPEN' ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def close_committee_position(
    position_id: int,
    exit_price: float,
    exit_reason: str = "MANUAL_CLOSE",
    db_path: Path = COMMITTEE_DB_FILE,
) -> Optional[Dict[str, Any]]:
    """Close an open position, calculate realized PnL, and log journal reflection."""
    import sqlite3
    init_committee_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM committee_positions WHERE id = ?", (position_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    pos = dict(row)
    entry_px = float(pos["entry_price"])
    shares = int(pos["shares"])
    pnl_pct = safe_div(exit_price - entry_px, entry_px, 0.0) * 100
    pnl_dlr = (exit_price - entry_px) * shares
    now_str = datetime.now().isoformat()

    cur.execute("""
    UPDATE committee_positions
    SET status = 'CLOSED', exit_time = ?, exit_price = ?, pnl_dollars = ?, pnl_pct = ?
    WHERE id = ?
    """, (now_str, float(exit_price), round(pnl_dlr, 2), round(pnl_pct, 2), position_id))

    outcome = "WIN" if pnl_pct > 0 else "LOSS"
    cur.execute("""
    INSERT INTO committee_journal (
        position_id, ticker, outcome, pnl_pct, reflection_notes, created_at
    ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        position_id, pos["ticker"], outcome, round(pnl_pct, 2),
        f"Exit: {exit_reason} at ${exit_price:.2f}. PnL: {pnl_pct:+.1f}% (${pnl_dlr:+,.2f})",
        now_str
    ))
    conn.commit()
    conn.close()
    pos.update({
        "exit_time": now_str,
        "exit_price": exit_price,
        "pnl_dollars": pnl_dlr,
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
    })
    return pos


def update_position_trailing_stop(position_id: int, new_stop: float, db_path: Path = COMMITTEE_DB_FILE):
    """Raise stop loss to lock in profits for a winning position."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    UPDATE committee_positions
    SET stop_loss = ?, trailing_stop_active = 1
    WHERE id = ?
    """, (float(new_stop), position_id))
    conn.commit()
    conn.close()


def check_open_positions_exits() -> List[Dict[str, Any]]:
    """
    Evaluate all currently owned positions for exits:
    1) Target hit (+20% or custom TP2) -> SELL / TAKE PROFIT alert
    2) Trailing stop lock (+10% reached) -> Auto-raise stop to +5%
    3) Stop loss hit -> CUT LOSS alert
    Strict rule: NEVER generates sell alerts for tickers not in committee_positions!
    """
    open_positions = get_open_positions()
    if not open_positions:
        return []

    alerts = []
    for pos in open_positions:
        ticker = pos["ticker"]
        entry_px = float(pos["entry_price"])
        stop_px = float(pos["stop_loss"])
        tp1_px = float(pos["take_profit_1"])
        tp2_px = float(pos["take_profit_2"])
        trail_active = bool(pos.get("trailing_stop_active", 0))

        cur_px = 0.0
        try:
            df = fetch_data(ticker, period="5d", interval="1h")
            if df is not None and not df.empty:
                cur_px = float(df.iloc[-1]["close"])
        except Exception as e:
            logger.debug(f"Price check error for {ticker}: {e}")

        if cur_px <= 0:
            continue

        gain_pct = (cur_px - entry_px) / entry_px * 100

        # 1. Check Take Profit (+20% target reached)
        if cur_px >= tp2_px:
            alerts.append({
                "type": "TAKE_PROFIT_20PCT",
                "position": pos,
                "current_price": cur_px,
                "gain_pct": gain_pct,
                "target_price": tp2_px,
            })
        # 2. Check Trailing Stop Activation (+10% gain locks in +5%)
        elif cur_px >= tp1_px and not trail_active:
            new_stop = round(entry_px * (1 + SWING_TRAIL_LOCK_PCT), 2)
            update_position_trailing_stop(pos["id"], new_stop)
            alerts.append({
                "type": "TRAIL_STOP_ARMED",
                "position": pos,
                "current_price": cur_px,
                "gain_pct": gain_pct,
                "new_stop": new_stop,
            })
        # 3. Check Stop Loss Hit
        elif cur_px <= stop_px:
            alerts.append({
                "type": "STOP_LOSS_HIT",
                "position": pos,
                "current_price": cur_px,
                "loss_pct": gain_pct,
                "stop_price": stop_px,
            })

    return alerts


# =============================================================================
# MULTI-AGENT COMMITTEE LOGIC (Option 2)
# =============================================================================

@dataclass
class CommitteeReport:
    ticker: str
    current_price: float
    regime: str
    verdict: str                  # "EXECUTE_BUY" | "HOLD_CASH" | "REJECT_SETUP"
    confidence_pct: int
    shares: int
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit_1: Optional[float]
    take_profit_2: Optional[float]
    risk_reward: float
    tech_stance: str
    tech_score: int
    sent_stance: str
    sent_score: int
    risk_approved: bool
    risk_veto_reason: Optional[str]
    executive_summary: str
    deliberation: Dict[str, Any]

class TechnicalsAgent:
    """Agent 1: Quantitative chartist evaluating price action, EMAs, divergences, and indicators."""
    @staticmethod
    def analyze(ticker: str, df: pd.DataFrame, spy_regime: str) -> Dict[str, Any]:
        if df.empty:
            return {
                "stance": "NEUTRAL",
                "score": 0,
                "confidence": 0,
                "entry": 0.0,
                "stop_loss": None,
                "tp1": None,
                "tp2": None,
                "risk_reward": 0.0,
                "reasons": ["No price data available."],
            }
        px = float(df.iloc[-1]["close"])
        ts = pd.Timestamp(df.iloc[-1]["timestamp"]).to_pydatetime()
        sig = generate_signal(ticker, df, px, ts, regime=spy_regime)
        if not sig:
            return {
                "stance": "NEUTRAL",
                "score": 40,
                "confidence": 0,
                "entry": px,
                "stop_loss": None,
                "tp1": None,
                "tp2": None,
                "risk_reward": 0.0,
                "reasons": ["No actionable technical breakout pattern detected."],
            }
        
        score = int(sig.confidence)
        is_bull = sig.direction in ("BUY", "LONG")
        stance = "BULLISH" if is_bull and score >= 75 else ("LEAN_BULLISH" if is_bull else "NEUTRAL")
        return {
            "stance": stance,
            "score": score,
            "confidence": sig.confidence,
            "direction": sig.direction,
            "entry": getattr(sig, "entry", getattr(sig, "entry_price", px)),
            "stop_loss": sig.stop_loss,
            "tp1": getattr(sig, "target_1", getattr(sig, "take_profit_1", None)),
            "tp2": getattr(sig, "target_2", getattr(sig, "take_profit_2", None)),
            "risk_reward": getattr(sig, "r_r", getattr(sig, "risk_reward", 0.0)),
            "reasons": getattr(sig, "rationale", getattr(sig, "factors", [])),
        }

class SentimentAgent:
    """Agent 2: Market intelligence evaluating Reddit, news headlines, and social mood."""
    @staticmethod
    def analyze(ticker: str) -> Dict[str, Any]:
        headlines: List[str] = []
        reddit_mentions: List[str] = []
        
        # Pull company news
        try:
            news_items = fetch_company_news(ticker)
            for item in news_items[:5]:
                t = ""
                if isinstance(item, dict):
                    t = item.get("title") or ""
                    if not t and "content" in item and isinstance(item["content"], dict):
                        t = item["content"].get("title", "")
                else:
                    t = str(item)
                if t:
                    headlines.append(t)
        except Exception as e:
            logger.debug(f"Sentiment news fetch note: {e}")

        # Pull Reddit chatter
        try:
            r_data = fetch_reddit_posts(limit=6)
            for sub, titles in r_data.items():
                for title in titles:
                    if ticker.upper() in title.upper():
                        reddit_mentions.append(f"r/{sub}: {title}")
        except Exception as e:
            logger.debug(f"Sentiment reddit fetch note: {e}")

        score = 0
        notes = []
        bull_words = ["surge", "record", "growth", "buy", "upgrade", "call", "beat", "rally", "strong", "outperform"]
        bear_words = ["drop", "fall", "downgrade", "investigation", "probe", "loss", "warning", "curb", "cut", "risk"]
        
        all_text = " ".join(headlines + reddit_mentions).lower()
        for bw in bull_words:
            if bw in all_text:
                score += 15
        for bw in bear_words:
            if bw in all_text:
                score -= 20

        score = max(-100, min(100, score))
        if score >= 25:
            stance = "SUPPORTIVE"
        elif score <= -25:
            stance = "HIGH_RISK_AVOID"
        else:
            stance = "NEUTRAL"

        return {
            "stance": stance,
            "score": score,
            "headlines_count": len(headlines),
            "reddit_mentions_count": len(reddit_mentions),
            "headlines_sample": headlines[:3],
            "reddit_sample": reddit_mentions[:2],
        }

class RiskAgent:
    """Agent 3: Chief Risk Officer enforcing position sizing and capital preservation."""
    @staticmethod
    def evaluate(
        account_balance: float,
        current_price: float,
        stop_loss: Optional[float],
        risk_per_trade_pct: float = 0.015,  # 1.5% portfolio risk
        max_position_pct: float = 0.20,      # Max 20% capital in single stock
    ) -> Dict[str, Any]:
        if not stop_loss or stop_loss >= current_price or current_price <= 0:
            return {
                "approved": False,
                "shares": 0,
                "reason": "Invalid or missing stop loss level relative to current price.",
            }

        risk_per_share = current_price - stop_loss
        stop_pct = risk_per_share / current_price
        
        # Veto if stop is unrealistically wide or tight
        if stop_pct > MAX_STOP_PCT:
            return {
                "approved": False,
                "shares": 0,
                "reason": f"Stop distance {stop_pct*100:.1f}% exceeds max allowable risk ceiling ({MAX_STOP_PCT*100:.1f}%).",
            }
        if stop_pct < MIN_STOP_PCT:
            return {
                "approved": False,
                "shares": 0,
                "reason": f"Stop distance {stop_pct*100:.1f}% is too tight (< {MIN_STOP_PCT*100:.1f}%). High risk of noise shakeout.",
            }

        dollar_risk_target = account_balance * risk_per_trade_pct
        shares_by_risk = int(dollar_risk_target / risk_per_share)
        max_dollars = account_balance * max_position_pct
        shares_by_cap = int(max_dollars / current_price)

        shares = max(0, min(shares_by_risk, shares_by_cap))
        if shares == 0:
            return {
                "approved": False,
                "shares": 0,
                "reason": "Calculated position size is 0 shares given account risk limits.",
            }

        return {
            "approved": True,
            "shares": shares,
            "dollar_risk": round(shares * risk_per_share, 2),
            "capital_required": round(shares * current_price, 2),
            "portfolio_risk_pct": round((shares * risk_per_share / account_balance) * 100, 2),
            "stop_pct": round(stop_pct * 100, 2),
        }

class ExecutiveCIOAgent:
    """Agent 4: Chief Investment Officer synthesizing reports and issuing final verdict."""
    @staticmethod
    def deliberate(
        ticker: str,
        regime: str,
        tech_data: Dict[str, Any],
        sent_data: Dict[str, Any],
        risk_data: Dict[str, Any],
    ) -> CommitteeReport:
        current_price = tech_data.get("entry", 0.0)
        tech_stance = tech_data.get("stance", "NEUTRAL")
        tech_score = tech_data.get("score", 0)
        sent_stance = sent_data.get("stance", "NEUTRAL")
        sent_score = sent_data.get("score", 0)
        risk_approved = risk_data.get("approved", False)
        
        # Check if user already holds this ticker in portfolio
        open_holdings = [p for p in get_open_positions() if p["ticker"] == ticker.upper()]
        holding_note = ""
        if open_holdings:
            hp = open_holdings[0]
            unrealized = safe_div(current_price - hp["entry_price"], hp["entry_price"], 0.0) * 100
            holding_note = f" [OWNED: Entered at ${hp['entry_price']:.2f}, PnL {unrealized:+.1f}%. Target: ${hp['take_profit_2']:.2f}]"

        is_etf = STOCKS.get(ticker, {}).get("sector") == "ETF" or ticker in (
            "SPY", "VOO", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "SMH"
        )

        # Rule 1: Regime check (SPY Macro condition)
        if BULL_REGIME_ONLY and regime != "BULL":
            verdict = "HOLD_CASH"
            summary = f"Macro regime is {regime}. CIO policy mandates preserving 100% cash until SPY reclaims bull trend.{holding_note}"
            shares = 0
        # Rule 2: Technical confirmation gate (No setup = hold cash)
        elif tech_stance == "NEUTRAL" or (not is_etf and tech_score < 70) or (is_etf and tech_score < 35) or tech_data.get("stop_loss") is None:
            verdict = "HOLD_CASH"
            if is_etf:
                summary = f"Core ETF {ticker} is currently extended / not in a dip zone (Score {tech_score}/100). Continue holding core ETF shares; wait for 2–4% pullback to add tranches.{holding_note}"
            else:
                summary = f"Technical confidence ({tech_score}/100) below execution threshold (70). No actionable swing dip setup detected — stay in cash.{holding_note}"
            shares = 0
        # Rule 3: Risk Officer vetoes candidate setup
        elif not risk_approved:
            verdict = "REJECT_SETUP"
            veto_reason = risk_data.get("reason", "Violated portfolio risk policy")
            summary = f"Risk Officer VETO: {veto_reason}.{holding_note}"
            shares = 0
        # Rule 4: Sentiment conflict check (toxic news / severe negative catalyst)
        elif sent_stance == "HIGH_RISK_AVOID":
            verdict = "REJECT_SETUP"
            summary = f"Sentiment Intelligence flags high risk negative catalysts/probe (Score {sent_score}). Technical setup bypassed.{holding_note}"
            shares = 0
        # Rule 5: Core ETF Accumulation
        elif is_etf and tech_score >= 35:
            verdict = "CORE_ACCUMULATE"
            shares = risk_data.get("shares", 0)
            summary = (
                f"Core ETF Accumulation: {ticker} is in a prime long-term compounding zone. "
                f"Hold through minor fluctuations; no short-term sell pressure.{holding_note}"
            )
        # Rule 6: Swing Buy Execution (~20% target)
        elif tech_stance in ("BULLISH", "LEAN_BULLISH") and tech_score >= 70:
            verdict = "EXECUTE_BUY"
            shares = risk_data.get("shares", 0)
            summary = (
                f"Committee Consensus BUY on {ticker}. Tech Score {tech_score} targeting ~20% swing gain "
                f"(${tech_data.get('tp2', 0):.2f}) with {risk_data.get('portfolio_risk_pct', 0)}% risk "
                f"({shares} shares). Sentiment is {sent_stance}.{holding_note}"
            )
        else:
            verdict = "HOLD_CASH"
            summary = f"Technical confidence ({tech_score}/100) below committee execution threshold (70). Watching for clean dip setups.{holding_note}"
            shares = 0

        deliberation_log = {
            "technicals": tech_data,
            "sentiment": sent_data,
            "risk": risk_data,
        }

        report = CommitteeReport(
            ticker=ticker,
            current_price=current_price,
            regime=regime,
            verdict=verdict,
            confidence_pct=tech_score,
            shares=shares,
            entry_price=tech_data.get("entry"),
            stop_loss=tech_data.get("stop_loss"),
            take_profit_1=tech_data.get("tp1"),
            take_profit_2=tech_data.get("tp2"),
            risk_reward=tech_data.get("risk_reward", 0.0),
            tech_stance=tech_stance,
            tech_score=tech_score,
            sent_stance=sent_stance,
            sent_score=sent_score,
            risk_approved=risk_approved,
            risk_veto_reason=risk_data.get("reason") if not risk_approved else None,
            executive_summary=summary,
            deliberation=deliberation_log,
        )

        # Log into SQLite memory
        try:
            log_committee_signal(
                ticker=ticker,
                regime=regime,
                current_price=current_price,
                verdict=verdict,
                shares=shares,
                entry_price=report.entry_price,
                stop_loss=report.stop_loss,
                tp1=report.take_profit_1,
                tp2=report.take_profit_2,
                tech_score=tech_score,
                tech_stance=tech_stance,
                sent_score=sent_score,
                sent_stance=sent_stance,
                risk_approved=risk_approved,
                summary=summary,
                deliberation=deliberation_log,
            )
        except Exception as e:
            logger.error(f"Error logging to committee db: {e}")

        return report

def run_committee_analysis(ticker: str, account_balance: float = ACCOUNT_BALANCE, cached_spy: Optional[Tuple[pd.DataFrame, str]] = None) -> CommitteeReport:
    """Full committee meeting on a given ticker."""
    if cached_spy:
        spy_df, spy_regime = cached_spy
    else:
        spy_df = fetch_data(REGIME_SYMBOL, period="730d", interval="1h")
        now = _now_et()
        if spy_df is not None and not spy_df.empty:
            ts_reg = pd.Timestamp(spy_df.iloc[-1]["timestamp"]).to_pydatetime()
        else:
            ts_reg = now
        spy_regime = _regime_from_spy(spy_df, ts_reg)
    
    meta = STOCKS.get(ticker, {"yf": ticker})
    df = fetch_data(meta["yf"], period="730d", interval="1h")
    if df.empty:
        raise ValueError(f"Could not load market data for {ticker}")

    # 1. Technicals Agent
    tech_data = TechnicalsAgent.analyze(ticker, df, spy_regime=spy_regime)
    
    # 2. Sentiment Agent
    sent_data = SentimentAgent.analyze(ticker)
    
    # 3. Risk Agent
    curr_price = float(df["close"].iloc[-1])
    risk_data = RiskAgent.evaluate(
        account_balance=account_balance,
        current_price=curr_price,
        stop_loss=tech_data.get("stop_loss"),
    )
    
    # 4. Executive CIO Agent
    return ExecutiveCIOAgent.deliberate(
        ticker=ticker,
        regime=spy_regime,
        tech_data=tech_data,
        sent_data=sent_data,
        risk_data=risk_data,
    )

def scan_committee_universe(tickers: Optional[List[str]] = None, only_approved_buys: bool = True) -> List[CommitteeReport]:
    """
    Run committee screening across the universe.
    Efficiently screens all symbols and convenes the full committee
    to return only the setups deemed worth investing (verdict = EXECUTE_BUY).
    """
    if not tickers:
        tickers = tradeable_tickers()

    spy_df = fetch_data(REGIME_SYMBOL, period="730d", interval="1h")
    now = _now_et()
    if spy_df is not None and not spy_df.empty:
        ts_reg = pd.Timestamp(spy_df.iloc[-1]["timestamp"]).to_pydatetime()
    else:
        ts_reg = now
    spy_regime = _regime_from_spy(spy_df, ts_reg)
    cached_spy = (spy_df, spy_regime)

    print(f"\n[Committee Screener] Macro Regime: {spy_regime.upper()} | Scanning {len(tickers)} symbols...")
    approved_reports: List[CommitteeReport] = []

    for t in tickers:
        try:
            meta = STOCKS.get(t, {"yf": t})
            df = fetch_data(meta["yf"], period="730d", interval="1h")
            if df.empty or len(df) < 220:
                continue

            px = float(df.iloc[-1]["close"])
            ts = pd.Timestamp(df.iloc[-1]["timestamp"]).to_pydatetime()
            
            # Pre-filter: Check if baseline technical criteria has setup potential
            pre_sig = generate_signal(t, df, px, ts, regime=spy_regime)
            
            # If screening for only high conviction buys, skip tickers with zero technical setup
            if only_approved_buys and pre_sig is None:
                continue

            # Convene full multi-agent committee debate
            rep = run_committee_analysis(t, account_balance=ACCOUNT_BALANCE, cached_spy=cached_spy)
            if only_approved_buys:
                if rep.verdict == "EXECUTE_BUY":
                    approved_reports.append(rep)
            else:
                approved_reports.append(rep)
        except Exception as e:
            logger.debug("Error screening %s: %s", t, e)

    return approved_reports

def print_committee_report(rep: CommitteeReport) -> None:
    """Format and print a professional Investment Memorandum."""
    verdict_tag = "[BUY]" if rep.verdict == "EXECUTE_BUY" else ("[HOLD]" if rep.verdict == "HOLD_CASH" else "[REJECT]")
    print("\n" + "=" * 68)
    print(f" INVESTMENT COMMITTEE MEMORANDUM: {rep.ticker} | Market Regime: {rep.regime.upper()}")
    print("=" * 68)
    print(f"VERDICT:            {verdict_tag} {rep.verdict}")
    print(f"EXECUTIVE SUMMARY:  {rep.executive_summary}")
    print("-" * 68)
    print("COMMITTEE DEBATE BREAKDOWN:")
    print(f" 1. Technical Agent: Stance={rep.tech_stance} | Confidence={rep.tech_score}/100")
    if rep.entry_price and rep.stop_loss and rep.take_profit_1:
        print(f"    Levels: Entry=${rep.entry_price:.2f} | Stop=${rep.stop_loss:.2f} | TP1=${rep.take_profit_1:.2f} | R:R={rep.risk_reward:.1f}")
    elif rep.entry_price:
        print(f"    Reference Price: ${rep.entry_price:.2f} (No active breakout setup)")
    print(f" 2. Sentiment Agent: Stance={rep.sent_stance} | Sentiment Score={rep.sent_score} (-100 to +100)")
    sample_news = rep.deliberation.get("sentiment", {}).get("headlines_sample", [])
    if sample_news:
        print(f"    Recent Catalyst: \"{sample_news[0][:65]}\"")
    print(f" 3. Chief Risk Officer: {'[APPROVED]' if rep.risk_approved else '[VETOED]'}")
    if rep.risk_approved:
        risk_info = rep.deliberation.get("risk", {})
        print(f"    Sizing: {rep.shares} shares (${risk_info.get('capital_required', 0):,.2f}) | Max Risk: ${risk_info.get('dollar_risk', 0):,.2f} ({risk_info.get('portfolio_risk_pct', 0)}% of portfolio)")
    else:
        print(f"    Reason: {rep.risk_veto_reason}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Signal Agent v3.2 LONG ONLY + Telegram")
    parser.add_argument(
        "mode",
        choices=["backtest", "scan", "signal", "live", "schedule", "factors", "committee", "debate"],
        nargs="?",
        default="backtest",
    )
    parser.add_argument("ticker", nargs="?", default=None, help="Ticker for signal mode")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--list", action="store_true", help="Print watchlist and exit")
    parser.add_argument(
        "--all-tickers",
        action="store_true",
        help="Use full universe (default already full unless USE_APPROVED_ONLY=1)",
    )
    parser.add_argument(
        "--approved",
        action="store_true",
        help="Restrict to core approved list only",
    )
    args = parser.parse_args()

    # Default follows USE_APPROVED_ONLY (full universe). Explicit flags override.
    if args.approved:
        use_approved = True
    elif args.all_tickers:
        use_approved = False
    else:
        use_approved = USE_APPROVED_ONLY

    if args.list:
        names = tradeable_tickers(use_approved=use_approved)
        label = "APPROVED LONG" if use_approved else "FULL LONG"
        print(f"{label} universe ({len(names)} symbols):")
        print(", ".join(names))
        print(f"Regime reference: {REGIME_SYMBOL}")
        print(f"Min confidence: {MIN_CONFIDENCE}")
        print_schedule()
        raise SystemExit(0)

    if args.mode == "backtest":
        run_backtest(tickers=args.tickers, start=args.start, end=args.end, use_approved=use_approved)
    elif args.mode == "scan":
        scan_live(tickers=args.tickers, use_approved=use_approved)
    elif args.mode == "live":
        run_live()
    elif args.mode == "schedule":
        print_schedule()
    elif args.mode in ("committee", "debate"):
        t = (args.ticker or (args.tickers[0] if args.tickers else "ALL")).upper()
        if t in ("ALL", "SCAN"):
            print("Convening Investment Committee across all 86 symbols...")
            approved = scan_committee_universe(only_approved_buys=True)
            if not approved:
                print("\n[COMMITTEE UNIVERSE VERDICT]: 100% CASH")
                print("All symbols evaluated. 0 stocks meet the strict investment criteria in the current market regime.")
                print("Capital preserved.\n")
            else:
                print(f"\nFound {len(approved)} APPROVED BUY setups:")
                for rep in approved:
                    print_committee_report(rep)
        else:
            print(f"Convening Investment Committee for {t}...")
            report = run_committee_analysis(t, account_balance=ACCOUNT_BALANCE)
            print_committee_report(report)
    elif args.mode == "factors":
        if not HAS_FACTORS:
            print("factor_engine.py not found next to this script.")
            raise SystemExit(1)
        names = args.tickers or tradeable_tickers(use_approved=use_approved)
        names = [n.upper() for n in names]
        print(f"Scoring fundamentals for {len(names)} ticker(s)...")
        scores = _factor_map(names, quiet=False)
        table = factor_engine.rank_table(scores, top=len(names))
        print()
        print(table.to_string(index=False) if not table.empty else "No factor data.")
    else:
        t = args.ticker or (args.tickers[0] if args.tickers else "AMZN")
        scan_live(tickers=[t.upper()], use_approved=False)
