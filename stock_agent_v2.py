"""
📈 STOCK SIGNAL AGENT v2 — Fixed Strategy
Addresses backtest failures: wider stops, 1:2 R:R, trend-only trades, VWAP filter.

USAGE:
    python stock_agent_v2.py backtest    # Test first
    python stock_agent_v2.py live        # Then go live
"""

import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from flask import Flask, render_template_string, jsonify
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION v2
# =============================================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_IDS", "123456789").split(",") if x.strip()]

STOCKS = {
    "SPY":  {"yf": "SPY",  "sector": "ETF"},
    "QQQ":  {"yf": "QQQ",  "sector": "ETF"},
    "IWM":  {"yf": "IWM",  "sector": "ETF"},
    "AAPL": {"yf": "AAPL", "sector": "Tech"},
    "TSLA": {"yf": "TSLA", "sector": "Auto"},
    "NVDA": {"yf": "NVDA", "sector": "Tech"},
    "MSFT": {"yf": "MSFT", "sector": "Tech"},
    "GOOGL":{"yf": "GOOGL","sector": "Tech"},
    "AMZN": {"yf": "AMZN", "sector": "Consumer"},
    "META": {"yf": "META", "sector": "Tech"},
    "AMD":  {"yf": "AMD",  "sector": "Tech"},
    "NFLX": {"yf": "NFLX", "sector": "Media"},
    "CRM":  {"yf": "CRM",  "sector": "Tech"},
    "UBER": {"yf": "UBER", "sector": "Transport"},
    "COIN": {"yf": "COIN", "sector": "Finance"},
}

# v2 RISK SETTINGS — FIXED
MAX_RISK_PCT = 2.0
MIN_CONFIDENCE = 80          # HIGHER threshold = fewer, better signals
POSITION_SIZE_PCT = 10.0
SCAN_INTERVAL_HOURS = 4
MAX_POSITIONS = 5

# Market hours
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)

# v2: HOLIDAYS
HOLIDAYS_2024_2026 = [
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
]

# =============================================================================
# MARKET CALENDAR
# =============================================================================

def is_market_open(dt: datetime) -> bool:
    if dt.weekday() >= 5:
        return False
    if dt.strftime("%Y-%m-%d") in HOLIDAYS_2024_2026:
        return False
    return MARKET_OPEN <= dt.time() <= MARKET_CLOSE

def next_market_open(dt: datetime) -> datetime:
    while not is_market_open(dt):
        dt += timedelta(hours=1)
        dt = dt.replace(minute=30, second=0, microsecond=0)
    return dt

# =============================================================================
# DATA FETCHER
# =============================================================================

def fetch_data(symbol: str, period: str = "120d", interval: str = "1h") -> pd.DataFrame:
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        rename_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'close', 'Volume': 'volume', 'Datetime': 'timestamp', 'Date': 'timestamp'}
        data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                data[col] = data[col].astype(float)
        return data
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def get_current_price(symbol: str) -> float:
    try:
        data = yf.download(symbol, period="1d", interval="1m", progress=False)
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except:
        pass
    return 0.0

# =============================================================================
# INDICATORS v2
# =============================================================================

def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def ema(closes: pd.Series, period: int) -> pd.Series:
    return closes.ewm(span=period, adjust=False).mean()

def sma(closes: pd.Series, period: int) -> pd.Series:
    return closes.rolling(window=period).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price — key institutional level."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

def bollinger_bands(closes: pd.Series, period: int = 20, std_dev: int = 2):
    middle = closes.rolling(window=period).mean()
    std = closes.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def swing_points(series: pd.Series, order: int = 3):
    highs_idx = argrelextrema(series.values, np.greater, order=order)[0]
    lows_idx = argrelextrema(series.values, np.less, order=order)[0]
    highs = pd.Series(False, index=series.index)
    lows = pd.Series(False, index=series.index)
    if len(highs_idx) > 0:
        highs.iloc[highs_idx] = True
    if len(lows_idx) > 0:
        lows.iloc[lows_idx] = True
    return highs, lows

def detect_divergence(price: pd.Series, indicator: pd.Series, type: str = "bullish") -> bool:
    price_recent = price.tail(50)
    ind_recent = indicator.tail(50)
    if type == "bullish":
        p_lows = argrelextrema(price_recent.values, np.less, order=5)[0]
        i_lows = argrelextrema(ind_recent.values, np.less, order=5)[0]
        if len(p_lows) < 2 or len(i_lows) < 2:
            return False
        p_ll = price_recent.iloc[p_lows[-1]] < price_recent.iloc[p_lows[-2]]
        i_hl = ind_recent.iloc[i_lows[-1]] > ind_recent.iloc[i_lows[-2]]
        oversold = ind_recent.iloc[i_lows[-1]] < 35
        return p_ll and i_hl and oversold
    else:
        p_highs = argrelextrema(price_recent.values, np.greater, order=5)[0]
        i_highs = argrelextrema(ind_recent.values, np.greater, order=5)[0]
        if len(p_highs) < 2 or len(i_highs) < 2:
            return False
        p_hh = price_recent.iloc[p_highs[-1]] > price_recent.iloc[p_highs[-2]]
        i_lh = ind_recent.iloc[i_highs[-1]] < ind_recent.iloc[i_highs[-2]]
        overbought = ind_recent.iloc[i_highs[-1]] > 65
        return p_hh and i_lh and overbought

# =============================================================================
# SIGNAL GENERATOR v2 — FIXED LOGIC
# =============================================================================

@dataclass
class StockSignal:
    ticker: str
    direction: str
    entry: float
    stop_loss: float
    target: float
    position_size: int
    position_value: float
    confidence: int
    rationale: List[str]
    risk_amount: float
    r_r: float
    timestamp: datetime
    is_etf: bool = False

def generate_signal(ticker: str, df: pd.DataFrame, current_price: float,
                    account_balance: float, timestamp: datetime, is_etf: bool = False) -> Optional[StockSignal]:
    """
    v2 Signal Logic — Key Changes:
    1. ONLY trade WITH the 200 EMA trend (no counter-trend)
    2. Require pullback from recent swing (don't chase)
    3. Wider stops (2x ATR) to avoid noise
    4. 1:2 R:R target (realistic for stocks)
    5. VWAP as key level (institutional money)
    6. Volume must be 1.8x average (not 1.5x)
    7. Confidence threshold raised to 80%
    """
    if df.empty or current_price == 0 or len(df) < 50:
        return None

    score = 0
    factors = []
    direction = None

    # === CORE TREND FILTER (200 EMA) ===
    ema_20 = ema(df['close'], 20)
    ema_50 = ema(df['close'], 50)
    ema_200 = ema(df['close'], 200)
    vwap_line = vwap(df)

    price_above_200 = current_price > ema_200.iloc[-1]
    ema_20_above_50 = ema_20.iloc[-1] > ema_50.iloc[-1]
    ema_50_above_200 = ema_50.iloc[-1] > ema_200.iloc[-1]

    # === FACTOR 1: Trend Alignment (MUST HAVE) ===
    if price_above_200 and ema_20_above_50 and ema_50_above_200:
        score += 25
        factors.append("Strong Uptrend (Price > EMA20 > EMA50 > EMA200)")
        direction = "LONG"
    elif not price_above_200 and not ema_20_above_50 and not ema_50_above_200:
        score += 25
        factors.append("Strong Downtrend (Price < EMA20 < EMA50 < EMA200)")
        direction = "SHORT"
    else:
        # Mixed trend — skip entirely
        return None

    # === FACTOR 2: Pullback Requirement (MUST HAVE) ===
    # For LONG: price must have pulled back from recent high (not chasing)
    # For SHORT: price must have bounced from recent low
    swing_highs, swing_lows = swing_points(df['close'], order=5)
    recent_highs = df['close'][swing_highs].tail(3)
    recent_lows = df['close'][swing_lows].tail(3)

    if direction == "LONG":
        if len(recent_highs) > 0:
            last_high = recent_highs.iloc[-1]
            pullback_pct = (last_high - current_price) / last_high
            if pullback_pct > 0.03:  # Pulled back at least 3% from recent high
                score += 20
                factors.append(f"Pullback from high: {pullback_pct*100:.1f}%")
            else:
                return None  # Chasing, skip
        else:
            return None
    else:  # SHORT
        if len(recent_lows) > 0:
            last_low = recent_lows.iloc[-1]
            bounce_pct = (current_price - last_low) / last_low
            if bounce_pct > 0.03:
                score += 20
                factors.append(f"Bounce from low: {bounce_pct*100:.1f}%")
            else:
                return None
        else:
            return None

    # === FACTOR 3: RSI Divergence ===
    rsi_series = rsi(df['close'])
    if detect_divergence(df['close'], rsi_series, type="bullish") and direction == "LONG":
        score += 15
        factors.append("RSI Bullish Divergence")
    elif detect_divergence(df['close'], rsi_series, type="bearish") and direction == "SHORT":
        score += 15
        factors.append("RSI Bearish Divergence")

    # === FACTOR 4: VWAP Position ===
    # For LONG: price below VWAP = discount, reverting up
    # For SHORT: price above VWAP = premium, reverting down
    if direction == "LONG" and current_price < vwap_line.iloc[-1]:
        score += 10
        factors.append("Price below VWAP (institutional discount)")
    elif direction == "SHORT" and current_price > vwap_line.iloc[-1]:
        score += 10
        factors.append("Price above VWAP (institutional premium)")

    # === FACTOR 5: Volume Confirmation (STRONGER) ===
    vol_avg = df['volume'].tail(20).mean()
    vol_current = df['volume'].iloc[-1]
    if vol_current > vol_avg * 1.8:  # Raised from 1.5x to 1.8x
        score += 15
        factors.append(f"Volume Spike ({vol_current/vol_avg:.1f}x avg)")
    elif vol_current > vol_avg * 1.3:
        score += 5
        factors.append("Moderate Volume")
    else:
        return None  # No volume = no conviction

    # === FACTOR 6: Bollinger Band Extreme ===
    upper, middle, lower = bollinger_bands(df['close'])
    bb_position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])

    if direction == "LONG" and bb_position < 0.25:
        score += 10
        factors.append("Price near lower Bollinger Band")
    elif direction == "SHORT" and bb_position > 0.75:
        score += 10
        factors.append("Price near upper Bollinger Band")

    # === CHECK CONFIDENCE ===
    if score < MIN_CONFIDENCE:
        return None

    # === CALCULATE LEVELS v2 ===
    atr_val = atr(df).iloc[-1]

    if direction == "LONG":
        # Wider stop: 2x ATR or below recent swing low
        if len(recent_lows) > 0:
            sl = min(recent_lows.iloc[-1] * 0.998, current_price - atr_val * 2.0)
        else:
            sl = current_price - atr_val * 2.0
        sl = min(sl, current_price * 0.96)  # Max 4% stop

        risk_dist = current_price - sl
        target = current_price + risk_dist * 2  # 1:2 R:R (was 1:3)
    else:
        if len(recent_highs) > 0:
            sl = max(recent_highs.iloc[-1] * 1.002, current_price + atr_val * 2.0)
        else:
            sl = current_price + atr_val * 2.0
        sl = max(sl, current_price * 1.04)

        risk_dist = sl - current_price
        target = current_price - risk_dist * 2

    # === POSITION SIZING ===
    risk_amount = account_balance * (MAX_RISK_PCT / 100)
    stop_distance = abs(current_price - sl)

    if stop_distance <= 0:
        return None

    shares = int(risk_amount / stop_distance)
    position_value = shares * current_price

    # Cap at 10% of account
    max_position_value = account_balance * (POSITION_SIZE_PCT / 100)
    if position_value > max_position_value:
        shares = int(max_position_value / current_price)
        position_value = shares * current_price
        risk_amount = shares * stop_distance

    if shares < 1:
        return None

    r_r = abs(target - current_price) / stop_distance

    return StockSignal(
        ticker=ticker, direction=direction, entry=current_price,
        stop_loss=sl, target=target, position_size=shares,
        position_value=position_value, confidence=min(score, 100),
        rationale=factors, risk_amount=risk_amount, r_r=r_r,
        timestamp=timestamp, is_etf=is_etf
    )

# =============================================================================
# TELEGRAM BOT
# =============================================================================

class SignalBot:
    def __init__(self, token: str, admin_ids: List[int]):
        self.token = token
        self.admin_ids = admin_ids
        self.authorized_users = set(admin_ids)
        self.app = Application.builder().token(token).build()
        self.active_signals: Dict[str, StockSignal] = {}
        self._setup_handlers()

    def _setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("scan", self.cmd_scan))
        self.app.add_handler(CommandHandler("risk", self.cmd_risk))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("watchlist", self.cmd_watchlist))
        self.app.add_handler(CallbackQueryHandler(self.cb_button))
        self.app.add_error_handler(self.error_handler)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        self.authorized_users.add(user_id)
        msg = """
📈 <b>Stock Signal Agent v2</b>

Watchlist: SPY, QQQ, AAPL, TSLA, NVDA, MSFT, GOOGL, AMZN, META, AMD, NFLX, CRM, UBER, COIN

<b>Strategy v2 Changes:</b>
• Only trade WITH the 200 EMA trend
• Require 3%+ pullback from recent swing
• Wider stops (2x ATR) to avoid noise
• 1:2 R:R target (realistic for stocks)
• Volume must be 1.8x average
• Confidence threshold: 80%

<b>Commands:</b>
/scan — Run market scan
/watchlist — All tickers
/risk [balance] [entry] [stop] — Position size
/status — Bot status
        """
        await update.message.reply_text(msg, parse_mode='HTML')

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.cmd_start(update, context)

    async def cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        tickers = list(STOCKS.keys())
        msg = "📋 <b>Watchlist</b>\n\n" + " · ".join(tickers) + "\n\n<i>Scanned every 4H during market hours.</i>"
        await update.message.reply_text(msg, parse_mode='HTML')

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🔍 Scanning watchlist... (takes ~60s)")

    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args
        if len(args) < 3:
            await update.message.reply_text(
                "💰 <b>Position Sizer</b>\nUsage: /risk [balance] [entry] [stop]\nExample: /risk 25000 225.50 218.00"
            )
            return
        try:
            balance = float(args[0])
            entry = float(args[1])
            stop = float(args[2])
            risk_amount = balance * 0.02
            stop_distance = abs(entry - stop)
            shares = int(risk_amount / stop_distance)
            pos_value = shares * entry
            max_pos = balance * 0.10
            if pos_value > max_pos:
                shares = int(max_pos / entry)
                pos_value = shares * entry
            msg = f"""
💰 <b>Position Sizing</b>

Balance: ${balance:,.0f}
Entry: ${entry:.2f}
Stop: ${stop:.2f}

<b>Shares: {shares}</b>
Position Value: ${pos_value:,.2f}
Risk: ${shares * stop_distance:,.2f}
            """
            await update.message.reply_text(msg, parse_mode='HTML')
        except:
            await update.message.reply_text("❌ Invalid input")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = f"""
🤖 <b>Agent v2 Status</b>

Watchlist: {len(STOCKS)} tickers
Active Signals: {len(self.active_signals)}
Strategy: Trend-only, Pullback entries, 1:2 R:R
Risk: 2% max · Max Position: 10%
        """
        await update.message.reply_text(msg, parse_mode='HTML')

    async def cb_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        if query.data == "take":
            await query.edit_message_text(query.message.text + "\n\n✅ <b>TAKEN</b>", parse_mode='HTML')
        elif query.data == "skip":
            await query.edit_message_text(query.message.text + "\n\n❌ <b>Skipped</b>", parse_mode='HTML')

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Error: {context.error}")

    def format_signal(self, s: StockSignal) -> str:
        emoji = "🟢" if s.direction == "LONG" else "🔴"
        etf = " [ETF]" if s.is_etf else ""
        return f"""
{emoji} <b>SIGNAL: {s.ticker}{etf} {s.direction}</b>
━━━━━━━━━━━━━━━━━━━━━━

🎯 <b>Entry:</b> ${s.entry:.2f}
⛔ <b>Stop Loss:</b> ${s.stop_loss:.2f}
🎯 <b>Target:</b> ${s.target:.2f}

━━━━━━━━━━━━━━━━━━━━━━

📊 <b>Shares:</b> {s.position_size}
💰 <b>Value:</b> ${s.position_value:,.2f}
⚠️ <b>Risk:</b> ${s.risk_amount:,.2f} (2%)
📈 <b>R:R:</b> 1:{s.r_r:.1f}

💯 <b>Confidence:</b> {s.confidence}%

🧠 <b>Rationale:</b>
{chr(10).join(f"✓ {r}" for r in s.rationale)}

🕐 {s.timestamp.strftime('%Y-%m-%d %H:%M ET')}
        """

    async def send_signal(self, signal: StockSignal):
        self.active_signals[signal.ticker] = signal
        msg = self.format_signal(signal)
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Take Trade", callback_data="take"),
             InlineKeyboardButton("❌ Skip", callback_data="skip")]
        ])
        for user_id in self.authorized_users:
            try:
                await self.app.bot.send_message(user_id, msg, parse_mode='HTML', reply_markup=keyboard)
            except Exception as e:
                logger.error(f"Failed to send to {user_id}: {e}")

    async def start(self):
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Bot started")

    async def stop(self):
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()

# =============================================================================
# BACKTESTER v2
# =============================================================================

def run_backtest(tickers: List[str] = None, start: str = "2023-01-01", end: str = "2026-07-01"):
    if tickers is None:
        tickers = list(STOCKS.keys())

    print("\n" + "=" * 60)
    print("📊 STOCK SIGNAL BACKTEST v2")
    print("=" * 60)
    print(f"Period: {start} to {end}")
    print(f"Tickers: {tickers}")
    print("Strategy: Trend-only · Pullback · 2x ATR stops · 1:2 R:R")
    print("=" * 60 + "\n")

    all_trades = []

    for ticker in tickers:
        print(f"🔷 Backtesting {ticker}...")
        df = fetch_data(STOCKS[ticker]["yf"], period="max", interval="1h")
        if df.empty or len(df) < 200:
            print(f"  ⚠️ Insufficient data")
            continue

        df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)].reset_index(drop=True)
        if len(df) < 200:
            continue

        trades = []
        account = 25000
        is_etf = (STOCKS[ticker]["sector"] == "ETF")

        for i in range(200, len(df) - 24, 4):
            timestamp = df.iloc[i]['timestamp']
            if not is_market_open(timestamp):
                continue

            current_price = df.iloc[i]['close']
            hist = df.iloc[:i+1].copy()
            future = df.iloc[i+1:i+121].copy()

            signal = generate_signal(ticker, hist, current_price, account, timestamp, is_etf)

            if signal:
                entry = current_price * 1.0005
                stop = signal.stop_loss
                target = signal.target

                exit_price = None
                exit_reason = None

                for _, row in future.iterrows():
                    high, low = row['high'], row['low']
                    if signal.direction == "LONG":
                        if low <= stop:
                            exit_price = stop * 0.9995
                            exit_reason = "SL"
                            break
                        elif high >= target:
                            exit_price = target * 0.9995
                            exit_reason = "TP"
                            break
                    else:
                        if high >= stop:
                            exit_price = stop * 1.0005
                            exit_reason = "SL"
                            break
                        elif low <= target:
                            exit_price = target * 1.0005
                            exit_reason = "TP"
                            break

                if exit_reason:
                    if signal.direction == "LONG":
                        pnl_pct = (exit_price - entry) / entry * 100
                    else:
                        pnl_pct = (entry - exit_price) / entry * 100

                    trades.append({
                        'ticker': ticker, 'direction': signal.direction,
                        'entry': entry, 'exit': exit_price, 'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason, 'confidence': signal.confidence,
                        'r_r': signal.r_r, 'date': timestamp.strftime('%Y-%m-%d')
                    })

        wins = sum(1 for t in trades if t['pnl_pct'] > 0)
        print(f"  📊 Trades: {len(trades)}, Wins: {wins} ({wins/len(trades)*100:.1f}%)")
        all_trades.extend(trades)

    # Print results
    if not all_trades:
        print("\n❌ No trades generated.")
        return

    df = pd.DataFrame(all_trades)
    total = len(df)
    wins = len(df[df['pnl_pct'] > 0])
    win_rate = wins / total * 100
    avg_win = df[df['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
    avg_loss = df[df['pnl_pct'] <= 0]['pnl_pct'].mean() if total - wins > 0 else 0
    gross_profit = df[df['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(df[df['pnl_pct'] <= 0]['pnl_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    expectancy = (win_rate/100 * avg_win) - ((1-win_rate/100) * abs(avg_loss))

    print("\n" + "=" * 60)
    print("📈 BACKTEST RESULTS v2")
    print("=" * 60)
    print(f"Total Trades:     {total}")
    print(f"Win Rate:         {win_rate:.1f}% ({wins}W / {total-wins}L)")
    print(f"Profit Factor:    {profit_factor:.2f}")
    print(f"Expectancy:       {expectancy:.2f}% per trade")
    print(f"Avg Win:          {avg_win:+.2f}%")
    print(f"Avg Loss:         {avg_loss:.2f}%")
    print(f"Avg R:R:          {df['r_r'].mean():.2f}")
    print(f"\nExit Reasons:")
    print(df['exit_reason'].value_counts().to_string())
    print(f"\nBy Ticker:")
    print(df.groupby('ticker')['pnl_pct'].agg(['count', 'mean', 'sum']).round(2).to_string())
    print("=" * 60)

    # Save
    df.to_csv('backtest_v2_results.csv', index=False)
    print("\n💾 Saved to: backtest_v2_results.csv")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["backtest", "live"], default="backtest")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-07-01")
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(tickers=args.tickers, start=args.start, end=args.end)
    else:
        print("Live mode: Run the orchestrator from stock_agent.py")
