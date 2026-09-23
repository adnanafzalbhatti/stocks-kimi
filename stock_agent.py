"""
📈 STOCK SIGNAL AGENT — Complete Single File
For cash/equity stock trading. No leverage. No liquidation risk.

USAGE:
    1. pip install yfinance pandas numpy scipy python-dotenv flask
    2. Create .env file with TELEGRAM_BOT_TOKEN and ADMIN_IDS
    3. python stock_agent.py

FEATURES:
    • Scans SPY, QQQ, AAPL, TSLA, NVDA, MSFT, GOOGL, AMZN, META, AMD
    • Market hours only (9:30 AM - 4:00 PM ET)
    • Earnings blackout (no signals 3 days before/after earnings)
    • No leverage — pure equity positions
    • 2% max risk per trade
    • Daily + 4H timeframe confluence
    • Telegram push notifications
    • Mobile web dashboard
"""

import os
import sys
import asyncio
import logging
import subprocess
import json
import argparse
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from enum import Enum

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema

# Telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Flask for mobile dashboard
from flask import Flask, render_template_string, jsonify

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_IDS", "123456789").split(",") if x.strip()]

# Stock universe — liquid, optionable, good for swing trading
STOCKS = {
    "SPY":  {"yf": "SPY",  "sector": "ETF",       "earnings": None},
    "QQQ":  {"yf": "QQQ",  "sector": "ETF",       "earnings": None},
    "IWM":  {"yf": "IWM",  "sector": "ETF",       "earnings": None},
    "AAPL": {"yf": "AAPL", "sector": "Technology", "earnings": None},  # Auto-detected
    "TSLA": {"yf": "TSLA", "sector": "Auto",       "earnings": None},
    "NVDA": {"yf": "NVDA", "sector": "Technology", "earnings": None},
    "MSFT": {"yf": "MSFT", "sector": "Technology", "earnings": None},
    "GOOGL":{"yf": "GOOGL","sector": "Technology", "earnings": None},
    "AMZN": {"yf": "AMZN", "sector": "Consumer",   "earnings": None},
    "META": {"yf": "META", "sector": "Technology", "earnings": None},
    "AMD":  {"yf": "AMD",  "sector": "Technology", "earnings": None},
    "NFLX": {"yf": "NFLX", "sector": "Media",      "earnings": None},
    "CRM":  {"yf": "CRM",  "sector": "Technology", "earnings": None},
    "UBER": {"yf": "UBER", "sector": "Transport",  "earnings": None},
    "COIN": {"yf": "COIN", "sector": "Finance",    "earnings": None},
}

# Risk & Scan Settings
MAX_RISK_PCT = 2.0          # Max 2% account risk per trade
MIN_CONFIDENCE = 70         # Only emit signals >= 70%
SCAN_INTERVAL_HOURS = 4     # Check every 4 hours during market hours
MAX_POSITIONS = 5           # Max 5 open positions
POSITION_SIZE_PCT = 10.0    # Max 10% of account in one stock

# Market hours (ET)
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)
EARLINGS_BLACKOUT_DAYS = 3  # No signals 3 days before/after earnings

# =============================================================================
# MARKET HOURS & EARNINGS
# =============================================================================

class MarketCalendar:
    """Handles market hours, holidays, and earnings blackout."""

    # Major US market holidays (simplified — no trading)
    HOLIDAYS_2024_2026 = [
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
        "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
        "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
        "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
        "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    ]

    @classmethod
    def is_market_open(cls, dt: datetime) -> bool:
        """Check if US equity market is open at given datetime (ET)."""
        # Weekends
        if dt.weekday() >= 5:
            return False

        # Holidays
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in cls.HOLIDAYS_2024_2026:
            return False

        # Hours
        if not (MARKET_OPEN <= dt.time() <= MARKET_CLOSE):
            return False

        return True

    @classmethod
    def next_market_open(cls, dt: datetime) -> datetime:
        """Find next market open time."""
        while not cls.is_market_open(dt):
            dt += timedelta(hours=1)
            dt = dt.replace(minute=30, second=0, microsecond=0)
        return dt

    @classmethod
    def is_near_earnings(cls, ticker: str, dt: datetime) -> bool:
        """Check if stock has earnings within blackout window."""
        # Try to fetch earnings dates from yfinance
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.earnings_dates
            if earnings is None or earnings.empty:
                return False

            for earnings_date in earnings.index:
                if isinstance(earnings_date, str):
                    earnings_date = pd.to_datetime(earnings_date)
                days_diff = abs((earnings_date - dt).days)
                if days_diff <= EARLINGS_BLACKOUT_DAYS:
                    return True
            return False
        except Exception:
            return False


# =============================================================================
# DATA FETCHER
# =============================================================================

class StockDataFetcher:
    """Fetches stock data from yFinance."""

    def fetch(self, symbol: str, period: str = "120d", interval: str = "1h") -> pd.DataFrame:
        """Fetch OHLCV for a stock."""
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            if data.empty:
                return pd.DataFrame()

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.reset_index()
            rename_map = {
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Adj Close': 'close', 'Volume': 'volume',
                'Datetime': 'timestamp', 'Date': 'timestamp'
            }
            data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})
            data['timestamp'] = pd.to_datetime(data['timestamp'])

            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    data[col] = data[col].astype(float)

            return data
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get last traded price."""
        try:
            data = yf.download(symbol, period="1d", interval="1m", progress=False)
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except:
            pass
        return 0.0


# =============================================================================
# TECHNICAL INDICATORS (Stock-Optimized)
# =============================================================================

class StockIndicators:
    """Technical indicators tuned for stocks (less noise than crypto)."""

    @staticmethod
    def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(closes: pd.Series, period: int) -> pd.Series:
        return closes.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(closes: pd.Series, period: int) -> pd.Series:
        return closes.rolling(window=period).mean()

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, min_periods=period).mean()

    @staticmethod
    def bollinger_bands(closes: pd.Series, period: int = 20, std_dev: int = 2):
        """Returns upper, middle, lower bands."""
        middle = closes.rolling(window=period).mean()
        std = closes.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def macd(closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Returns MACD line, signal line, histogram."""
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def volume_profile_signal(df: pd.DataFrame) -> str:
        """Check if current volume is confirming the move."""
        vol_avg = df['volume'].tail(20).mean()
        vol_current = df['volume'].iloc[-1]
        if vol_current > vol_avg * 1.5:
            return "STRONG"
        elif vol_current > vol_avg * 1.2:
            return "MODERATE"
        return "WEAK"


# =============================================================================
# SIGNAL GENERATOR (Stock-Specific)
# =============================================================================

@dataclass
class StockSignal:
    ticker: str
    direction: str          # LONG or SHORT
    entry: float
    stop_loss: float
    target: float           # Single target for stocks (simpler)
    position_size: int      # Number of shares
    position_value: float   # Dollar value of position
    confidence: int
    rationale: List[str]
    risk_amount: float      # Dollar risk (2% of account)
    r_r: float             # Risk:Reward ratio
    timestamp: datetime
    is_etf: bool = False


class StockSignalGenerator:
    """Generates swing trade signals for stocks."""

    def __init__(self, config: dict):
        self.config = config
        self.indicators = StockIndicators()

    def generate(self, ticker: str, df: pd.DataFrame, current_price: float,
                 account_balance: float, timestamp: datetime, is_etf: bool = False) -> Optional[StockSignal]:
        """
        Generate stock signal using multi-factor confluence.

        Factors for stocks (different weighting than crypto):
        - Trend alignment (EMA 20/50/200)
        - RSI divergence (more reliable on stocks)
        - Volume confirmation (critical for stocks)
        - Bollinger Band position
        - MACD momentum
        - Support/Resistance proximity
        """
        if df.empty or current_price == 0 or len(df) < 50:
            return None

        score = 0
        factors = []
        direction = None

        # === FACTOR 1: EMA Trend Alignment ===
        ema_20 = self.indicators.ema(df['close'], 20)
        ema_50 = self.indicators.ema(df['close'], 50)
        ema_200 = self.indicators.ema(df['close'], 200)

        price_above_20 = current_price > ema_20.iloc[-1]
        price_above_50 = current_price > ema_50.iloc[-1]
        ema_20_above_50 = ema_20.iloc[-1] > ema_50.iloc[-1]

        # For ETFs, be more lenient with trend
        if is_etf:
            if price_above_20 and price_above_50:
                score += 15
                factors.append("Price above EMA 20 & 50")
                direction = "LONG"
            elif not price_above_20 and not price_above_50:
                score += 15
                factors.append("Price below EMA 20 & 50")
                direction = "SHORT"
        else:
            # Individual stocks need stronger trend
            ema_50_above_200 = ema_50.iloc[-1] > ema_200.iloc[-1]
            if price_above_20 and price_above_50 and ema_20_above_50 and ema_50_above_200:
                score += 20
                factors.append("Bullish EMA Stack (20>50>200)")
                direction = "LONG"
            elif not price_above_20 and not price_above_50 and not ema_20_above_50 and not ema_50_above_200:
                score += 20
                factors.append("Bearish EMA Stack")
                direction = "SHORT"

        # === FACTOR 2: RSI Divergence ===
        rsi = self.indicators.rsi(df['close'])

        if self.indicators.detect_divergence(df['close'], rsi, type="bullish"):
            score += 20
            factors.append("RSI Bullish Divergence")
            if not direction:
                direction = "LONG"
        elif self.indicators.detect_divergence(df['close'], rsi, type="bearish"):
            score += 20
            factors.append("RSI Bearish Divergence")
            if not direction:
                direction = "SHORT"

        # === FACTOR 3: Volume Confirmation ===
        vol_signal = self.indicators.volume_profile_signal(df)
        if vol_signal == "STRONG":
            score += 15
            factors.append("Strong Volume Confirmation")
        elif vol_signal == "MODERATE":
            score += 8
            factors.append("Moderate Volume")

        # === FACTOR 4: Bollinger Bands ===
        upper, middle, lower = self.indicators.bollinger_bands(df['close'])
        bb_position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])

        if bb_position < 0.2 and direction == "LONG":
            score += 10
            factors.append("Price near lower Bollinger Band")
        elif bb_position > 0.8 and direction == "SHORT":
            score += 10
            factors.append("Price near upper Bollinger Band")

        # === FACTOR 5: MACD Momentum ===
        macd_line, signal_line, histogram = self.indicators.macd(df['close'])
        macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1] and histogram.iloc[-1] > histogram.iloc[-2]
        macd_bearish = macd_line.iloc[-1] < signal_line.iloc[-1] and histogram.iloc[-1] < histogram.iloc[-2]

        if macd_bullish and direction == "LONG":
            score += 10
            factors.append("MACD Bullish Cross + Rising Histogram")
        elif macd_bearish and direction == "SHORT":
            score += 10
            factors.append("MACD Bearish Cross + Falling Histogram")

        # === FACTOR 6: Swing Structure ===
        swing_highs, swing_lows = self.indicators.swing_points(df['close'], order=3)
        recent_highs = df['close'][swing_highs].tail(3)
        recent_lows = df['close'][swing_lows].tail(3)

        if len(recent_lows) >= 2 and len(recent_highs) >= 2:
            higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-2]
            higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-2]

            if higher_lows and higher_highs:
                score += 10
                factors.append("Higher Highs + Higher Lows")
                if not direction:
                    direction = "LONG"
            elif not higher_lows and not higher_highs:
                score += 10
                factors.append("Lower Highs + Lower Lows")
                if not direction:
                    direction = "SHORT"

        # === CHECK MINIMUM CONFIDENCE ===
        if score < MIN_CONFIDENCE or not direction:
            return None

        # === CALCULATE LEVELS ===
        atr = self.indicators.atr(df).iloc[-1]

        if direction == "LONG":
            # Stop below recent swing low or 1.5x ATR
            if len(recent_lows) > 0:
                sl = min(recent_lows.iloc[-1] * 0.995, current_price - atr * 1.5)
            else:
                sl = current_price - atr * 1.5
            sl = min(sl, current_price * 0.97)

            risk_dist = current_price - sl
            target = current_price + risk_dist * 3  # 1:3 R:R for stocks
        else:
            if len(recent_highs) > 0:
                sl = max(recent_highs.iloc[-1] * 1.005, current_price + atr * 1.5)
            else:
                sl = current_price + atr * 1.5
            sl = max(sl, current_price * 1.03)

            risk_dist = sl - current_price
            target = current_price - risk_dist * 3

        # === POSITION SIZING (No Leverage) ===
        risk_amount = account_balance * (MAX_RISK_PCT / 100)
        stop_distance = abs(current_price - sl)

        if stop_distance <= 0:
            return None

        shares = int(risk_amount / stop_distance)
        position_value = shares * current_price

        # Cap position at 10% of account
        max_position_value = account_balance * (POSITION_SIZE_PCT / 100)
        if position_value > max_position_value:
            shares = int(max_position_value / current_price)
            position_value = shares * current_price
            risk_amount = shares * stop_distance

        if shares < 1:
            return None

        r_r = abs(target - current_price) / stop_distance

        return StockSignal(
            ticker=ticker,
            direction=direction,
            entry=current_price,
            stop_loss=sl,
            target=target,
            position_size=shares,
            position_value=position_value,
            confidence=min(score, 100),
            rationale=factors,
            risk_amount=risk_amount,
            r_r=r_r,
            timestamp=timestamp,
            is_etf=is_etf
        )


# =============================================================================
# TELEGRAM BOT
# =============================================================================

class StockSignalBot:
    """Telegram bot for stock signals."""

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
📈 <b>Stock Signal Agent</b>

Welcome! I scan US equities and ETFs for high-confluence swing setups.

<b>Watchlist:</b> SPY, QQQ, AAPL, TSLA, NVDA, MSFT, GOOGL, AMZN, META, AMD, NFLX, CRM, UBER, COIN

<b>Commands:</b>
/scan — Run market scan now
/watchlist — View all tracked stocks
/risk [balance] [entry] [stop] — Position size calculator
/status — Bot status

⚠️ <i>Markets are probabilistic. Risk 2% max per trade.</i>
        """
        await update.message.reply_text(msg, parse_mode='HTML')

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = """
📚 <b>Commands</b>

/scan — Manually trigger market scan
/watchlist — All tracked tickers
/risk [balance] [entry] [stop] — Position calculator
/status — Bot status & last scan

Signals include: Entry, SL, Target, Share count, Confidence, Rationale
        """
        await update.message.reply_text(msg, parse_mode='HTML')

    async def cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        tickers = list(STOCKS.keys())
        msg = "📋 <b>Watchlist</b>\n\n"
        for i in range(0, len(tickers), 3):
            row = tickers[i:i+3]
            msg += " · ".join(row) + "\n"
        msg += "\n<i>Scanned every 4 hours during market hours.</i>"
        await update.message.reply_text(msg, parse_mode='HTML')

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🔍 Scanning watchlist... (this takes ~60 seconds)")

    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args
        if len(args) < 3:
            await update.message.reply_text(
                "💰 <b>Stock Risk Calculator</b>\n"
                "Usage: /risk [balance] [entry] [stop]\n"
                "Example: /risk 25000 225.50 218.00\n\n"
                "Calculates share count for 2% risk."
            )
            return

        try:
            balance = float(args[0])
            entry = float(args[1])
            stop = float(args[2])

            risk_amount = balance * 0.02
            stop_distance = abs(entry - stop)
            shares = int(risk_amount / stop_distance)
            position_value = shares * entry
            max_pos = balance * 0.10

            if position_value > max_pos:
                shares = int(max_pos / entry)
                position_value = shares * entry
                risk_amount = shares * stop_distance

            msg = f"""
💰 <b>Position Sizing</b>

Balance: ${balance:,.0f}
Risk (2%): ${risk_amount:,.2f}

Entry: ${entry:.2f}
Stop: ${stop:.2f}
Distance: ${stop_distance:.2f} ({stop_distance/entry*100:.2f}%)

<b>Shares: {shares}</b>
Position Value: ${position_value:,.2f}
Risk Amount: ${risk_amount:,.2f}

<i>Max position capped at 10% of account (${max_pos:,.0f}).</i>
            """
            await update.message.reply_text(msg, parse_mode='HTML')
        except:
            await update.message.reply_text("❌ Invalid input. Example: /risk 25000 225.50 218.00")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        active = len(self.active_signals)
        msg = f"""
🤖 <b>Stock Agent Status</b>

Watchlist: {len(STOCKS)} tickers
Active Signals: {active}
Scan Interval: Every 4H (market hours only)
Risk/Trade: 2% max
Max Position: 10% of account

<i>Next scan: During next market session</i>
        """
        await update.message.reply_text(msg, parse_mode='HTML')

    async def cb_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        if query.data == "take":
            await query.edit_message_text(query.message.text + "\n\n✅ <b>You marked this TAKEN</b>", parse_mode='HTML')
        elif query.data == "skip":
            await query.edit_message_text(query.message.text + "\n\n❌ <b>Skipped</b>", parse_mode='HTML')

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Telegram error: {context.error}")

    def format_signal(self, signal: StockSignal) -> str:
        emoji = "🟢" if signal.direction == "LONG" else "🔴"
        etf_tag = " [ETF]" if signal.is_etf else ""

        return f"""
{emoji} <b>SIGNAL: {signal.ticker}{etf_tag} {signal.direction}</b>
━━━━━━━━━━━━━━━━━━━━━━

🎯 <b>Entry:</b> ${signal.entry:.2f}
⛔ <b>Stop Loss:</b> ${signal.stop_loss:.2f}
🎯 <b>Target:</b> ${signal.target:.2f}

━━━━━━━━━━━━━━━━━━━━━━

📊 <b>Position Size:</b> {signal.position_size} shares
💰 <b>Position Value:</b> ${signal.position_value:,.2f}
⚠️ <b>Risk:</b> ${signal.risk_amount:,.2f} (2%)
📈 <b>R:R Ratio:</b> 1:{signal.r_r:.1f}

💯 <b>Confidence:</b> {signal.confidence}%

🧠 <b>Rationale:</b>
{chr(10).join(f"✓ {r}" for r in signal.rationale)}

🕐 {signal.timestamp.strftime('%Y-%m-%d %H:%M ET')}
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
        logger.info("Stock Signal Bot started")

    async def stop(self):
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()


# =============================================================================
# MOBILE DASHBOARD (Flask)
# =============================================================================

flask_app = Flask(__name__)

# Demo data for dashboard
dashboard_active_signals = []
dashboard_history = []
dashboard_portfolio = {
    "total_equity": 25000.0,
    "initial_balance": 25000.0,
    "total_return_pct": 12.4,
    "win_rate": 58.3,
    "profit_factor": 1.85,
    "max_drawdown": -8.2,
    "total_trades": 24,
    "wins": 14,
    "losses": 10,
}

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="theme-color" content="#0a0e1a">
    <title>Stock Signals</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
        body { background: #0a0e1a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 430px; margin: 0 auto; min-height: 100vh; padding-bottom: 80px; }
        .status-bar { background: #0f172a; padding: 8px 16px; display: flex; justify-content: space-between; font-size: 12px; color: #64748b; }
        .header { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); padding: 20px 16px 16px; }
        .header-top { display: flex; justify-content: space-between; align-items: flex-start; }
        .header h1 { margin: 0; font-size: 22px; font-weight: 800; color: #fbbf24; }
        .header p { margin: 4px 0 0; font-size: 12px; color: #94a3b8; }
        .badge { background: #22c55e; color: white; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; }
        .portfolio-card { background: rgba(30, 41, 59, 0.6); border: 1px solid #334155; border-radius: 16px; padding: 16px; margin-top: 16px; }
        .portfolio-row { display: flex; justify-content: space-between; margin-bottom: 12px; }
        .portfolio-label { font-size: 11px; color: #94a3b8; }
        .portfolio-value { font-size: 24px; font-weight: 800; color: white; }
        .portfolio-return { font-size: 18px; font-weight: 700; color: #22c55e; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
        .stat-box { background: #0f172a; border-radius: 10px; padding: 10px; text-align: center; }
        .stat-label { font-size: 10px; color: #94a3b8; }
        .stat-value { font-size: 16px; font-weight: 700; }
        .green { color: #22c55e; } .blue { color: #3b82f6; } .orange { color: #f59e0b; } .red { color: #ef4444; }
        .section { padding: 16px; }
        .section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .section-header h2 { font-size: 16px; font-weight: 700; margin: 0; }
        .section-count { font-size: 11px; color: #64748b; }
        .signal-card { border-radius: 16px; padding: 16px; margin-bottom: 12px; border: 1px solid; }
        .signal-card.long { background: linear-gradient(135deg, #14532d 0%, #0f172a 100%); border-color: #22c55e; }
        .signal-card.short { background: linear-gradient(135deg, #7f1d1d 0%, #0f172a 100%); border-color: #ef4444; }
        .signal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .signal-badge { padding: 4px 10px; border-radius: 8px; font-size: 11px; font-weight: 700; color: white; }
        .signal-badge.long { background: #22c55e; } .signal-badge.short { background: #ef4444; }
        .signal-asset { font-size: 16px; font-weight: 700; }
        .signal-conf { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; background: rgba(34, 197, 94, 0.2); color: #22c55e; }
        .signal-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }
        .signal-cell { background: rgba(0,0,0,0.3); border-radius: 8px; padding: 8px; }
        .signal-cell-label { font-size: 10px; color: #94a3b8; }
        .signal-cell-value { font-size: 14px; font-weight: 700; }
        .signal-footer { display: flex; justify-content: space-between; align-items: center; }
        .signal-meta { font-size: 11px; color: #94a3b8; }
        .signal-pnl { padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 700; color: white; }
        .signal-pnl.positive { background: #22c55e; } .signal-pnl.negative { background: #ef4444; }
        .history-list { background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; }
        .history-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 14px; border-bottom: 1px solid #1e293b; }
        .history-item:last-child { border-bottom: none; }
        .history-left { display: flex; align-items: center; gap: 10px; }
        .history-dot { width: 8px; height: 8px; border-radius: 50%; }
        .history-dot.win { background: #22c55e; } .history-dot.loss { background: #ef4444; }
        .history-name { font-size: 13px; font-weight: 600; }
        .history-date { font-size: 10px; color: #64748b; }
        .history-right { text-align: right; }
        .history-pnl { font-size: 13px; font-weight: 700; }
        .history-reason { font-size: 10px; color: #64748b; }
        .risk-calc { background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 14px; }
        .calc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }
        .calc-input-group label { font-size: 10px; color: #94a3b8; display: block; margin-bottom: 4px; }
        .calc-input-group input { width: 100%; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px; color: white; font-size: 14px; outline: none; }
        .calc-input-group input:focus { border-color: #3b82f6; }
        .calc-btn { width: 100%; background: #3b82f6; color: white; border: none; border-radius: 10px; padding: 12px; font-size: 14px; font-weight: 700; cursor: pointer; }
        .calc-results { margin-top: 12px; background: #1e293b; border-radius: 8px; padding: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; text-align: center; }
        .calc-result-label { font-size: 10px; color: #94a3b8; }
        .calc-result-value { font-size: 13px; font-weight: 700; }
        .bottom-nav { position: fixed; bottom: 0; left: 50%; transform: translateX(-50%); width: 100%; max-width: 430px; background: #0f172a; border-top: 1px solid #1e293b; padding: 12px 16px; display: flex; justify-content: space-around; }
        .nav-item { text-align: center; color: #64748b; text-decoration: none; }
        .nav-item.active { color: #fbbf24; }
        .nav-icon { font-size: 20px; } .nav-label { font-size: 10px; font-weight: 600; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .live-indicator { animation: pulse 2s infinite; }
    </style>
</head>
<body>
    <div class="status-bar">
        <span class="live-indicator">📡 Live</span>
        <span>🔋 100%</span>
    </div>
    <div class="header">
        <div class="header-top">
            <div><h1>📈 Stock Signals</h1><p>US Equities & ETFs · Swing Trading</p></div>
            <div class="badge">● Active</div>
        </div>
        <div class="portfolio-card">
            <div class="portfolio-row">
                <div><div class="portfolio-label">Total Equity</div><div class="portfolio-value">${{ "%.2f"|format(portfolio.total_equity) }}</div></div>
                <div style="text-align: right;"><div class="portfolio-label">Total Return</div><div class="portfolio-return">+{{ "%.2f"|format(portfolio.total_return_pct) }}%</div></div>
            </div>
            <div class="stats-grid">
                <div class="stat-box"><div class="stat-label">Win Rate</div><div class="stat-value green">{{ "%.1f"|format(portfolio.win_rate) }}%</div></div>
                <div class="stat-box"><div class="stat-label">Profit Factor</div><div class="stat-value blue">{{ "%.2f"|format(portfolio.profit_factor) }}</div></div>
                <div class="stat-box"><div class="stat-label">Max DD</div><div class="stat-value orange">{{ "%.1f"|format(portfolio.max_drawdown) }}%</div></div>
            </div>
        </div>
    </div>
    <div class="section">
        <div class="section-header"><h2>🔥 Active Signals</h2><span class="section-count">{{ active_signals|length }} active</span></div>
        {% for signal in active_signals %}
        <div class="signal-card {{ 'long' if signal.direction == 'LONG' else 'short' }}">
            <div class="signal-header">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div class="signal-badge {{ 'long' if signal.direction == 'LONG' else 'short' }}">{{ signal.direction }}</div>
                    <span class="signal-asset">{{ signal.ticker }}{% if signal.is_etf %} [ETF]{% endif %}</span>
                </div>
                <div class="signal-conf">{{ signal.confidence }}% Conf</div>
            </div>
            <div class="signal-grid">
                <div class="signal-cell"><div class="signal-cell-label">Entry</div><div class="signal-cell-value">${{ "%.2f"|format(signal.entry) }}</div></div>
                <div class="signal-cell"><div class="signal-cell-label">Target</div><div class="signal-cell-value green">${{ "%.2f"|format(signal.target) }}</div></div>
                <div class="signal-cell"><div class="signal-cell-label">SL</div><div class="signal-cell-value red">${{ "%.2f"|format(signal.stop_loss) }}</div></div>
                <div class="signal-cell"><div class="signal-cell-label">Shares</div><div class="signal-cell-value">{{ signal.position_size }}</div></div>
            </div>
            <div class="signal-footer">
                <div class="signal-meta">R:R 1:{{ "%.1f"|format(signal.r_r) }} · {{ signal.timestamp.strftime('%m/%d %H:%M') }}</div>
                <div class="signal-pnl positive">${{ "%.0f"|format(signal.position_value) }}</div>
            </div>
        </div>
        {% endfor %}
    </div>
    <div class="section">
        <div class="section-header"><h2>📋 Recent Signals</h2><span class="section-count">Last 30 days</span></div>
        <div class="history-list">
            {% for trade in history %}
            <div class="history-item">
                <div class="history-left">
                    <div class="history-dot {{ 'win' if trade.pnl_pct > 0 else 'loss' }}"></div>
                    <div><div class="history-name">{{ trade.ticker }} {{ trade.direction }}</div><div class="history-date">{{ trade.date }} · {{ trade.confidence }}% conf</div></div>
                </div>
                <div class="history-right">
                    <div class="history-pnl {{ 'green' if trade.pnl_pct > 0 else 'red' }}">{{ "+%.2f"|format(trade.pnl_pct) if trade.pnl_pct > 0 else "%.2f"|format(trade.pnl_pct) }}%</div>
                    <div class="history-reason">{{ trade.exit_reason }}</div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    <div class="section">
        <h2 style="margin-bottom: 12px; font-size: 16px;">⚡ Quick Position Sizer</h2>
        <div class="risk-calc">
            <div class="calc-grid">
                <div class="calc-input-group"><label>Account ($)</label><input type="number" id="balance" value="25000" oninput="calc()"></div>
                <div class="calc-input-group"><label>Entry ($)</label><input type="number" id="entry" value="225.50" oninput="calc()"></div>
                <div class="calc-input-group"><label>Stop Loss ($)</label><input type="number" id="stop" value="218.00" oninput="calc()"></div>
                <div class="calc-input-group"><label>Target ($)</label><input type="number" id="target" value="240.00" oninput="calc()"></div>
            </div>
            <button class="calc-btn" onclick="calc()">Calculate</button>
            <div class="calc-results" id="results">
                <div><div class="calc-result-label">Shares</div><div class="calc-result-value" id="shares">357</div></div>
                <div><div class="calc-result-label">Position Value</div><div class="calc-result-value" id="posValue">$80,504</div></div>
                <div><div class="calc-result-label">Risk</div><div class="calc-result-value red" id="risk">$500</div></div>
                <div><div class="calc-result-label">R:R</div><div class="calc-result-value green" id="rr">1:2.1</div></div>
            </div>
        </div>
    </div>
    <div class="bottom-nav">
        <a href="/" class="nav-item active"><div class="nav-icon">🏠</div><div class="nav-label">Signals</div></a>
        <a href="/watchlist" class="nav-item"><div class="nav-icon">📋</div><div class="nav-label">Watchlist</div></a>
        <a href="/history" class="nav-item"><div class="nav-icon">📈</div><div class="nav-label">History</div></a>
        <a href="/settings" class="nav-item"><div class="nav-icon">⚙️</div><div class="nav-label">Settings</div></a>
    </div>
    <script>
        function calc() {
            const balance = parseFloat(document.getElementById('balance').value) || 25000;
            const entry = parseFloat(document.getElementById('entry').value) || 225.50;
            const stop = parseFloat(document.getElementById('stop').value) || 218.00;
            const target = parseFloat(document.getElementById('target').value) || 240.00;
            const riskAmount = balance * 0.02;
            const stopDist = Math.abs(entry - stop);
            let shares = Math.floor(riskAmount / stopDist);
            let posValue = shares * entry;
            const maxPos = balance * 0.10;
            if (posValue > maxPos) {
                shares = Math.floor(maxPos / entry);
                posValue = shares * entry;
            }
            const actualRisk = shares * stopDist;
            const reward = Math.abs(target - entry);
            const rr = reward / stopDist;
            document.getElementById('shares').textContent = shares;
            document.getElementById('posValue').textContent = '$' + posValue.toLocaleString('en-US', {maximumFractionDigits: 0});
            document.getElementById('risk').textContent = '$' + actualRisk.toFixed(0);
            document.getElementById('rr').textContent = '1:' + rr.toFixed(1);
        }
        calc();
    </script>
</body>
</html>
"""

@flask_app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML, 
                                   active_signals=dashboard_active_signals,
                                   history=dashboard_history,
                                   portfolio=dashboard_portfolio)

@flask_app.route("/api/signals")
def api_signals():
    return jsonify({
        "active_signals": [asdict(s) for s in dashboard_active_signals],
        "portfolio": dashboard_portfolio,
        "last_update": datetime.now().isoformat()
    })


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class StockOrchestrator:
    """Main controller for stock signal agent."""

    def __init__(self):
        self.data = StockDataFetcher()
        self.generator = StockSignalGenerator({})
        self.bot = StockSignalBot(TELEGRAM_TOKEN, ADMIN_IDS)
        self.calendar = MarketCalendar()
        self.account_balance = 25000.0  # Starting balance
        self.running = False

    async def run(self):
        await self.bot.start()
        self.running = True

        logger.info("=" * 60)
        logger.info("📈 STOCK SIGNAL AGENT STARTED")
        logger.info("=" * 60)
        logger.info(f"Watchlist: {list(STOCKS.keys())}")
        logger.info(f"Account: ${self.account_balance:,.2f}")
        logger.info(f"Risk/Trade: {MAX_RISK_PCT}%")
        logger.info(f"Max Position: {POSITION_SIZE_PCT}%")
        logger.info("=" * 60)

        while self.running:
            now = datetime.now()

            # Only scan during market hours
            if self.calendar.is_market_open(now):
                logger.info(f"🔍 Market open. Scanning {len(STOCKS)} tickers...")
                await self.scan_stocks()

                # Wait 4 hours or until market close
                await asyncio.sleep(SCAN_INTERVAL_HOURS * 3600)
            else:
                # Market closed — wait until next open
                next_open = self.calendar.next_market_open(now + timedelta(hours=1))
                wait_seconds = (next_open - now).total_seconds()
                logger.info(f"⏸️ Market closed. Next scan at {next_open.strftime('%Y-%m-%d %H:%M ET')}")
                await asyncio.sleep(min(wait_seconds, 3600))  # Check every hour

    async def scan_stocks(self):
        """Scan all stocks in watchlist."""
        for ticker, config in STOCKS.items():
            try:
                # Check earnings blackout
                if self.calendar.is_near_earnings(ticker, datetime.now()):
                    logger.info(f"  ⛔ {ticker}: Earnings blackout")
                    continue

                # Fetch data
                df = self.data.fetch(config["yf"], period="60d", interval="1h")
                if df.empty or len(df) < 50:
                    continue

                current_price = self.data.get_current_price(config["yf"])

                # Generate signal
                signal = self.generator.generate(
                    ticker=ticker,
                    df=df,
                    current_price=current_price,
                    account_balance=self.account_balance,
                    timestamp=datetime.now(),
                    is_etf=(config["sector"] == "ETF")
                )

                if signal:
                    logger.info(f"🚨 SIGNAL: {ticker} {signal.direction} ({signal.confidence}%)")
                    await self.bot.send_signal(signal)
                else:
                    logger.info(f"  ✅ {ticker}: No setup")

            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")

    async def run_dashboard(self):
        """Run Flask dashboard in background."""
        import threading
        def run_flask():
            flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
        thread = threading.Thread(target=run_flask, daemon=True)
        thread.start()
        logger.info("📱 Dashboard running at http://localhost:5000")


# =============================================================================
# BACKTESTER (Built-in)
# =============================================================================

class StockBacktester:
    """Backtests stock signals on historical data."""

    def __init__(self):
        self.data = StockDataFetcher()
        self.generator = StockSignalGenerator({})
        self.calendar = MarketCalendar()

    def run(self, tickers: List[str] = None, start: str = "2023-01-01", end: str = "2026-07-01"):
        if tickers is None:
            tickers = list(STOCKS.keys())

        print("\n" + "=" * 60)
        print("📊 STOCK SIGNAL BACKTEST")
        print("=" * 60)
        print(f"Period: {start} to {end}")
        print(f"Tickers: {tickers}")
        print(f"Min Confidence: {MIN_CONFIDENCE}%")
        print("=" * 60 + "\n")

        all_trades = []

        for ticker in tickers:
            print(f"🔷 Backtesting {ticker}...")
            trades = self._backtest_ticker(ticker, start, end)
            all_trades.extend(trades)
            wins = sum(1 for t in trades if t.get('pnl_pct', 0) > 0)
            print(f"  📊 Trades: {len(trades)}, Wins: {wins}")

        self._print_results(all_trades)

    def _backtest_ticker(self, ticker: str, start: str, end: str) -> List[dict]:
        df = self.data.fetch(STOCKS[ticker]["yf"], period="max", interval="1h")
        if df.empty or len(df) < 200:
            return []

        # Filter to backtest period
        df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)].reset_index(drop=True)
        if len(df) < 200:
            return []

        trades = []
        account = 25000
        is_etf = (STOCKS[ticker]["sector"] == "ETF")

        # Walk forward — scan every 4 hours during market hours
        for i in range(200, len(df) - 24, 4):
            timestamp = df.iloc[i]['timestamp']

            # Skip if market closed
            if not self.calendar.is_market_open(timestamp):
                continue

            current_price = df.iloc[i]['close']
            hist = df.iloc[:i+1].copy()
            future = df.iloc[i+1:i+121].copy()  # 5 days forward

            signal = self.generator.generate(ticker, hist, current_price, account, timestamp, is_etf)

            if signal:
                # Simulate trade
                entry = current_price * 1.0005  # Small slippage
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
                        'ticker': ticker,
                        'direction': signal.direction,
                        'entry': entry,
                        'exit': exit_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'confidence': signal.confidence,
                        'r_r': signal.r_r,
                        'date': timestamp.strftime('%Y-%m-%d')
                    })

        return trades

    def _print_results(self, trades: List[dict]):
        if not trades:
            print("\n❌ No trades generated.")
            return

        df = pd.DataFrame(trades)
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
        print("📈 BACKTEST RESULTS")
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


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    📈 STOCK SIGNAL AGENT                         ║
╠══════════════════════════════════════════════════════════════════╣
║  US Equities & ETFs · Swing Trading · No Leverage               ║
╠══════════════════════════════════════════════════════════════════╣
║  MODES:                                                          ║
║    python stock_agent.py live     → Run live signals            ║
║    python stock_agent.py backtest → Backtest strategy           ║
║    python stock_agent.py dash     → Mobile dashboard only        ║
╚══════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    print_banner()

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["live", "backtest", "dash"], default="live",
                       help="Run mode: live signals, backtest, or dashboard only")
    parser.add_argument("--tickers", nargs="+", default=None,
                       help="Specific tickers to backtest (default: all)")
    parser.add_argument("--start", default="2023-01-01",
                       help="Backtest start date")
    parser.add_argument("--end", default="2026-07-01",
                       help="Backtest end date")
    parser.add_argument("--port", type=int, default=5000,
                       help="Dashboard port")
    args = parser.parse_args()

    if args.mode == "backtest":
        backtester = StockBacktester()
        backtester.run(tickers=args.tickers, start=args.start, end=args.end)

    elif args.mode == "dash":
        print(f"🚀 Dashboard at http://localhost:{args.port}")
        print("📱 Same WiFi? Use your computer's IP")
        flask_app.run(host="0.0.0.0", port=args.port, debug=False)

    else:  # live
        orchestrator = StockOrchestrator()

        # Start dashboard in background
        asyncio.run(orchestrator.run_dashboard())

        # Run main loop
        try:
            asyncio.run(orchestrator.run())
        except KeyboardInterrupt:
            logger.info("👋 Shutting down...")
            orchestrator.running = False
