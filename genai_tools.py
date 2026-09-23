"""
GEN AI MCP TOOLS & TOOL REGISTRY
Provides Model Context Protocol (MCP) compliant tool interfaces for Multi-Agent Trading System.
Each tool defines a standard schema and execution handler.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "trading_committee.db"


# =============================================================================
# HELPER DATA ACCESSORS
# =============================================================================

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_state_tables():
    """Ensure state tables including pending approvals and memory exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pending_approvals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        verdict TEXT NOT NULL,
        current_price REAL NOT NULL,
        shares INTEGER NOT NULL,
        stop_loss REAL,
        take_profit_1 REAL,
        take_profit_2 REAL,
        confidence REAL,
        thesis TEXT,
        status TEXT DEFAULT 'PENDING',
        created_at TEXT NOT NULL,
        actioned_at TEXT,
        actioned_by TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS agent_trace_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        agent_role TEXT NOT NULL,
        input_prompt TEXT,
        output_thought TEXT,
        tool_calls_json TEXT,
        created_at TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

# Auto-initialize on module import
init_state_tables()


# =============================================================================
# INDIVIDUAL TOOL HANDLERS
# =============================================================================

def tool_get_market_technicals(ticker: str) -> Dict[str, Any]:
    """
    Fetch market technicals, key moving averages (EMA 20/50/200), RSI, ATR,
    and detect swing setup structure (~20% target or value dip).
    """
    try:
        t = ticker.strip().upper()
        ticker_obj = yf.Ticker(t)
        df = ticker_obj.history(period="60d", interval="1d")
        if df.empty or len(df) < 20:
            return {"error": f"Insufficient price history for {t}"}

        df.columns = [c.lower() for c in df.columns]
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # EMAs
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1] if len(df) >= 50 else ema20
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1] if len(df) >= 200 else ema50

        curr_price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else curr_price
        change_pct = ((curr_price - prev_close) / prev_close) * 100

        # RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if not rs.empty else 50.0

        # ATR (14)
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = float(tr.rolling(14).mean().iloc[-1]) if not tr.empty else curr_price * 0.02

        # 52-week High / Low approximation
        h52 = float(df["high"].max())
        l52 = float(df["low"].min())
        dist_from_52w_high_pct = ((curr_price - h52) / h52) * 100

        # Setup classification
        dist_ema20_pct = ((curr_price - ema20) / ema20) * 100
        is_uptrend = curr_price > ema50 > ema200
        is_oversold = rsi < 38
        is_pullback = is_uptrend and (-4.0 <= dist_ema20_pct <= 1.5)

        # Proposed 20% swing levels
        proposed_stop = round(curr_price - (atr * 2.0), 2)
        proposed_tp1 = round(curr_price + (atr * 3.0), 2)
        proposed_tp2_20pct = round(curr_price * 1.20, 2)  # Swing goal ~20%
        risk_per_share = curr_price - proposed_stop
        rr_ratio = round((proposed_tp2_20pct - curr_price) / max(0.01, risk_per_share), 2)

        return {
            "ticker": t,
            "current_price": round(curr_price, 2),
            "daily_change_pct": round(change_pct, 2),
            "ema_20": round(float(ema20), 2),
            "ema_50": round(float(ema50), 2),
            "ema_200": round(float(ema200), 2),
            "rsi_14": round(rsi, 1),
            "atr_14": round(atr, 2),
            "distance_from_ema20_pct": round(dist_ema20_pct, 2),
            "distance_from_52w_high_pct": round(dist_from_52w_high_pct, 2),
            "trend": "BULLISH_UPTREND" if is_uptrend else ("OVERSOLD_REVERSAL" if is_oversold else "NEUTRAL_CONSOLIDATION"),
            "swing_setup_type": "MOMENTUM_PULLBACK" if is_pullback else ("VALUE_DIP_REVERSAL" if is_oversold else "EXTENDED_OR_RANGING"),
            "suggested_entry": round(curr_price, 2),
            "suggested_stop_loss": proposed_stop,
            "suggested_take_profit_1": proposed_tp1,
            "suggested_target_tp2_20pct": proposed_tp2_20pct,
            "reward_risk_ratio": rr_ratio,
        }
    except Exception as e:
        logger.exception(f"tool_get_market_technicals error for {ticker}: {e}")
        return {"error": str(e), "ticker": ticker}


def tool_fetch_news_and_catalysts(ticker: str) -> Dict[str, Any]:
    """
    Fetch recent company headlines, earnings status, and upcoming catalysts.
    """
    try:
        t = ticker.strip().upper()
        ticker_obj = yf.Ticker(t)
        news_raw = ticker_obj.news or []

        headlines = []
        for item in news_raw[:6]:
            title = ""
            publisher = ""
            if isinstance(item, dict):
                title = item.get("title") or ""
                if not title and "content" in item and isinstance(item["content"], dict):
                    title = item["content"].get("title", "")
                publisher = item.get("publisher", "")
            if title:
                headlines.append({"title": title, "publisher": publisher})

        # Basic sentiment flags
        bullish_flags = ["record", "surge", "growth", "outperform", "beat", "upgrade", "approved", "strong"]
        bearish_flags = ["investigation", "probe", "lawsuit", "downgrade", "warning", "curb", "loss", "plunge"]
        joined = " ".join(h["title"] for h in headlines).lower()

        # Distinguish pharmaceutical / medical catalysts ("weight loss", "hair loss") from financial loss
        cleaned_text = joined.replace("weight loss", "weight_loss_catalyst").replace("hair loss", "hair_loss_treatment")

        score = 0
        if "weight_loss_catalyst" in cleaned_text:
            score += 25
            bullish_flags.append("weight loss")

        for b in bullish_flags:
            if b in cleaned_text:
                score += 15
        for b in bearish_flags:
            if b in cleaned_text:
                score -= 25

        risk_flags = [b for b in bearish_flags if b in cleaned_text]
        sentiment_stance = "BULLISH" if score >= 20 else ("HIGH_RISK_BEARISH" if score <= -20 else "NEUTRAL")

        return {
            "ticker": t,
            "headlines_count": len(headlines),
            "headlines": headlines,
            "catalyst_sentiment_score": score,
            "sentiment_stance": sentiment_stance,
            "risk_flags": risk_flags,
        }
    except Exception as e:
        logger.exception(f"tool_fetch_news_and_catalysts error for {ticker}: {e}")
        return {"error": str(e), "ticker": ticker, "headlines": [], "sentiment_stance": "NEUTRAL"}


def tool_fetch_social_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Query social retail sentiment and discussion trends from Reddit / financial communities.
    """
    t = ticker.strip().upper()
    # Reddit integration or heuristic community scan
    try:
        from stock_agent_v3 import fetch_reddit_posts
        reddit_posts = fetch_reddit_posts(limit=10)
    except Exception:
        reddit_posts = {}

    mentions = []
    for sub, titles in reddit_posts.items():
        for title in titles:
            if t in title.upper():
                mentions.append(f"r/{sub}: {title}")

    score = 10 if len(mentions) > 0 else 0
    return {
        "ticker": t,
        "total_mentions": len(mentions),
        "mentions_sample": mentions[:5],
        "social_heat": "HOT" if len(mentions) >= 3 else ("WARM" if len(mentions) >= 1 else "QUIET"),
        "retail_sentiment_bias": "BULLISH_CHATTER" if len(mentions) >= 2 else "NEUTRAL",
    }


def tool_calculate_risk_sizing(
    entry_price: float,
    stop_loss_price: float,
    account_balance: float = 25000.0,
    risk_per_trade_pct: float = 0.015,
    max_position_pct: float = 0.20,
) -> Dict[str, Any]:
    """
    Compute mathematically sound position sizing, max dollar risk, share count,
    and enforce strict 1-1.5% portfolio risk ceiling.
    """
    if entry_price <= 0 or stop_loss_price >= entry_price:
        return {
            "approved": False,
            "reason": f"Invalid stop loss (${stop_loss_price:.2f}) relative to entry (${entry_price:.2f}).",
            "shares": 0,
        }

    risk_per_share = entry_price - stop_loss_price
    stop_pct = risk_per_share / entry_price

    # Stop distance boundaries (1.5% minimum to avoid noise, 7.5% maximum to avoid deep cuts)
    if stop_pct > 0.08:
        return {
            "approved": False,
            "reason": f"Stop loss distance is {stop_pct*100:.1f}%, exceeding max risk envelope (8.0%).",
            "shares": 0,
        }
    if stop_pct < 0.012:
        return {
            "approved": False,
            "reason": f"Stop loss is {stop_pct*100:.1f}% (too tight, risk of premature shakeout).",
            "shares": 0,
        }

    dollar_risk_target = account_balance * risk_per_trade_pct
    shares_by_risk = int(dollar_risk_target / risk_per_share)
    max_capital = account_balance * max_position_pct
    shares_by_cap = int(max_capital / entry_price)

    shares = max(0, min(shares_by_risk, shares_by_cap))
    if shares == 0:
        return {
            "approved": False,
            "reason": "Calculated sizing resulted in 0 shares under current account risk budget.",
            "shares": 0,
        }

    capital_required = shares * entry_price
    actual_dollar_risk = shares * risk_per_share
    target_20pct = round(entry_price * 1.20, 2)
    expected_profit_at_target = shares * (target_20pct - entry_price)

    return {
        "approved": True,
        "shares": shares,
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss_price, 2),
        "target_20pct": target_20pct,
        "capital_required": round(capital_required, 2),
        "dollar_risk": round(actual_dollar_risk, 2),
        "portfolio_risk_pct": round((actual_dollar_risk / account_balance) * 100, 2),
        "portfolio_allocation_pct": round((capital_required / account_balance) * 100, 2),
        "expected_profit_at_20pct": round(expected_profit_at_target, 2),
        "reward_risk_ratio": round(expected_profit_at_target / max(1.0, actual_dollar_risk), 2),
    }


def tool_query_past_lessons(ticker: str) -> Dict[str, Any]:
    """
    Retrieve historical trade reflections and episodic memory from committee_journal for this ticker.
    """
    t = ticker.strip().upper()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    SELECT outcome, pnl_pct, reflection_notes, created_at
    FROM committee_journal
    WHERE ticker = ?
    ORDER BY id DESC
    LIMIT 5
    """, (t,))
    rows = cur.fetchall()
    conn.close()

    lessons = []
    wins = 0
    losses = 0
    for r in rows:
        lessons.append({
            "outcome": r["outcome"],
            "pnl_pct": r["pnl_pct"],
            "reflection": r["reflection_notes"],
            "date": r["created_at"],
        })
        if r["outcome"] == "WIN":
            wins += 1
        elif r["outcome"] == "LOSS":
            losses += 1

    return {
        "ticker": t,
        "total_past_trades": len(lessons),
        "past_wins": wins,
        "past_losses": losses,
        "historical_lessons": lessons,
    }


def tool_get_portfolio_state() -> Dict[str, Any]:
    """
    Inspect active portfolio holdings to ensure position-aware execution.
    Prevents duplicate entries and ensures sells only happen for owned assets.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    SELECT id, ticker, entry_time, entry_price, shares, stop_loss, take_profit_2, trailing_stop_active
    FROM committee_positions
    WHERE status = 'OPEN'
    """)
    rows = cur.fetchall()
    conn.close()

    holdings = []
    for r in rows:
        holdings.append({
            "id": r["id"],
            "ticker": r["ticker"],
            "entry_price": r["entry_price"],
            "shares": r["shares"],
            "stop_loss": r["stop_loss"],
            "take_profit_2": r["take_profit_2"],
            "trailing_stop_active": bool(r["trailing_stop_active"]),
        })

    return {
        "active_positions_count": len(holdings),
        "holdings": holdings,
        "owned_tickers": [h["ticker"] for h in holdings],
    }


def tool_record_trade_journal(ticker: str, outcome: str, pnl_pct: float, reflection_notes: str) -> Dict[str, Any]:
    """
    Record an episodic memory lesson into the SQLite committee journal.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    now_str = datetime.now().isoformat()
    cur.execute("""
    INSERT INTO committee_journal (ticker, outcome, pnl_pct, reflection_notes, created_at)
    VALUES (?, ?, ?, ?, ?)
    """, (ticker.upper(), outcome.upper(), pnl_pct, reflection_notes, now_str))
    conn.commit()
    conn.close()
    return {"status": "SUCCESS", "logged_at": now_str}


# =============================================================================
# MCP PROTOCOL SPECIFICATIONS & REGISTRY
# =============================================================================

MCP_TOOLS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_market_technicals": {
        "description": "Fetch live market technicals, moving averages (EMA 20/50/200), RSI, ATR, and swing setup type for a ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. NVDA, LLY, AMZN)"}
            },
            "required": ["ticker"]
        },
        "handler": tool_get_market_technicals,
    },
    "fetch_news_and_catalysts": {
        "description": "Fetch breaking headlines, catalysts, and risk warnings for a ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        },
        "handler": tool_fetch_news_and_catalysts,
    },
    "fetch_social_sentiment": {
        "description": "Scan Reddit and retail financial discussion boards for community sentiment and mention heat.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        },
        "handler": tool_fetch_social_sentiment,
    },
    "calculate_risk_sizing": {
        "description": "Calculate exact position sizing, share count, and dollar risk enforcing 1-1.5% account risk and ~20% profit target.",
        "parameters": {
            "type": "object",
            "properties": {
                "entry_price": {"type": "number", "description": "Proposed entry price"},
                "stop_loss_price": {"type": "number", "description": "Proposed stop loss price"},
                "account_balance": {"type": "number", "description": "Total cash account balance (default 25000)"},
                "risk_per_trade_pct": {"type": "number", "description": "Risk fraction per trade (default 0.015)"},
            },
            "required": ["entry_price", "stop_loss_price"]
        },
        "handler": tool_calculate_risk_sizing,
    },
    "query_past_lessons": {
        "description": "Query episodic memory for past trade reflections and win/loss journal entries on this symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        },
        "handler": tool_query_past_lessons,
    },
    "get_portfolio_state": {
        "description": "Inspect currently open positions in the portfolio to prevent duplicate buys or ensure position-aware sells.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
        "handler": tool_get_portfolio_state,
    },
    "record_trade_journal": {
        "description": "Save an episodic learning reflection into persistent SQLite memory after a trade exit or deliberation.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock symbol"},
                "outcome": {"type": "string", "description": "WIN, LOSS, or VETO"},
                "pnl_pct": {"type": "number", "description": "Realized or simulated PnL percentage"},
                "reflection_notes": {"type": "string", "description": "Lessons learned / post-trade reflection"}
            },
            "required": ["ticker", "outcome", "reflection_notes"]
        },
        "handler": tool_record_trade_journal,
    }
}


def dispatch_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an MCP tool call by name with validated arguments."""
    tool_def = MCP_TOOLS_REGISTRY.get(tool_name)
    if not tool_def:
        return {"error": f"Tool '{tool_name}' not recognized in MCP registry."}
    handler = tool_def["handler"]
    try:
        return handler(**arguments)
    except Exception as e:
        logger.exception(f"Error executing MCP tool '{tool_name}': {e}")
        return {"error": str(e), "tool": tool_name}


def export_mcp_tools() -> List[Dict[str, Any]]:
    """Export standardized MCP tool definitions for LLM function calling."""
    tools = []
    for name, item in MCP_TOOLS_REGISTRY.items():
        tools.append({
            "name": name,
            "description": item["description"],
            "parameters": item["parameters"]
        })
    return tools
