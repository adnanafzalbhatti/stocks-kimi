"""
GEN AI MULTI-AGENT TRADING COMMITTEE — TRUE AGENTIC VERSION
============================================================
The LLM (Gemini 3.5 Flash-Lite) is the REASONING ENGINE that:
  1. Decides which tools to call (not hardcoded)
  2. Inspects tool outputs and reasons about them
  3. Can call multiple tools in variable order
  4. Synthesizes nuanced investment theses with natural language understanding
  5. Falls back to deterministic scoring ONLY when the API is unavailable

Architecture:
  - Supervisor-Worker pattern with Gemini function calling
  - Each worker sends its system prompt + available tools to the LLM
  - The LLM autonomously selects tools, interprets results, and produces structured output
  - Agent trace logging captures every LLM prompt, tool call, and response
  - Deterministic fallback preserves 100% uptime when API is down
"""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

import genai_tools

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM SDK availability
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOOL_ROUNDS = int(os.getenv("LLM_MAX_TOOL_ROUNDS", "8"))


# =============================================================================
# SHARED CONTEXT BLACKBOARD
# =============================================================================

@dataclass
class SharedBlackboard:
    """
    Central shared state blackboard for the multi-agent committee.
    Accessible to Supervisor and all Worker Agents.
    """
    ticker: str
    macro_regime: str = "BULL"
    current_price: float = 0.0
    is_etf: bool = False
    technicals: Dict[str, Any] = field(default_factory=dict)
    sentiment: Dict[str, Any] = field(default_factory=dict)
    risk: Dict[str, Any] = field(default_factory=dict)
    past_lessons: Dict[str, Any] = field(default_factory=dict)
    portfolio_state: Dict[str, Any] = field(default_factory=dict)
    dialogue: List[Dict[str, Any]] = field(default_factory=list)
    verdict: str = "HOLD_CASH"
    confidence: float = 0.0
    shares: int = 0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    investment_memo: str = ""
    human_approval_id: Optional[int] = None
    approval_status: str = "NONE"  # NONE, PENDING_APPROVAL, APPROVED, REJECTED
    llm_mode: str = "FALLBACK"  # "LLM" or "FALLBACK" — tracks which engine was used

    def post_update(self, agent_name: str, stance: str, details: Dict[str, Any], notes: str):
        entry = {
            "agent": agent_name,
            "stance": stance,
            "timestamp": datetime.now().isoformat(),
            "notes": notes,
            "details": details,
        }
        self.dialogue.append(entry)


# =============================================================================
# AGENT TRACE LOGGER
# =============================================================================

class AgentTraceLogger:
    """Logs every LLM prompt, tool call, and response to SQLite for full audit trail."""

    @staticmethod
    def log(ticker: str, agent_role: str, input_prompt: str,
            output_thought: str, tool_calls: List[Dict[str, Any]] | None = None):
        try:
            conn = genai_tools.get_db_connection()
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO agent_trace_logs (ticker, agent_role, input_prompt, output_thought, tool_calls_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ticker, agent_role,
                input_prompt[:4000] if input_prompt else "",
                output_thought[:4000] if output_thought else "",
                json.dumps(tool_calls or [])[:8000],
                datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"AgentTraceLogger error: {e}")


# =============================================================================
# GLOBAL RATE LIMITER (free-tier safe)
# =============================================================================

class _GeminiRateLimiter:
    """Thread-safe rate limiter for Gemini API calls.
    
    Enforces a minimum interval between API calls to stay within free-tier limits.
    gemini-3.5-flash-lite: tested at ~12 req with 1s gaps OK.
    We use a conservative 3s minimum gap to handle multi-round tool-calling loops.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._last_call_ts = 0.0
            cls._instance._min_interval = float(os.getenv("LLM_RATE_LIMIT_SECONDS", "3.0"))
            cls._instance._call_lock = threading.Lock()
        return cls._instance

    def wait(self):
        """Block until it's safe to make the next API call."""
        with self._call_lock:
            now = time.time()
            elapsed = now - self._last_call_ts
            if elapsed < self._min_interval:
                sleep_for = self._min_interval - elapsed
                logger.debug(f"[RateLimiter] Sleeping {sleep_for:.1f}s before next API call")
                time.sleep(sleep_for)
            self._last_call_ts = time.time()


# Global singleton
rate_limiter = _GeminiRateLimiter()


# =============================================================================
# BASE AGENT WITH REAL LLM TOOL-CALLING LOOP
# =============================================================================

class GenAIAgentBase:
    """
    Base agent with REAL Gemini function calling.
    The LLM is the reasoning engine — it decides which tools to call,
    inspects results, and produces structured output.
    Falls back to deterministic scoring when API is unavailable.
    """

    def __init__(self, name: str, role: str, system_prompt: str,
                 tools: List[str] | None = None, model: str | None = None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model or LLM_MODEL
        self.tool_names = tools or []  # names from MCP_TOOLS_REGISTRY
        self.gemini_client = None
        self._llm_available = False

        if HAS_GOOGLE_GENAI and GEMINI_API_KEY:
            try:
                self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                self._llm_available = True
                logger.info(f"[{self.name}] Gemini client initialized (model={self.model})")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to initialize Gemini client: {e}")

    def _build_tool_functions(self) -> list:
        """
        Build Python function references for Gemini automatic function calling.
        The SDK inspects the function signatures + docstrings and creates
        FunctionDeclarations automatically.
        """
        functions = []
        for tool_name in self.tool_names:
            tool_def = genai_tools.MCP_TOOLS_REGISTRY.get(tool_name)
            if tool_def and "handler" in tool_def:
                functions.append(tool_def["handler"])
        return functions

    def _build_manual_tool_declarations(self) -> genai_types.Tool | None:
        """
        Build manual FunctionDeclarations from MCP registry schemas.
        This gives the LLM the tool schemas so it can decide which to call.
        """
        if not HAS_GOOGLE_GENAI:
            return None

        declarations = []
        for tool_name in self.tool_names:
            tool_def = genai_tools.MCP_TOOLS_REGISTRY.get(tool_name)
            if tool_def:
                fd = genai_types.FunctionDeclaration(
                    name=tool_name,
                    description=tool_def["description"],
                    parameters_json_schema=tool_def["parameters"],
                )
                declarations.append(fd)

        if declarations:
            return genai_types.Tool(function_declarations=declarations)
        return None

    def reason_with_tools(self, user_prompt: str, ticker: str = "") -> Tuple[str, List[Dict[str, Any]]]:
        """
        CORE AGENTIC LOOP: Send prompt + tool schemas to Gemini.
        The LLM decides which tools to call, we execute them, feed results back.
        Loop continues until LLM produces a text response (natural termination).

        Returns:
            (llm_text_response, list_of_tool_calls_made)
        """
        if not self._llm_available:
            return "", []

        tool_decl = self._build_manual_tool_declarations()
        tool_calls_log = []

        # Build initial conversation
        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=user_prompt)],
            )
        ]

        config = genai_types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=LLM_TEMPERATURE,
            tools=[tool_decl] if tool_decl else None,
        )

        delay = 1.0
        for round_num in range(LLM_MAX_TOOL_ROUNDS):
            try:
                rate_limiter.wait()  # Respect free-tier rate limits
                response = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                logger.warning(f"[{self.name}] LLM call failed (round {round_num}): {e}")
                # Retry with backoff for transient errors
                if round_num < 2:
                    jitter = random.uniform(0.2, 0.6)
                    time.sleep(delay + jitter)
                    delay *= 2.0
                    continue
                return "", tool_calls_log

            # Check if the model wants to call functions
            if response.function_calls:
                # The LLM decided to call tools — this is true agentic behavior
                model_content = response.candidates[0].content
                contents.append(model_content)

                # Execute each function call the LLM requested
                function_response_parts = []
                for fc in response.function_calls:
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    logger.info(f"[{self.name}] LLM invoked tool: {tool_name}({tool_args})")

                    # Execute via our MCP dispatcher
                    try:
                        result = genai_tools.dispatch_tool_call(tool_name, tool_args)
                    except Exception as e:
                        result = {"error": str(e), "tool": tool_name}

                    tool_calls_log.append({
                        "round": round_num,
                        "tool": tool_name,
                        "args": tool_args,
                        "result_summary": str(result)[:500],
                    })

                    # Feed tool result back to the LLM as a function response
                    fr_part = genai_types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result},
                    )
                    function_response_parts.append(fr_part)

                # Add all function responses as a single turn
                # NOTE: Gemini 3.x models require role='user' for function responses
                # (role='tool' is deprecated and returns 400 INVALID_ARGUMENT)
                contents.append(genai_types.Content(
                    role="user",
                    parts=function_response_parts,
                ))
                continue  # Let the LLM reason about the results

            # No function calls — the LLM produced a final text response
            if response.text:
                return response.text, tool_calls_log

            # Empty response — shouldn't happen but handle gracefully
            logger.warning(f"[{self.name}] Empty response at round {round_num}")
            return "", tool_calls_log

        # Exhausted max rounds
        logger.warning(f"[{self.name}] Exhausted {LLM_MAX_TOOL_ROUNDS} tool rounds")
        return "", tool_calls_log


# =============================================================================
# DETERMINISTIC FALLBACK WORKERS (preserved from original for API-down scenarios)
# =============================================================================

def _fallback_technicals(blackboard: SharedBlackboard) -> Dict[str, Any]:
    """Original deterministic scoring — used ONLY when Gemini API is unavailable."""
    raw_tech = genai_tools.dispatch_tool_call("get_market_technicals", {"ticker": blackboard.ticker})
    blackboard.technicals = raw_tech

    if "error" in raw_tech:
        return {"stance": "NEUTRAL", "confidence_score": 0, "setup_summary": f"Data error: {raw_tech['error']}"}

    curr_px = raw_tech.get("current_price", 0.0)
    blackboard.current_price = curr_px
    setup_type = raw_tech.get("swing_setup_type", "EXTENDED_OR_RANGING")
    rsi = raw_tech.get("rsi_14", 50.0)
    trend = raw_tech.get("trend", "NEUTRAL_CONSOLIDATION")

    score = 50
    if trend == "BULLISH_UPTREND":
        score += 25
    elif trend == "OVERSOLD_REVERSAL":
        score += 20
    if setup_type == "MOMENTUM_PULLBACK":
        score += 20
    elif setup_type == "VALUE_DIP_REVERSAL":
        score += 20
    elif setup_type == "EXTENDED_OR_RANGING":
        score -= 20
    if 40 <= rsi <= 55:
        score += 10
    elif rsi < 35:
        score += 15
    score = max(10, min(95, score))

    if score >= 75:
        stance = "BULLISH"
    elif score >= 60:
        stance = "LEAN_BULLISH"
    elif score <= 40:
        stance = "BEARISH"
    else:
        stance = "NEUTRAL"

    summary = (
        f"{blackboard.ticker} at ${curr_px:.2f} ({trend}). Setup: {setup_type}. "
        f"RSI={rsi:.1f}, dist from 20 EMA={raw_tech.get('distance_from_ema20_pct', 0):+.1f}%. "
        f"Target 20% swing: ${raw_tech.get('suggested_target_tp2_20pct', 0):.2f}."
    )

    return {
        "stance": stance,
        "confidence_score": score,
        "setup_summary": summary,
        "levels": {
            "entry": curr_px,
            "stop_loss": raw_tech.get("suggested_stop_loss"),
            "tp1": raw_tech.get("suggested_take_profit_1"),
            "tp2_20pct": raw_tech.get("suggested_target_tp2_20pct"),
            "reward_risk": raw_tech.get("reward_risk_ratio"),
        }
    }


def _fallback_sentiment(blackboard: SharedBlackboard) -> Dict[str, Any]:
    """Original deterministic sentiment — used ONLY when Gemini API is unavailable."""
    news = genai_tools.dispatch_tool_call("fetch_news_and_catalysts", {"ticker": blackboard.ticker})
    social = genai_tools.dispatch_tool_call("fetch_social_sentiment", {"ticker": blackboard.ticker})
    blackboard.sentiment = {"news": news, "social": social}

    risk_flags = news.get("risk_flags", [])
    news_score = news.get("catalyst_sentiment_score", 0)
    social_heat = social.get("social_heat", "QUIET")

    if risk_flags:
        return {"stance": "TOXIC_AVOID", "sentiment_score": -60,
                "summary": f"Risk terms detected: {risk_flags}", "social_heat": social_heat}
    elif news_score > 10 or social_heat in ("WARM", "HOT"):
        return {"stance": "SUPPORTIVE", "sentiment_score": 30 + (news_score // 2),
                "summary": f"Positive catalysts. Retail interest: {social_heat}.", "social_heat": social_heat}
    else:
        return {"stance": "NEUTRAL", "sentiment_score": 0,
                "summary": "Catalyst environment quiet and neutral.", "social_heat": social_heat}


def _fallback_risk(blackboard: SharedBlackboard, account_balance: float) -> Dict[str, Any]:
    """Original deterministic risk — used ONLY when Gemini API is unavailable."""
    past_lessons = genai_tools.dispatch_tool_call("query_past_lessons", {"ticker": blackboard.ticker})
    blackboard.past_lessons = past_lessons
    port_state = genai_tools.dispatch_tool_call("get_portfolio_state", {})
    blackboard.portfolio_state = port_state

    is_owned = blackboard.ticker in port_state.get("owned_tickers", [])
    if is_owned:
        return {"approved": False, "veto_reason": f"{blackboard.ticker} ALREADY OWNED.", "shares": 0}

    entry = blackboard.technicals.get("suggested_entry")
    stop = blackboard.technicals.get("suggested_stop_loss")
    if not entry or not stop:
        return {"approved": False, "veto_reason": "Missing entry/stop levels.", "shares": 0}

    sizing = genai_tools.dispatch_tool_call("calculate_risk_sizing", {
        "entry_price": entry, "stop_loss_price": stop, "account_balance": account_balance, "risk_per_trade_pct": 0.015
    })
    blackboard.risk = sizing
    if not sizing.get("approved", False):
        return {"approved": False, "veto_reason": sizing.get("reason", "Sizing policy violation"), "shares": 0}

    return {
        "approved": True, "veto_reason": None,
        "shares": sizing.get("shares", 0),
        "dollar_risk": sizing.get("dollar_risk", 0.0),
        "capital_required": sizing.get("capital_required", 0.0),
        "portfolio_risk_pct": sizing.get("portfolio_risk_pct", 0.0),
        "summary": f"Risk APPROVED: {sizing.get('shares')} shares, ${sizing.get('dollar_risk', 0):,.2f} risk."
    }


def _fallback_supervisor_synthesis(blackboard: SharedBlackboard, tech_res, sent_res, risk_res, is_etf: bool):
    """Original deterministic verdict logic — used ONLY when Gemini API is unavailable."""
    tech_stance = tech_res.get("stance", "NEUTRAL")
    tech_score = tech_res.get("confidence_score", 0)
    sent_stance = sent_res.get("stance", "NEUTRAL")
    risk_approved = risk_res.get("approved", False)

    blackboard.entry_price = tech_res.get("levels", {}).get("entry")
    blackboard.stop_loss = tech_res.get("levels", {}).get("stop_loss")
    blackboard.take_profit_1 = tech_res.get("levels", {}).get("tp1")
    blackboard.take_profit_2 = tech_res.get("levels", {}).get("tp2_20pct")
    blackboard.confidence = tech_score

    if not risk_approved:
        blackboard.verdict = "REJECT_VETO"
        blackboard.shares = 0
        blackboard.investment_memo = f"Risk Officer VETO: {risk_res.get('veto_reason')}"
    elif sent_stance == "TOXIC_AVOID":
        blackboard.verdict = "REJECT_VETO"
        blackboard.shares = 0
        blackboard.investment_memo = f"Sentiment VETO: Severe negative catalysts ({sent_res.get('summary')})."
    elif is_etf and tech_score >= 35:
        blackboard.verdict = "CORE_ACCUMULATE"
        blackboard.shares = risk_res.get("shares", 0)
        blackboard.investment_memo = (
            f"Core ETF Accumulation: {blackboard.ticker} in prime compounding zone. "
            f"{blackboard.shares} shares. Hold long-term."
        )
    elif tech_stance in ("BULLISH", "LEAN_BULLISH") and tech_score >= 70:
        blackboard.verdict = "EXECUTE_BUY"
        blackboard.shares = risk_res.get("shares", 0)
        blackboard.approval_status = "PENDING_APPROVAL"
        blackboard.investment_memo = (
            f"[FALLBACK] High-Conviction BUY on {blackboard.ticker}. Score {tech_score}/100. "
            f"Target ~20%: ${blackboard.take_profit_2:.2f}, Stop: ${blackboard.stop_loss:.2f}. "
            f"Risk: {risk_res.get('portfolio_risk_pct')}% (${risk_res.get('dollar_risk'):,.2f}) for {blackboard.shares} shares."
        )
    else:
        blackboard.verdict = "HOLD_CASH"
        blackboard.shares = 0
        blackboard.investment_memo = (
            f"[FALLBACK] No actionable edge (Score {tech_score}/100 < 70). "
            f"Wait for clean pullback or deep value dip."
        )


# =============================================================================
# SPECIALIZED WORKER AGENTS WITH LLM REASONING
# =============================================================================

class TechnicalsWorker(GenAIAgentBase):
    """Worker 1: Market Technicals & Price Action Specialist — LLM-powered."""

    def __init__(self):
        super().__init__(
            name="TechnicalsWorker",
            role="Senior Technical Market Analyst",
            system_prompt="""You are a veteran technical swing analyst for a $25,000 portfolio.

Your job: Analyze a stock's price action and technical indicators to determine if there's a clean swing trade setup with ~20% upside potential.

AVAILABLE TOOLS: You have access to get_market_technicals which returns EMAs (20/50/200), RSI-14, ATR-14, swing setup classification, and proposed price levels.

ANALYSIS FRAMEWORK:
1. Call get_market_technicals for the ticker
2. Evaluate the trend (BULLISH_UPTREND, OVERSOLD_REVERSAL, or NEUTRAL_CONSOLIDATION)
3. Check RSI for oversold (<35) or healthy pullback (40-55) zones
4. Assess the setup type (MOMENTUM_PULLBACK vs VALUE_DIP_REVERSAL vs EXTENDED)
5. Evaluate the reward:risk ratio (want >3:1 for ~20% targets)

RESPOND WITH VALID JSON ONLY (no markdown, no code fences):
{
    "stance": "BULLISH" | "LEAN_BULLISH" | "NEUTRAL" | "BEARISH",
    "confidence_score": 0-100,
    "setup_summary": "2-3 sentence analysis explaining your reasoning",
    "levels": {
        "entry": <price>,
        "stop_loss": <price>,
        "tp1": <price>,
        "tp2_20pct": <price>,
        "reward_risk": <ratio>
    }
}""",
            tools=["get_market_technicals"],
        )

    def evaluate(self, blackboard: SharedBlackboard) -> Dict[str, Any]:
        prompt = (
            f"Analyze {blackboard.ticker} for a swing trade setup. "
            f"Market regime is {blackboard.macro_regime}. "
            f"{'This is a core ETF for long-term compounding.' if blackboard.is_etf else 'This is a single stock targeting ~20% swing profit.'}"
        )

        llm_text, tool_calls = self.reason_with_tools(prompt, ticker=blackboard.ticker)

        # Log the agent trace
        AgentTraceLogger.log(
            ticker=blackboard.ticker, agent_role=self.role,
            input_prompt=prompt, output_thought=llm_text, tool_calls=tool_calls,
        )

        if llm_text:
            # Parse LLM's structured JSON response
            try:
                # Strip markdown fences if present
                cleaned = llm_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                    cleaned = cleaned.rsplit("```", 1)[0]
                result = json.loads(cleaned)

                # Extract price from tool calls for the blackboard
                for tc in tool_calls:
                    if tc["tool"] == "get_market_technicals":
                        try:
                            raw = json.loads(tc["result_summary"]) if isinstance(tc["result_summary"], str) else tc["result_summary"]
                            blackboard.technicals = raw if isinstance(raw, dict) else {}
                        except (json.JSONDecodeError, TypeError):
                            pass

                if not blackboard.technicals and "levels" in result:
                    blackboard.technicals = {
                        "suggested_entry": result["levels"].get("entry"),
                        "suggested_stop_loss": result["levels"].get("stop_loss"),
                        "suggested_take_profit_1": result["levels"].get("tp1"),
                        "suggested_target_tp2_20pct": result["levels"].get("tp2_20pct"),
                        "reward_risk_ratio": result["levels"].get("reward_risk"),
                    }

                blackboard.current_price = result.get("levels", {}).get("entry", blackboard.current_price)
                blackboard.post_update(self.name, result.get("stance", "NEUTRAL"), result, result.get("setup_summary", ""))
                return result

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[{self.name}] Failed to parse LLM response: {e}. Using fallback.")

        # FALLBACK: deterministic scoring
        logger.info(f"[{self.name}] Using deterministic fallback for {blackboard.ticker}")
        result = _fallback_technicals(blackboard)
        blackboard.post_update(self.name, result["stance"], result, result["setup_summary"])
        return result


class SentimentWorker(GenAIAgentBase):
    """Worker 2: Sentiment, News & Catalyst Hunter — LLM-powered."""

    def __init__(self):
        super().__init__(
            name="SentimentWorker",
            role="Intelligence & Catalyst Analyst",
            system_prompt="""You are a forensic market sentiment and news intelligence analyst.

Your job: Scan headlines and retail chatter for a stock to determine if the catalyst environment supports or threatens a swing trade.

AVAILABLE TOOLS:
- fetch_news_and_catalysts: Gets recent headlines, bullish/bearish flags, and sentiment scores
- fetch_social_sentiment: Scans Reddit for retail discussion heat and mention volume

ANALYSIS FRAMEWORK:
1. Call fetch_news_and_catalysts to get headlines and risk flags
2. Call fetch_social_sentiment to gauge retail interest
3. CRITICAL: Distinguish pharmaceutical catalysts ("weight loss drug approved") from financial losses ("company reports loss"). A pharma company's weight loss drug is BULLISH, not bearish.
4. Look for toxic catalysts: DOJ investigations, FDA rejections, accounting fraud, SEC probes
5. Evaluate if sentiment supports or threatens the trade thesis

RESPOND WITH VALID JSON ONLY (no markdown, no code fences):
{
    "stance": "SUPPORTIVE" | "NEUTRAL" | "TOXIC_AVOID",
    "sentiment_score": -100 to +100,
    "summary": "2-3 sentence analysis of the catalyst environment",
    "top_headline": "Most relevant headline",
    "social_heat": "HOT" | "WARM" | "QUIET"
}""",
            tools=["fetch_news_and_catalysts", "fetch_social_sentiment"],
        )

    def evaluate(self, blackboard: SharedBlackboard) -> Dict[str, Any]:
        prompt = (
            f"Analyze the sentiment and catalyst environment for {blackboard.ticker}. "
            f"Is the news flow supportive of a swing trade, or are there toxic catalysts that should kill the trade?"
        )

        llm_text, tool_calls = self.reason_with_tools(prompt, ticker=blackboard.ticker)

        AgentTraceLogger.log(
            ticker=blackboard.ticker, agent_role=self.role,
            input_prompt=prompt, output_thought=llm_text, tool_calls=tool_calls,
        )

        if llm_text:
            try:
                cleaned = llm_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                    cleaned = cleaned.rsplit("```", 1)[0]
                result = json.loads(cleaned)

                # Populate blackboard sentiment from tool calls
                for tc in tool_calls:
                    if tc["tool"] == "fetch_news_and_catalysts":
                        try:
                            raw = json.loads(tc["result_summary"]) if isinstance(tc["result_summary"], str) else {}
                            blackboard.sentiment["news"] = raw if isinstance(raw, dict) else {}
                        except (json.JSONDecodeError, TypeError):
                            pass
                    elif tc["tool"] == "fetch_social_sentiment":
                        try:
                            raw = json.loads(tc["result_summary"]) if isinstance(tc["result_summary"], str) else {}
                            blackboard.sentiment["social"] = raw if isinstance(raw, dict) else {}
                        except (json.JSONDecodeError, TypeError):
                            pass

                blackboard.post_update(self.name, result.get("stance", "NEUTRAL"), result, result.get("summary", ""))
                return result

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[{self.name}] Failed to parse LLM response: {e}. Using fallback.")

        logger.info(f"[{self.name}] Using deterministic fallback for {blackboard.ticker}")
        result = _fallback_sentiment(blackboard)
        blackboard.post_update(self.name, result["stance"], result, result["summary"])
        return result


class RiskGovernorWorker(GenAIAgentBase):
    """Worker 3: Chief Risk Officer — LLM-powered with veto authority."""

    def __init__(self):
        super().__init__(
            name="RiskGovernorWorker",
            role="Chief Risk Officer (Veto Authority)",
            system_prompt="""You are the Chief Risk Officer with absolute veto authority. Your mandate is capital preservation.

AVAILABLE TOOLS:
- query_past_lessons: Retrieve historical trade journal for this ticker (episodic memory)
- get_portfolio_state: Check active positions to prevent duplicates
- calculate_risk_sizing: Compute position size with 1-1.5% max account risk

ANALYSIS FRAMEWORK:
1. Call get_portfolio_state — if the ticker is ALREADY OWNED, VETO immediately
2. Call query_past_lessons — review past trades on this ticker. If we lost money before on this name, note the pattern
3. Call calculate_risk_sizing with the entry and stop loss prices
4. Enforce: stop loss distance must be 1.5%-7.5% of entry price
5. Enforce: max 1.5% of $25,000 account at risk per trade ($375 max dollar risk)
6. Enforce: reward-to-risk ratio should be >3:1 for swing trades

RESPOND WITH VALID JSON ONLY (no markdown, no code fences):
{
    "approved": true | false,
    "veto_reason": null | "reason string if vetoed",
    "shares": <int>,
    "dollar_risk": <float>,
    "capital_required": <float>,
    "portfolio_risk_pct": <float>,
    "summary": "2-3 sentence risk assessment including past lessons if relevant"
}""",
            tools=["query_past_lessons", "get_portfolio_state", "calculate_risk_sizing"],
        )

    def evaluate(self, blackboard: SharedBlackboard, account_balance: float = 25000.0) -> Dict[str, Any]:
        entry = blackboard.technicals.get("suggested_entry", blackboard.current_price)
        stop = blackboard.technicals.get("suggested_stop_loss", 0)

        prompt = (
            f"Evaluate the risk for a swing trade on {blackboard.ticker}. "
            f"Proposed entry: ${entry:.2f}, Proposed stop loss: ${stop:.2f}. "
            f"Account balance: ${account_balance:,.2f}. "
            f"Check if this ticker is already owned, review past trade history, and calculate proper sizing."
        )

        llm_text, tool_calls = self.reason_with_tools(prompt, ticker=blackboard.ticker)

        AgentTraceLogger.log(
            ticker=blackboard.ticker, agent_role=self.role,
            input_prompt=prompt, output_thought=llm_text, tool_calls=tool_calls,
        )

        if llm_text:
            try:
                cleaned = llm_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                    cleaned = cleaned.rsplit("```", 1)[0]
                result = json.loads(cleaned)

                # Populate blackboard from tool calls
                for tc in tool_calls:
                    if tc["tool"] == "query_past_lessons":
                        try:
                            raw = json.loads(tc["result_summary"]) if isinstance(tc["result_summary"], str) else {}
                            blackboard.past_lessons = raw if isinstance(raw, dict) else {}
                        except (json.JSONDecodeError, TypeError):
                            pass
                    elif tc["tool"] == "get_portfolio_state":
                        try:
                            raw = json.loads(tc["result_summary"]) if isinstance(tc["result_summary"], str) else {}
                            blackboard.portfolio_state = raw if isinstance(raw, dict) else {}
                        except (json.JSONDecodeError, TypeError):
                            pass

                blackboard.risk = result
                blackboard.post_update(
                    self.name,
                    "APPROVED" if result.get("approved") else "VETO",
                    result,
                    result.get("summary", result.get("veto_reason", ""))
                )
                return result

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[{self.name}] Failed to parse LLM response: {e}. Using fallback.")

        logger.info(f"[{self.name}] Using deterministic fallback for {blackboard.ticker}")
        result = _fallback_risk(blackboard, account_balance)
        blackboard.risk = result
        blackboard.post_update(
            self.name,
            "APPROVED" if result.get("approved") else "VETO",
            result,
            result.get("summary", result.get("veto_reason", ""))
        )
        return result


# =============================================================================
# EXECUTIVE SUPERVISOR AGENT (ROUTER & SYNTHESIZER)
# =============================================================================

class ExecutiveSupervisorAgent(GenAIAgentBase):
    """
    Supervisor Agent (Chief Investment Officer).
    Orchestrates workers, then uses LLM to synthesize a nuanced investment thesis
    from their reports — or falls back to deterministic if/elif when API is down.
    """

    def __init__(self):
        super().__init__(
            name="ExecutiveSupervisor",
            role="Chief Investment Officer (Executive Supervisor)",
            system_prompt="""You are the Executive Supervisor and Chief Investment Officer for a $25,000 swing trading portfolio.

You have received analysis from three specialized agents:
1. TechnicalsWorker — price action, trend, EMAs, RSI, setup classification
2. SentimentWorker — news catalysts, risk flags, retail chatter
3. RiskGovernorWorker — position sizing, portfolio risk, past trade history

Your job: Synthesize their reports into a final investment decision.

DECISION RULES:
- If RiskGovernor vetoed → REJECT (no override)
- If Sentiment is TOXIC_AVOID → REJECT (serious catalyst risk)
- For ETFs: accumulate on any pullback (lower threshold)
- For stocks: need strong technicals (confidence ≥70) + supportive sentiment for BUY
- Always target ~20% swing profit with strict risk management
- Consider past lessons if we've traded this name before

RESPOND WITH VALID JSON ONLY (no markdown, no code fences):
{
    "verdict": "EXECUTE_BUY" | "CORE_ACCUMULATE" | "HOLD_CASH" | "REJECT_VETO",
    "confidence": 0-100,
    "shares": <int>,
    "investment_memo": "3-5 sentence authoritative investment thesis explaining the decision, key drivers, and risk factors",
    "risk_factors": ["factor1", "factor2"]
}""",
        )
        self.tech_worker = TechnicalsWorker()
        self.sent_worker = SentimentWorker()
        self.risk_worker = RiskGovernorWorker()

    def convene_committee(
        self,
        ticker: str,
        regime: str = "BULL",
        account_balance: float = 25000.0,
        is_etf: bool = False,
    ) -> SharedBlackboard:
        """Orchestrate end-to-end multi-agent deliberation for a ticker."""
        ticker = ticker.strip().upper()
        blackboard = SharedBlackboard(ticker=ticker, macro_regime=regime, is_etf=is_etf)
        using_llm = self._llm_available

        logger.info(
            f"[Supervisor] Convening Committee for {ticker} "
            f"(Regime={regime}, Engine={'GEMINI LLM' if using_llm else 'DETERMINISTIC FALLBACK'})..."
        )

        # Step 1: Circuit Breaker Check (always deterministic — safety critical)
        if regime in ("BEAR", "PANIC_CRASH"):
            blackboard.verdict = "HOLD_CASH"
            blackboard.investment_memo = (
                f"CIO Macro Circuit Breaker: SPY regime is {regime}. "
                f"Policy mandates 100% cash preservation. All buy actions halted."
            )
            blackboard.post_update(self.name, "CIRCUIT_BREAKER", {}, blackboard.investment_memo)
            return blackboard

        # Step 2: Route to Workers (each worker tries LLM, falls back if needed)
        tech_res = self.tech_worker.evaluate(blackboard)
        sent_res = self.sent_worker.evaluate(blackboard)
        risk_res = self.risk_worker.evaluate(blackboard, account_balance=account_balance)

        # Step 3: Executive Synthesis
        if using_llm:
            blackboard.llm_mode = "LLM"
            self._llm_synthesis(blackboard, tech_res, sent_res, risk_res, is_etf, account_balance)
        else:
            blackboard.llm_mode = "FALLBACK"
            _fallback_supervisor_synthesis(blackboard, tech_res, sent_res, risk_res, is_etf)

        # Step 4: Register pending approval if BUY
        if blackboard.verdict == "EXECUTE_BUY":
            blackboard.approval_status = "PENDING_APPROVAL"
            approval_id = self._register_pending_approval(blackboard)
            blackboard.human_approval_id = approval_id

        blackboard.post_update(self.name, blackboard.verdict, asdict(blackboard), blackboard.investment_memo)
        self._persist_deliberation_trace(blackboard)
        return blackboard

    def _llm_synthesis(self, blackboard: SharedBlackboard,
                       tech_res: Dict, sent_res: Dict, risk_res: Dict,
                       is_etf: bool, account_balance: float):
        """Use Gemini to synthesize worker reports into an investment thesis."""
        # Build a rich prompt with all worker outputs for the supervisor LLM
        synthesis_prompt = f"""Synthesize the following multi-agent analysis into a final investment decision for {blackboard.ticker}.

TICKER: {blackboard.ticker}
TYPE: {'Core ETF (long-term compounding)' if is_etf else 'Single stock (swing trade, ~20% target)'}
MARKET REGIME: {blackboard.macro_regime}
ACCOUNT BALANCE: ${account_balance:,.2f}

--- TECHNICALS WORKER REPORT ---
{json.dumps(tech_res, indent=2, default=str)}

--- SENTIMENT WORKER REPORT ---
{json.dumps(sent_res, indent=2, default=str)}

--- RISK GOVERNOR REPORT ---
{json.dumps(risk_res, indent=2, default=str)}

Make your decision. Remember: Risk Governor veto is absolute. Toxic sentiment is absolute. For stocks, need confidence >= 70 for a BUY."""

        try:
            # Supervisor doesn't need tools — it synthesizes worker outputs
            response = self.gemini_client.models.generate_content(
                model=self.model,
                contents=synthesis_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=LLM_TEMPERATURE,
                ),
            )

            if response and response.text:
                llm_text = response.text.strip()

                AgentTraceLogger.log(
                    ticker=blackboard.ticker, agent_role=self.role,
                    input_prompt=synthesis_prompt[:2000], output_thought=llm_text,
                )

                # Parse the structured response
                cleaned = llm_text
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                    cleaned = cleaned.rsplit("```", 1)[0]

                decision = json.loads(cleaned)

                blackboard.verdict = decision.get("verdict", "HOLD_CASH")
                blackboard.confidence = decision.get("confidence", 0)
                blackboard.shares = decision.get("shares", 0)
                blackboard.investment_memo = decision.get("investment_memo", "")

                # Ensure levels are set from technicals
                blackboard.entry_price = tech_res.get("levels", {}).get("entry", blackboard.current_price)
                blackboard.stop_loss = tech_res.get("levels", {}).get("stop_loss")
                blackboard.take_profit_1 = tech_res.get("levels", {}).get("tp1")
                blackboard.take_profit_2 = tech_res.get("levels", {}).get("tp2_20pct")

                # If LLM says BUY but risk vetoed, enforce the veto (safety override)
                if not risk_res.get("approved", False) and blackboard.verdict == "EXECUTE_BUY":
                    blackboard.verdict = "REJECT_VETO"
                    blackboard.shares = 0
                    blackboard.investment_memo = f"SAFETY OVERRIDE: Risk Governor vetoed. {risk_res.get('veto_reason')}"

                # Ensure shares match risk sizing
                if blackboard.verdict == "EXECUTE_BUY" and risk_res.get("shares"):
                    blackboard.shares = risk_res["shares"]

                logger.info(f"[Supervisor] LLM synthesis complete: {blackboard.verdict} (confidence={blackboard.confidence})")
                return

        except Exception as e:
            logger.warning(f"[Supervisor] LLM synthesis failed: {e}. Falling back to deterministic.")

        # Fallback if LLM synthesis fails
        blackboard.llm_mode = "FALLBACK"
        _fallback_supervisor_synthesis(blackboard, tech_res, sent_res, risk_res, is_etf)

    def _register_pending_approval(self, bb: SharedBlackboard) -> int:
        """Register trade recommendation awaiting human confirmation."""
        conn = genai_tools.get_db_connection()
        cur = conn.cursor()
        now_str = datetime.now().isoformat()
        cur.execute("""
        INSERT INTO pending_approvals (
            ticker, verdict, current_price, shares, stop_loss, take_profit_1, take_profit_2,
            confidence, thesis, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
        """, (
            bb.ticker, bb.verdict, bb.current_price, bb.shares, bb.stop_loss,
            bb.take_profit_1, bb.take_profit_2, bb.confidence, bb.investment_memo, now_str
        ))
        approval_id = cur.lastrowid
        conn.commit()
        conn.close()
        return approval_id

    def _persist_deliberation_trace(self, bb: SharedBlackboard):
        """Save full multi-agent trace to SQLite memory."""
        try:
            conn = genai_tools.get_db_connection()
            cur = conn.cursor()
            now_str = datetime.now().isoformat()
            dialogue_json = json.dumps(bb.dialogue)
            cur.execute("""
            INSERT INTO committee_signals (
                timestamp, ticker, regime, current_price, verdict, shares,
                entry_price, stop_loss, take_profit_1, take_profit_2,
                technical_score, technical_stance, sentiment_score, sentiment_stance,
                risk_approved, executive_summary, deliberation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_str, bb.ticker, bb.macro_regime, bb.current_price, bb.verdict, bb.shares,
                bb.entry_price, bb.stop_loss, bb.take_profit_1, bb.take_profit_2,
                bb.confidence, bb.technicals.get("trend", "NEUTRAL"),
                bb.sentiment.get("news", {}).get("catalyst_sentiment_score", 0),
                bb.sentiment.get("news", {}).get("sentiment_stance", "NEUTRAL"),
                1 if bb.risk.get("approved") else 0,
                bb.investment_memo,
                dialogue_json
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to persist deliberation trace: {e}")


# Convenience Singleton for external modules
supervisor = ExecutiveSupervisorAgent()


def handle_hitl_action(approval_id: int, action: str, user_id: int = 0) -> Dict[str, Any]:
    """
    Handle human approval, rejection, or trace request for a pending trade.
    State machine: PENDING -> APPROVED or REJECTED.
    """
    conn = genai_tools.get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return {"status": "ERROR", "message": f"Approval request #{approval_id} not found."}

    req = dict(row)
    now_str = datetime.now().isoformat()

    if action == "APPROVE":
        if req["status"] == "APPROVED":
            conn.close()
            return {"status": "ALREADY_APPROVED", "message": f"Setup for {req['ticker']} was already approved."}

        cur.execute("""
        INSERT INTO committee_positions (
            ticker, entry_time, entry_price, shares, stop_loss,
            take_profit_1, take_profit_2, status, trailing_stop_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', 0)
        """, (
            req["ticker"], now_str, req["current_price"], req["shares"],
            req["stop_loss"], req["take_profit_1"], req["take_profit_2"]
        ))
        pos_id = cur.lastrowid

        cur.execute("""
        UPDATE pending_approvals
        SET status = 'APPROVED', actioned_at = ?, actioned_by = ?
        WHERE id = ?
        """, (now_str, str(user_id), approval_id))

        cur.execute("""
        INSERT INTO committee_journal (
            position_id, ticker, outcome, pnl_pct, reflection_notes, created_at
        ) VALUES (?, ?, 'OPEN', 0.0, ?, ?)
        """, (
            pos_id, req["ticker"],
            f"Human Approved via Telegram HITL. Target +20%: ${req['take_profit_2']:.2f}, Stop: ${req['stop_loss']:.2f}",
            now_str
        ))
        conn.commit()
        conn.close()
        return {
            "status": "APPROVED",
            "position_id": pos_id,
            "ticker": req["ticker"],
            "shares": req["shares"],
            "entry_price": req["current_price"],
            "target_20pct": req["take_profit_2"],
            "stop_loss": req["stop_loss"],
        }

    elif action == "REJECT":
        cur.execute("""
        UPDATE pending_approvals
        SET status = 'REJECTED', actioned_at = ?, actioned_by = ?
        WHERE id = ?
        """, (now_str, str(user_id), approval_id))

        cur.execute("""
        INSERT INTO committee_journal (
            ticker, outcome, pnl_pct, reflection_notes, created_at
        ) VALUES (?, 'VETO', 0.0, 'Human passed/rejected recommendation via Telegram HITL.', ?)
        """, (req["ticker"], now_str))
        conn.commit()
        conn.close()
        return {
            "status": "REJECTED",
            "ticker": req["ticker"],
            "message": "Recommendation rejected. Capital preserved.",
        }

    elif action == "GET_TRACE":
        # For traces, also fetch agent_trace_logs
        cur.execute("""
        SELECT agent_role, input_prompt, output_thought, tool_calls_json, created_at
        FROM agent_trace_logs
        WHERE ticker = ?
        ORDER BY id DESC
        LIMIT 10
        """, (req["ticker"],))
        trace_rows = cur.fetchall()
        traces = [dict(r) for r in trace_rows] if trace_rows else []
        conn.close()
        return {
            "status": "SUCCESS",
            "ticker": req["ticker"],
            "thesis": req["thesis"],
            "confidence": req["confidence"],
            "created_at": req["created_at"],
            "agent_traces": traces,
        }

    conn.close()
    return {"status": "ERROR", "message": f"Unknown action: {action}"}
