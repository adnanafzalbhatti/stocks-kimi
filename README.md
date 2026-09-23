# Stock Trading AI Agent Committee 🤖📈

A **true AI agent system** that uses Google's Gemini LLM as the reasoning engine for autonomous stock trading analysis. The LLM decides which tools to call, interprets real market data, and produces nuanced investment theses — not a wrapper or rule-based script with AI branding.

## What Makes This a True AI Agent?

| AI Agent Criterion | Implementation |
|---|---|
| **LLM as Reasoning Engine** | Gemini 3.5 Flash-Lite decides which tools to invoke and how to interpret results |
| **Autonomous Tool Selection** | The LLM picks tools from a registry — not hardcoded Python if/else |
| **ReAct Loop** | Multi-round observe → think → act → observe cycle until the LLM naturally terminates |
| **Multi-Agent System** | 4 specialized agents: Technicals, Sentiment, Risk Governor, Executive Supervisor |
| **Supervisor-Worker Pattern** | Executive Supervisor orchestrates workers, synthesizes reports, renders final verdict |
| **Human-in-the-Loop** | Telegram approval buttons before any trade executes — LLM proposes, human disposes |
| **Graceful Degradation** | Deterministic fallback scoring when API is unavailable — 100% uptime guarantee |
| **Agent Trace Logging** | Every LLM prompt, tool call, and response logged to SQLite for full audit trail |
| **Shared State (Blackboard)** | All agents read/write to a central `SharedBlackboard` dataclass |
| **Safety Overrides** | Risk Governor has veto authority — even if the LLM says BUY, risk rules are enforced |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              Executive Supervisor (LLM)               │
│         Synthesizes all worker reports into           │
│         a final investment thesis + verdict           │
├──────────┬──────────────────┬─────────────────────────┤
│          │                  │                         │
│  ┌───────▼───────┐  ┌──────▼───────┐  ┌─────────────▼──┐
│  │  Technicals   │  │  Sentiment   │  │ Risk Governor  │
│  │   Worker      │  │   Worker     │  │   Worker       │
│  │  (LLM+Tools)  │  │  (LLM+Tools) │  │  (LLM+Tools)  │
│  └───────┬───────┘  └──────┬───────┘  └────────┬───────┘
│          │                  │                   │
│  ┌───────▼───────────────────▼───────────────────▼───────┐
│  │              MCP Tool Registry (7 Tools)              │
│  │  get_market_technicals  │  fetch_news_and_catalysts   │
│  │  fetch_social_sentiment │  get_portfolio_state        │
│  │  query_past_lessons     │  calculate_risk_sizing      │
│  │  check_active_position                                │
│  └───────────────────────────────────────────────────────┘
│                              │
│  ┌───────────────────────────▼───────────────────────────┐
│  │        SharedBlackboard (Central State)                │
│  │  technicals, sentiment, risk, verdict, memo           │
│  └───────────────────────────────────────────────────────┘
│                              │
│  ┌───────────────────────────▼───────────────────────────┐
│  │     Human-in-the-Loop (Telegram Approval Buttons)     │
│  │              ✅ APPROVE    ❌ REJECT                   │
│  └───────────────────────────────────────────────────────┘
└──────────────────────────────────────────────────────────┘
```

## How It Works

1. **Scheduled scans** run at 10:15, 12:15, 14:15 ET during market hours
2. Each scan evaluates 86+ tickers through the **LLM-powered committee**
3. The LLM autonomously selects tools to gather technicals, sentiment, and risk data
4. Workers produce structured reports; the Supervisor synthesizes a final verdict
5. If verdict = BUY, a **Telegram approval request** is sent with trade details
6. Human approves/rejects; only approved trades are logged as positions
7. Full audit trail in SQLite — every LLM decision is traceable

## Sample LLM-Generated Investment Memo

> "While the Risk Governor approved the trade and risk parameters are fully compliant with a solid 3.76:1 reward-to-risk ratio, the Technicals Worker's confidence score sits at 68, which falls just short of our strict 70 threshold for single-stock swing buys. Combined with a neutral sentiment backdrop offering no high-impact catalysts, we prefer to preserve capital and await a stronger confirmation signal or deeper pullback."

This is **real LLM reasoning** — not a template. The model interprets market data, weighs risk parameters, and makes nuanced judgment calls.

## Tech Stack

- **LLM**: Google Gemini 3.5 Flash-Lite (via `google-genai` SDK)
- **Framework**: Custom multi-agent system with ReAct loop and MCP tool registry
- **Bot**: python-telegram-bot (async, webhook-ready)
- **Data**: yfinance, Reddit API, Twitter/X API, Google News RSS
- **Storage**: SQLite (positions, signals, journal, agent traces)
- **Deployment**: Docker + Docker Compose (cloud-ready)

## Quick Start

```bash
# Clone
git clone https://github.com/adnanafzalbhatti/stocks-kimi.git
cd stocks-kimi

# Configure
cp .env.example .env
# Edit .env with your API keys (Gemini, Telegram, Reddit, Twitter)

# Install dependencies
pip install -r requirements.txt

# Run live bot
python stock_agent_v3.py live

# Or run a single committee analysis
python test_llm_committee.py
```

### Docker

```bash
docker compose up -d
```

## Key Files

| File | Description |
|------|-------------|
| `genai_agents.py` | Multi-agent system: ReAct loop, workers, supervisor, fallback |
| `genai_tools.py` | MCP tool registry: 7 tools for market data, sentiment, risk |
| `stock_agent_v3.py` | Telegram bot, scheduler, backtester, position management |
| `factor_engine.py` | Technical analysis factor computation engine |
| `backtester.py` | Historical backtesting framework |

## Backtest Results (86 tickers, 2024–2026)

| Metric | Value |
|--------|-------|
| Total Trades | 2,901 |
| Win Rate | 42% |
| Profit Factor | 1.35 |
| Net PnL | +2,359% |

## License

MIT

## Author

**Adnan Afzal Bhatti** — [LinkedIn](https://www.linkedin.com/in/adnanafzalbhatti) · [GitHub](https://github.com/adnanafzalbhatti)
