# Telegram signals — Stock Agent v3.2 (LONG ONLY)

## What you get
- **BUY-only** alerts on the **full liquid universe (~90 names)** by default
- Optional core-only mode: set `USE_APPROVED_ONLY=1` in `.env` or run with `--approved`
- Each alert includes: entry, stop, TP1, TP2, share size, confidence, why, action plan
- Buttons: **Take** / **Skip** (for your own tracking)

## How often?
| Item | Value |
|------|--------|
| Scans per trading day | **4** |
| Clock times | **10:15, 11:30, 12:45, 14:15 US/Eastern** |
| Session window | **10:00–15:30 ET**, Mon–Fri (no holidays) |
| Same ticker re-alert | At most once every **36 hours** |
| Min confidence | **85** (full universe + tight gates; override with `MIN_CONFIDENCE=`) |
| Extra gates | Full EMA stack required · SPY bull only · tighter pullback/RSI/ADX/stop |
| Weekends / holidays | **No scans** |
| Heartbeat | After **every** scan you get a short “scan done” message (even if 0 buys) |

**More names = more choices, not guaranteed daily BUYs.** Quality can be lower than the old 20-name / conf-78 mode.  
**If you get nothing for days**, first check that `python stock_agent_v3.py live` is still running — the bot does **not** run in the cloud by itself.

### Why Telegram goes quiet
1. Live process stopped (PC sleep / closed terminal / crash) → no scans at all  
2. When it *is* running but no setups pass filters → heartbeat only, no BUY cards  
3. Still long-only + SPY regime filter (no new longs in bear regime)

## Starting from when?
Signals start **as soon as you run live mode** and the next scan slot hits:

```bash
python stock_agent_v3.py live
```

- If the US market is **open** when you start → an **initial scan runs immediately**
- If the market is **closed** → bot stays online and waits for the **next** 10:15 / 12:15 / 14:15 ET slot
- Check the next slot anytime:

```bash
python stock_agent_v3.py schedule
```

## One-time setup

### 1. BotFather
1. In Telegram, open **@BotFather**
2. `/newbot` → copy the **token**

### 2. Your user id
1. Open **@userinfobot** (or similar) and note your numeric **user id**

### 3. `.env` in this folder
```env
TELEGRAM_BOT_TOKEN=123456:ABC-your-token
ADMIN_IDS=123456789
ACCOUNT_BALANCE=25000
```

`ACCOUNT_BALANCE` is only used to suggest share count (1% risk, 10% max position).

### 4. Install deps (if needed)
```bash
pip install python-telegram-bot python-dotenv yfinance pandas numpy scipy
```

### 5. Subscribe
1. Open your bot in Telegram
2. Send **`/start`**
3. Start the agent:

```bash
python stock_agent_v3.py live
```

Keep that terminal/PC running (or use Task Scheduler / a small VPS later).

## Useful commands (in Telegram)
| Command | Action |
|---------|--------|
| `/start` | Subscribe + show schedule |
| `/scan` | Force a scan now |
| `/watchlist` | Full scanned ticker list |
| `/status` | Next scan + last scan |
| `/recap` | EOD / open-book scorecard (also auto ~16:05 ET) |
| `/risk 25000 225.5 218` | Position size helper |

## Local commands
```bash
python stock_agent_v3.py --list          # approved list + schedule
python stock_agent_v3.py schedule       # when is next scan
python stock_agent_v3.py scan           # one-off scan in terminal
python stock_agent_v3.py backtest       # approved list, conf≥78
python stock_agent_v3.py live           # Telegram loop
```

## Notes
- Not financial advice. Past backtests ≠ future results.
- Long only — suitable for cash accounts (no shorting).
- Bear SPY regime → no new buy alerts.
- To temporarily use all 62 names: `python stock_agent_v3.py scan --all-tickers`
