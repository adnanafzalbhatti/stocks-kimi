"""
FACTOR ENGINE — non-price conviction scoring for stock_agent_v3.

Blends five evidence buckets into one 0-100 score used to gate/rank technical
BUY setups:

    analyst   — sell-side consensus, target upside, recent upgrades/downgrades
    growth    — revenue / earnings growth, forward vs trailing PE
    quality   — margins, ROE, FCF, leverage
    ownership — institutional + insider holding ("who is accumulating")
    strength  — relative strength vs SPY (money actually flowing in)
    social    — OPTIONAL reddit mention pressure (off by default, tiny weight)

IMPORTANT: all data here is CURRENT-state (yfinance has no point-in-time
fundamentals). Using it inside a historical backtest creates look-ahead bias,
so the agent only applies it to LIVE scans.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).with_name("factor_cache.json")
CACHE_TTL_HOURS = float(os.getenv("FACTOR_CACHE_TTL_HOURS", "12"))

# Weights (normalised at runtime; social only counts when enabled)
W_ANALYST = float(os.getenv("W_ANALYST", "22"))
W_GROWTH = float(os.getenv("W_GROWTH", "22"))
W_QUALITY = float(os.getenv("W_QUALITY", "18"))
W_OWNERSHIP = float(os.getenv("W_OWNERSHIP", "10"))
W_STRENGTH = float(os.getenv("W_STRENGTH", "20"))
W_SOCIAL = float(os.getenv("W_SOCIAL", "8"))

# Do not buy into an earnings print (gap risk kills fixed stops)
EARNINGS_BLACKOUT_DAYS = int(os.getenv("EARNINGS_BLACKOUT_DAYS", "5"))

ENABLE_SOCIAL = os.getenv("ENABLE_SOCIAL", "0").strip().lower() in ("1", "true", "yes")
SOCIAL_SUBREDDITS = [
    s.strip()
    for s in os.getenv(
        "SOCIAL_SUBREDDITS",
        "TheRaceTo10Million,stocks,wallstreetbets,investing,StockMarket",
    ).split(",")
    if s.strip()
]
SOCIAL_USER_AGENT = os.getenv("SOCIAL_USER_AGENT", "stock-agent-factor-engine/1.0")

NEUTRAL = 50.0

# Words that look like tickers but are noise in reddit text
_SOCIAL_STOPWORDS = {
    "A", "I", "IT", "IS", "BE", "DD", "CEO", "CFO", "USA", "ATH", "YOLO", "FD",
    "IMO", "EPS", "IPO", "ETF", "AI", "US", "EV", "PT", "TA", "PE", "RH", "WSB",
    "ITM", "OTM", "EOD", "AH", "PM", "OP", "TL", "DR", "IV", "ER", "GDP", "FED",
    "CPI", "SEC", "IRS", "NYSE", "ALL", "AND", "FOR", "THE", "YOU", "BUY", "SELL",
}


# =============================================================================
# SCORE CONTAINER
# =============================================================================

@dataclass
class FactorScore:
    ticker: str
    score: float = NEUTRAL
    grade: str = "C"
    analyst: float = NEUTRAL
    growth: float = NEUTRAL
    quality: float = NEUTRAL
    ownership: float = NEUTRAL
    strength: float = NEUTRAL
    social: float = NEUTRAL
    days_to_earnings: Optional[int] = None
    earnings_blackout: bool = False
    target_upside_pct: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    fetched_at: str = ""
    partial: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        parts = [
            f"analyst {self.analyst:.0f}",
            f"growth {self.growth:.0f}",
            f"quality {self.quality:.0f}",
            f"owners {self.ownership:.0f}",
            f"RS {self.strength:.0f}",
        ]
        if ENABLE_SOCIAL:
            parts.append(f"social {self.social:.0f}")
        return " | ".join(parts)


def _grade(score: float) -> str:
    if score >= 80:
        return "A+"
    if score >= 70:
        return "A"
    if score >= 60:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


def _num(value) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# =============================================================================
# CACHE
# =============================================================================

def _load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    except Exception as exc:
        logger.warning("factor cache write failed: %s", exc)


def _cache_fresh(entry: dict) -> bool:
    try:
        ts = datetime.fromisoformat(entry.get("fetched_at", ""))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
    return age_h < CACHE_TTL_HOURS


# =============================================================================
# BUCKET SCORERS
# =============================================================================

def score_analyst(info: dict, upgrades: Optional[pd.DataFrame], price: Optional[float],
                  notes: List[str]) -> tuple[float, Optional[float]]:
    """Sell-side consensus + target upside + recent rating actions."""
    pts: List[float] = []
    upside = None

    # recommendationMean: 1 = strong buy ... 5 = sell
    rec = _num(info.get("recommendationMean"))
    n_analysts = _num(info.get("numberOfAnalystOpinions")) or 0
    if rec is not None and n_analysts >= 3:
        pts.append(_clamp((3.0 - rec) / 1.5 * 50.0 + 50.0))
        notes.append(f"Analyst consensus {rec:.2f} (n={int(n_analysts)})")
    elif rec is not None:
        pts.append(NEUTRAL)

    target = _num(info.get("targetMeanPrice"))
    if target and price and price > 0:
        upside = (target / price - 1.0) * 100.0
        # 0% upside -> 45, +20% -> 85, negative -> penalised
        pts.append(_clamp(45.0 + upside * 2.0))
        notes.append(f"Mean target ${target:,.2f} ({upside:+.1f}%)")

    if upgrades is not None and not upgrades.empty:
        try:
            recent = upgrades.tail(12)
            grades = recent.get("ToGrade")
            actions = recent.get("Action")
            ups = downs = 0
            if actions is not None:
                ups = int((actions.astype(str).str.lower() == "up").sum())
                downs = int((actions.astype(str).str.lower() == "down").sum())
            elif grades is not None:
                g = grades.astype(str).str.lower()
                ups = int(g.str.contains("buy|outperform|overweight").sum())
                downs = int(g.str.contains("sell|underperform|underweight").sum())
            if ups or downs:
                pts.append(_clamp(50.0 + (ups - downs) * 8.0))
                notes.append(f"Recent rating actions: {ups} up / {downs} down")
        except Exception:
            pass

    return (float(np.mean(pts)) if pts else NEUTRAL), upside


def score_growth(info: dict, notes: List[str]) -> float:
    """Revenue / earnings growth and whether forward earnings are improving."""
    pts: List[float] = []

    rev_g = _num(info.get("revenueGrowth"))
    if rev_g is not None:
        pts.append(_clamp(50.0 + rev_g * 200.0))  # +25% yoy -> 100
        notes.append(f"Revenue growth {rev_g * 100:+.1f}% yoy")

    eps_g = _num(info.get("earningsGrowth"))
    if eps_g is None:
        eps_g = _num(info.get("earningsQuarterlyGrowth"))
    if eps_g is not None:
        pts.append(_clamp(50.0 + eps_g * 100.0))  # +50% -> 100
        notes.append(f"Earnings growth {eps_g * 100:+.1f}%")

    fwd_pe = _num(info.get("forwardPE"))
    trail_pe = _num(info.get("trailingPE"))
    if fwd_pe and trail_pe and fwd_pe > 0 and trail_pe > 0:
        # forward cheaper than trailing => earnings expected to rise
        ratio = trail_pe / fwd_pe
        pts.append(_clamp(50.0 + (ratio - 1.0) * 60.0))
        if ratio > 1.05:
            notes.append("Forward PE below trailing (earnings expanding)")

    peg = _num(info.get("pegRatio"))
    if peg and 0 < peg < 10:
        pts.append(_clamp(100.0 - (peg - 1.0) * 40.0))

    return float(np.mean(pts)) if pts else NEUTRAL


def score_quality(info: dict, notes: List[str]) -> float:
    """Margins, returns, cash generation, leverage."""
    pts: List[float] = []

    margin = _num(info.get("profitMargins"))
    if margin is not None:
        pts.append(_clamp(40.0 + margin * 250.0))  # 20% margin -> 90
        notes.append(f"Net margin {margin * 100:.1f}%")

    roe = _num(info.get("returnOnEquity"))
    if roe is not None:
        pts.append(_clamp(40.0 + roe * 200.0))

    fcf = _num(info.get("freeCashflow"))
    mcap = _num(info.get("marketCap"))
    if fcf is not None and mcap and mcap > 0:
        fcf_yield = fcf / mcap
        pts.append(_clamp(45.0 + fcf_yield * 600.0))
        if fcf_yield > 0.03:
            notes.append(f"FCF yield {fcf_yield * 100:.1f}%")
        elif fcf < 0:
            notes.append("Negative free cash flow")

    d2e = _num(info.get("debtToEquity"))
    if d2e is not None:
        pts.append(_clamp(90.0 - d2e / 3.0))  # 120 D/E -> 50
        if d2e > 200:
            notes.append(f"High leverage (D/E {d2e:.0f})")

    return float(np.mean(pts)) if pts else NEUTRAL


def score_ownership(info: dict, notes: List[str]) -> float:
    """Institutional conviction + insider alignment."""
    pts: List[float] = []

    inst = _num(info.get("heldPercentInstitutions"))
    if inst is not None:
        inst = min(inst, 1.0)
        pts.append(_clamp(30.0 + inst * 70.0))
        notes.append(f"Institutions hold {inst * 100:.0f}%")

    insiders = _num(info.get("heldPercentInsiders"))
    if insiders is not None:
        insiders = min(insiders, 1.0)
        pts.append(_clamp(50.0 + insiders * 150.0))

    short_pct = _num(info.get("shortPercentOfFloat"))
    if short_pct is not None:
        pts.append(_clamp(70.0 - short_pct * 200.0))
        if short_pct > 0.08:
            notes.append(f"Heavy short interest {short_pct * 100:.1f}% of float")

    return float(np.mean(pts)) if pts else NEUTRAL


def score_strength(hist: Optional[pd.DataFrame], spy_hist: Optional[pd.DataFrame],
                   notes: List[str]) -> float:
    """Relative strength vs SPY over ~3m and ~6m of daily closes."""
    if hist is None or hist.empty or spy_hist is None or spy_hist.empty:
        return NEUTRAL

    def _ret(frame: pd.DataFrame, lookback: int) -> Optional[float]:
        closes = frame["Close"].dropna() if "Close" in frame else frame.iloc[:, 0].dropna()
        if len(closes) <= lookback:
            return None
        return float(closes.iloc[-1] / closes.iloc[-lookback - 1] - 1.0)

    pts: List[float] = []
    for lookback, label in ((63, "3m"), (126, "6m")):
        r_stock = _ret(hist, lookback)
        r_spy = _ret(spy_hist, lookback)
        if r_stock is None or r_spy is None:
            continue
        rel = (r_stock - r_spy) * 100.0
        pts.append(_clamp(50.0 + rel * 1.6))
        if abs(rel) >= 5:
            notes.append(f"{label} vs SPY {rel:+.1f}pp")

    return float(np.mean(pts)) if pts else NEUTRAL


def _days_to_earnings(tk: yf.Ticker) -> Optional[int]:
    today = datetime.now().date()
    candidates: List[datetime] = []

    try:
        cal = tk.calendar
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("earningsDate") or []
            if not isinstance(raw, (list, tuple)):
                raw = [raw]
            for d in raw:
                if isinstance(d, datetime):
                    candidates.append(d)
                elif hasattr(d, "year"):
                    candidates.append(datetime(d.year, d.month, d.day))
        elif isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
            for d in cal.loc["Earnings Date"].tolist():
                ts = pd.to_datetime(d, errors="coerce")
                if pd.notna(ts):
                    candidates.append(ts.to_pydatetime())
    except Exception:
        pass

    if not candidates:
        try:
            ed = tk.get_earnings_dates(limit=8)
            if ed is not None and not ed.empty:
                for idx in ed.index:
                    ts = pd.to_datetime(idx, errors="coerce")
                    if pd.notna(ts):
                        candidates.append(ts.to_pydatetime())
        except Exception:
            pass

    future = []
    for c in candidates:
        d = c.date() if hasattr(c, "date") else c
        delta = (d - today).days
        if -1 <= delta <= 400:
            future.append(delta)
    return min(future) if future else None


# =============================================================================
# SOCIAL (optional, deliberately low weight)
# =============================================================================

def fetch_social_mentions(tickers: Sequence[str]) -> Dict[str, int]:
    """Count ticker mentions in recent posts of public subreddits (no auth)."""
    if not ENABLE_SOCIAL:
        return {}
    try:
        import requests
    except ImportError:
        logger.warning("ENABLE_SOCIAL=1 but 'requests' is not installed")
        return {}

    wanted = {t.upper() for t in tickers} - _SOCIAL_STOPWORDS
    counts: Dict[str, int] = {t: 0 for t in wanted}
    pattern = re.compile(r"\$?\b([A-Z]{1,5}(?:-[A-Z])?)\b")

    for sub in SOCIAL_SUBREDDITS:
        if not re.fullmatch(r"[A-Za-z0-9_]{2,30}", sub):
            continue
        url = f"https://www.reddit.com/r/{sub}/new.json?limit=100"
        try:
            resp = requests.get(url, headers={"User-Agent": SOCIAL_USER_AGENT}, timeout=12)
            if resp.status_code != 200:
                continue
            children = resp.json().get("data", {}).get("children", [])
        except Exception as exc:
            logger.debug("reddit fetch failed for %s: %s", sub, exc)
            continue

        for child in children:
            data = child.get("data", {}) or {}
            text = f"{data.get('title', '')} {data.get('selftext', '')}"[:4000]
            for match in set(pattern.findall(text)):
                if match in counts:
                    counts[match] += 1
        time.sleep(1.0)  # be polite to the public endpoint

    return counts


def _social_score(ticker: str, counts: Dict[str, int], notes: List[str]) -> float:
    if not counts:
        return NEUTRAL
    values = np.array(list(counts.values()), dtype=float)
    mine = float(counts.get(ticker.upper(), 0))
    if values.std() < 1e-9:
        return NEUTRAL
    z = (mine - values.mean()) / values.std()
    # Hype is a tiebreak, never a thesis: cap the swing hard.
    score = _clamp(50.0 + z * 10.0, 35.0, 70.0)
    if mine > 0 and z > 1.0:
        notes.append(f"Elevated retail chatter ({int(mine)} mentions)")
    return score


# =============================================================================
# PUBLIC API
# =============================================================================

def _weights() -> Dict[str, float]:
    w = {
        "analyst": W_ANALYST,
        "growth": W_GROWTH,
        "quality": W_QUALITY,
        "ownership": W_OWNERSHIP,
        "strength": W_STRENGTH,
    }
    if ENABLE_SOCIAL:
        w["social"] = W_SOCIAL
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def fetch_factor_score(
    ticker: str,
    yf_symbol: Optional[str] = None,
    spy_hist: Optional[pd.DataFrame] = None,
    social_counts: Optional[Dict[str, int]] = None,
    use_cache: bool = True,
) -> FactorScore:
    """Build the composite non-price score for one ticker."""
    symbol = yf_symbol or ticker
    cache = _load_cache() if use_cache else {}
    entry = cache.get(ticker)
    if use_cache and entry and _cache_fresh(entry):
        try:
            return FactorScore(**entry)
        except TypeError:
            pass

    notes: List[str] = []
    info: dict = {}
    upgrades = None
    hist = None
    partial = False

    try:
        tk = yf.Ticker(symbol)
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        if not info:
            partial = True
        try:
            upgrades = tk.upgrades_downgrades
        except Exception:
            upgrades = None
        try:
            hist = tk.history(period="1y", interval="1d", auto_adjust=True)
        except Exception:
            hist = None
        dte = _days_to_earnings(tk)
    except Exception as exc:
        logger.warning("factor fetch failed for %s: %s", ticker, exc)
        return FactorScore(ticker=ticker, notes=["Factor data unavailable"], partial=True,
                           fetched_at=datetime.now(timezone.utc).isoformat())

    price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    if price is None and hist is not None and not hist.empty:
        price = float(hist["Close"].iloc[-1])

    analyst, upside = score_analyst(info, upgrades, price, notes)
    growth = score_growth(info, notes)
    quality = score_quality(info, notes)
    ownership = score_ownership(info, notes)
    strength = score_strength(hist, spy_hist, notes)
    social = _social_score(ticker, social_counts or {}, notes) if ENABLE_SOCIAL else NEUTRAL

    buckets = {
        "analyst": analyst,
        "growth": growth,
        "quality": quality,
        "ownership": ownership,
        "strength": strength,
    }
    if ENABLE_SOCIAL:
        buckets["social"] = social

    weights = _weights()
    composite = sum(buckets[k] * weights[k] for k in buckets)

    blackout = dte is not None and 0 <= dte <= EARNINGS_BLACKOUT_DAYS
    if blackout:
        notes.append(f"Earnings in {dte}d — blackout")
    elif dte is not None and dte <= 21:
        notes.append(f"Earnings in {dte}d")

    result = FactorScore(
        ticker=ticker,
        score=round(_clamp(composite), 1),
        grade=_grade(composite),
        analyst=round(analyst, 1),
        growth=round(growth, 1),
        quality=round(quality, 1),
        ownership=round(ownership, 1),
        strength=round(strength, 1),
        social=round(social, 1),
        days_to_earnings=dte,
        earnings_blackout=blackout,
        target_upside_pct=round(upside, 1) if upside is not None else None,
        notes=notes[:8],
        fetched_at=datetime.now(timezone.utc).isoformat(),
        partial=partial,
    )

    if use_cache:
        cache[ticker] = result.to_dict()
        _save_cache(cache)
    return result


def fetch_factor_scores(
    tickers: Sequence[str],
    yf_map: Optional[Dict[str, str]] = None,
    use_cache: bool = True,
    progress: bool = False,
) -> Dict[str, FactorScore]:
    """Score a whole watchlist (SPY history and social counts fetched once)."""
    yf_map = yf_map or {}
    try:
        spy_hist = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
    except Exception:
        spy_hist = None

    social_counts = fetch_social_mentions(tickers) if ENABLE_SOCIAL else {}

    out: Dict[str, FactorScore] = {}
    for i, t in enumerate(tickers, 1):
        out[t] = fetch_factor_score(
            t,
            yf_symbol=yf_map.get(t, t),
            spy_hist=spy_hist,
            social_counts=social_counts,
            use_cache=use_cache,
        )
        if progress:
            fs = out[t]
            print(f"  [{i}/{len(tickers)}] {t:6s} factor={fs.score:5.1f} {fs.grade:2s} "
                  f"({fs.summary()})")
    return out


def rank_table(scores: Dict[str, FactorScore], top: int = 25) -> pd.DataFrame:
    rows = [s.to_dict() for s in scores.values()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = ["ticker", "score", "grade", "analyst", "growth", "quality",
            "ownership", "strength", "target_upside_pct", "days_to_earnings"]
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values("score", ascending=False).head(top).reset_index(drop=True)


if __name__ == "__main__":
    import sys

    syms = [s.upper() for s in sys.argv[1:]] or ["MSFT", "NVDA", "SNDK", "JPM"]
    print(f"Scoring {len(syms)} tickers (social={'ON' if ENABLE_SOCIAL else 'OFF'})...")
    res = fetch_factor_scores(syms, progress=True)
    print()
    print(rank_table(res, top=len(syms)).to_string(index=False))
    for t, s in res.items():
        if s.notes:
            print(f"\n{t}: " + "; ".join(s.notes))
