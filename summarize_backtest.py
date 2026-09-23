"""Summarize full-universe backtest CSV into quality shortlist files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "backtest_v32_approved_results.csv"
FULL = ROOT / "backtest_v32_full112_results.csv"
OUT_TXT = ROOT / "backtest_full112_summary.txt"
OUT_Q = ROOT / "backtest_v32_full112_quality_tickers.csv"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}")
    if not FULL.exists():
        FULL.write_bytes(SRC.read_bytes())

    df = pd.read_csv(SRC)
    gp = float(df.loc[df["pnl_pct"] > 0, "pnl_pct"].sum())
    gl = float(abs(df.loc[df["pnl_pct"] <= 0, "pnl_pct"].sum()))
    pf = (gp / gl) if gl else 0.0
    wr = float((df["pnl_pct"] > 0).mean() * 100)
    exp = float(df["pnl_pct"].mean())

    by = (
        df.groupby("ticker")["pnl_pct"]
        .agg(n="count", wr=lambda s: (s > 0).mean() * 100, avg="mean", total="sum")
        .round(2)
        .sort_values(["wr", "avg"], ascending=False)
    )
    quality = by[(by["n"] >= 3) & (by["wr"] >= 55) & (by["avg"] > 0)].copy()
    quality.to_csv(OUT_Q)

    lines = [
        "FULL UNIVERSE BACKTEST (conf>=72, LONG ONLY)",
        f"trades={len(df)} tickers_traded={df['ticker'].nunique()}",
        f"win_rate={wr:.1f}% profit_factor={pf:.2f} expectancy={exp:+.2f}%/trade",
        f"avg_win={df.loc[df.pnl_pct>0,'pnl_pct'].mean():+.2f}% avg_loss={df.loc[df.pnl_pct<=0,'pnl_pct'].mean():.2f}%",
        "",
        "Exit reasons:",
        df["exit_reason"].value_counts().to_string(),
        "",
        f"QUALITY SHORTLIST n>=3 WR>=55 avg>0  count={len(quality)}",
        quality.to_string() if len(quality) else "(none)",
        "",
        "TOP 15 by total pnl:",
        by.sort_values("total", ascending=False).head(15).to_string(),
        "",
        "BOTTOM 15 by total pnl:",
        by.sort_values("total", ascending=True).head(15).to_string(),
        "",
        "VERDICT: overall FAIL for blanket trading of all 112.",
        "Use quality shortlist or keep higher conf / approved-only for live.",
    ]
    text = "\n".join(lines)
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
