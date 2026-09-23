"""Review today's live alerts vs subsequent price action."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from stock_agent_v3 import (
    APPROVED_TICKERS,
    REGIME_SYMBOL,
    STOCKS,
    _regime_from_spy,
    fetch_data,
    generate_signal,
    simulate_trade,
)

ET = ZoneInfo("America/New_York")
STATE = Path(__file__).with_name("live_signal_state.json")


def align_ts(ts: pd.Series, alert: datetime) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    alert_ts = pd.Timestamp(alert)
    if getattr(ts.dt, "tz", None) is not None:
        if alert_ts.tzinfo is None:
            alert_ts = alert_ts.tz_localize(ET)
        return alert_ts.tz_convert(ts.dt.tz)
    # naive bars — compare as naive ET wall clock
    if alert_ts.tzinfo is not None:
        return pd.Timestamp(alert_ts.tz_convert(ET).replace(tzinfo=None))
    return alert_ts


def main() -> None:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    alerts = state.get("last_alerts", {})
    last_scan = state.get("last_scan")
    print("=" * 72)
    print("TODAY LIVE SIGNAL REVIEW")
    print("=" * 72)
    print(f"Now ET:     {datetime.now(ET).strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"Last scan:  {last_scan}")
    print(f"Alerted:    {', '.join(alerts.keys()) or '(none)'}")
    if not alerts:
        print("No alerts stored.")
        return

    # Use earliest alert timestamp as reconstruction point
    alert_time = min(datetime.fromisoformat(v) for v in alerts.values())
    print(f"Reconstruct at: {alert_time.isoformat()}")
    print()

    spy = fetch_data(REGIME_SYMBOL, period="120d", interval="1h")
    alerted = list(alerts.keys())
    order = alerted + [t for t in APPROVED_TICKERS if t not in alerted]

    rows = []
    for t in order:
        df = fetch_data(STOCKS[t]["yf"], period="120d", interval="1h")
        if df.empty:
            print(f"{t}: no data")
            continue
        ts = pd.to_datetime(df["timestamp"])
        alert_ts = align_ts(ts, alert_time)
        hist = df.loc[ts <= alert_ts].reset_index(drop=True)
        if len(hist) < 220:
            # still try with what we have if close
            if len(hist) < 100:
                if t in alerted:
                    print(f"{t}: insufficient history at alert ({len(hist)} bars)")
                continue

        px = float(hist.iloc[-1]["close"])
        bar_ts = pd.Timestamp(hist.iloc[-1]["timestamp"]).to_pydatetime()
        regime = _regime_from_spy(spy, bar_ts)
        sig = generate_signal(t, hist, px, bar_ts, regime=regime)

        future = df.loc[ts > pd.Timestamp(hist.iloc[-1]["timestamp"])].reset_index(drop=True)
        last_px = float(df.iloc[-1]["close"])
        last_ts = pd.Timestamp(df.iloc[-1]["timestamp"])

        if sig is None:
            if t in alerted:
                print(f"{t}: ALERTED but cannot rebuild signal now (data/filter drift)")
                print(f"  last hist bar {bar_ts} px={px:.2f} regime={regime} n={len(hist)}")
            continue

        sim = simulate_trade(sig, future) if len(future) else None
        mtm = (last_px - sig.entry) / sig.entry * 100.0
        rows.append(
            {
                "ticker": t,
                "alerted": t in alerted,
                "conf": sig.confidence,
                "entry": sig.entry,
                "sl": sig.stop_loss,
                "tp1": sig.target_1,
                "tp2": sig.target_2,
                "risk_pct": sig.risk_pct,
                "regime": sig.regime,
                "last_px": last_px,
                "last_ts": str(last_ts),
                "mtm": mtm,
                "exit": sim["exit_reason"] if sim else "NO_FUTURE",
                "pnl": sim["pnl_pct"] if sim else None,
                "bars": sim["bars_held"] if sim else 0,
                "why": sig.rationale,
                "sig": sig,
                "future": future,
            }
        )

    print("-" * 72)
    print("ALERTED SIGNALS")
    print("-" * 72)
    alert_rows = [r for r in rows if r["alerted"]]
    if not alert_rows:
        print("Could not rebuild any alerted signals.")
    for r in alert_rows:
        pnl_s = f"{r['pnl']:+.2f}%" if r["pnl"] is not None else "n/a"
        print(f"\n{r['ticker']}  conf={r['conf']}  regime={r['regime']}")
        print(
            f"  Entry ${r['entry']:.2f} | SL ${r['sl']:.2f} | "
            f"TP1 ${r['tp1']:.2f} | TP2 ${r['tp2']:.2f} | risk {r['risk_pct']:.2f}%"
        )
        print(f"  Last ${r['last_px']:.2f} ({r['last_ts']}) | raw MTM {r['mtm']:+.2f}%")
        print(f"  Model exit: {r['exit']} | managed PnL {pnl_s} | bars after entry: {r['bars']}")
        print("  Why:")
        for w in r["why"]:
            print(f"    - {w}")

        # Hourly path
        sig = r["sig"]
        fut = r["future"]
        print("  Hourly path after entry:")
        hit_sl = hit_tp1 = hit_tp2 = be = False
        risk = abs(sig.entry - sig.stop_loss)
        if fut is None or fut.empty:
            print("    (no bars yet)")
        else:
            for _, row in fut.iterrows():
                hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
                tm = pd.Timestamp(row["timestamp"])
                flags = []
                if lo <= sig.stop_loss:
                    hit_sl = True
                    flags.append("touched SL")
                if hi >= sig.target_1:
                    hit_tp1 = True
                    flags.append("touched TP1")
                if hi >= sig.target_2:
                    hit_tp2 = True
                    flags.append("touched TP2")
                if hi >= sig.entry + 0.8 * risk:
                    be = True
                    flags.append("BE trigger")
                chg = (cl - sig.entry) / sig.entry * 100
                extra = f"  << {', '.join(flags)}" if flags else ""
                print(
                    f"    {tm}  H={hi:.2f} L={lo:.2f} C={cl:.2f} ({chg:+.2f}%){extra}"
                )
            print(
                f"  Hits so far: SL={hit_sl} | TP1={hit_tp1} | TP2={hit_tp2} | BE_trigger={be}"
            )

    print()
    print("-" * 72)
    print("OTHER APPROVED NAMES THAT ALSO QUALIFIED AT SAME TIME")
    print("-" * 72)
    others = [r for r in rows if not r["alerted"]]
    if not others:
        print("(none)")
    else:
        for r in others:
            pnl_s = f"{r['pnl']:+.2f}%" if r["pnl"] is not None else "n/a"
            print(
                f"  {r['ticker']:5s} conf={r['conf']:3d} entry={r['entry']:.2f} "
                f"mtm={r['mtm']:+.2f}% exit={r['exit']} pnl={pnl_s}"
            )

    print()
    print("=" * 72)
    print("SCORECARD (alerted only)")
    print("=" * 72)
    if alert_rows:
        wins = sum(1 for r in alert_rows if r["pnl"] is not None and r["pnl"] > 0)
        losses = sum(1 for r in alert_rows if r["pnl"] is not None and r["pnl"] <= 0)
        avg = sum(r["pnl"] for r in alert_rows if r["pnl"] is not None) / max(
            sum(1 for r in alert_rows if r["pnl"] is not None), 1
        )
        print(f"Signals: {len(alert_rows)} | model wins: {wins} | model losses: {losses}")
        print(f"Avg managed PnL so far: {avg:+.2f}%")
        print("Note: swing holds can last days; same-day MTM is only a first look.")
    print("Done.")


if __name__ == "__main__":
    main()
