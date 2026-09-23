"""Wait for full-universe backtest log to finish and print summary."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Prefer newest active log if present
_LOG_CANDIDATES = [
    ROOT / "backtest_v33_tight_run.log",
    ROOT / "backtest_full_run.log",
    ROOT / "backtest_v33_tight_run.log",
]
LOG = next((p for p in _LOG_CANDIDATES if p.exists()), _LOG_CANDIDATES[0])
CSV_CANDIDATES = [
    ROOT / "backtest_v33_full_tight_results.csv",
    ROOT / "backtest_v33_approved_tight_results.csv",
    ROOT / "backtest_v32_approved_results.csv",
    ROOT / "backtest_v32_full_results.csv",
    ROOT / "backtest_full_results.csv",
]


def read_log() -> str:
    if not LOG.exists():
        return ""
    raw = LOG.read_bytes()
    # Tee-Object often writes UTF-16 LE
    for enc in ("utf-16", "utf-16-le", "utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def backtest_running() -> bool:
    try:
        out = subprocess.check_output(
            [
                "wmic",
                "process",
                "where",
                "name='python.exe'",
                "get",
                "ProcessId,CommandLine",
                "/FORMAT:CSV",
            ],
            text=True,
            errors="ignore",
        )
    except Exception:
        return False
    return any("backtest" in ln and "stock_agent_v3" in ln for ln in out.splitlines())


def main() -> int:
    deadline = time.time() + 50 * 60
    last_bt = -1
    while time.time() < deadline:
        text = read_log()
        done = len(re.findall(r"^Backtesting\s+\S+", text, flags=re.M))
        if done != last_bt:
            print(f"progress tickers_done={done}", flush=True)
            last_bt = done
        if "BACKTEST RESULTS" in text or "Saved:" in text:
            print("COMPLETE", flush=True)
            # print last 80 lines
            lines = text.splitlines()
            print("\n".join(lines[-80:]), flush=True)
            return 0
        if not backtest_running() and done > 0:
            print("PROCESS_ENDED_WITHOUT_MARKER", flush=True)
            lines = text.splitlines()
            print("\n".join(lines[-100:]), flush=True)
            return 1
        time.sleep(20)
    print("TIMEOUT", flush=True)
    print(read_log().splitlines()[-40:])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
