"""Quick health check for live bot config + process + state."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import stock_agent_v3 as a  # noqa: E402


def main() -> None:
    names = a.tradeable_tickers()
    print(f"universe_size={len(names)}")
    print(f"approved_only={a.USE_APPROVED_ONLY}")
    print(f"min_confidence={a.MIN_CONFIDENCE}")
    print(f"cooldown_h={a.SIGNAL_COOLDOWN_HOURS}")
    print(f"scan_times={[t.strftime('%H:%M') for t in a.SCAN_TIMES_ET]}")
    print(f"sample={','.join(names[:12])}...")

    state_path = ROOT / "live_signal_state.json"
    if state_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        print(f"last_scan={st.get('last_scan')}")
        print(f"last_alerts={list((st.get('last_alerts') or {}).keys())}")
        print(f"subscribers={st.get('subscribers')}")
    else:
        print("last_scan=NO_STATE")

    # Find python processes running stock_agent_v3
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine", "/FORMAT:CSV"],
            text=True,
            errors="ignore",
        )
    except Exception as e:
        print(f"process_check_error={e}")
        out = ""

    live_pids = []
    for line in out.splitlines():
        if "stock_agent_v3" in line:
            parts = [p.strip() for p in line.split(",")]
            pid = parts[-1] if parts else "?"
            live_pids.append(pid)
            print(f"LIVE_PROCESS {line[:200]}")
    if not live_pids:
        print("LIVE_PROCESS none")

    stdout = ROOT / "live_stdout.log"
    stderr = ROOT / "live_stderr.log"
    if stdout.exists():
        txt = stdout.read_text(encoding="utf-8", errors="replace")
        print("---STDOUT_TAIL---")
        print("\n".join(txt.splitlines()[-20:]))
    if stderr.exists():
        txt = stderr.read_text(encoding="utf-8", errors="replace")
        # redact token-ish paths
        safe = []
        for ln in txt.splitlines()[-30:]:
            if "api.telegram.org/bot" in ln:
                safe.append(ln.split("/bot")[0] + "/bot<redacted> ...")
            else:
                safe.append(ln)
        print("---STDERR_TAIL---")
        print("\n".join(safe))


if __name__ == "__main__":
    main()
