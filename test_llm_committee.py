"""Quick test: Run a live LLM-powered committee analysis on NVDA."""
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

from genai_agents import supervisor

print("Running LLM-powered committee analysis on NVDA...")
print("This will make REAL Gemini API calls with function calling.")
print()

bb = supervisor.convene_committee("NVDA", "BULL", 25000)

print()
print("=" * 60)
print(f"ENGINE:     {bb.llm_mode}")
print(f"VERDICT:    {bb.verdict}")
print(f"CONFIDENCE: {bb.confidence}")
print(f"SHARES:     {bb.shares}")
print(f"ENTRY:      {bb.entry_price}")
print(f"STOP:       {bb.stop_loss}")
print(f"TP2 (20%):  {bb.take_profit_2}")
print(f"APPROVAL:   {bb.approval_status}")
print(f"MEMO:       {bb.investment_memo[:500]}")
print("=" * 60)
print(f"DIALOGUE ENTRIES: {len(bb.dialogue)}")
for d in bb.dialogue:
    agent = d.get("agent", "?")
    stance = d.get("stance", "?")
    notes = d.get("notes", "")[:150]
    print(f"  [{agent}] {stance}: {notes}")

# Check agent trace logs
import sqlite3
from pathlib import Path
db = Path(__file__).resolve().parent / "trading_committee.db"
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("SELECT COUNT(*) as cnt FROM agent_trace_logs WHERE ticker='NVDA'")
trace_count = cur.fetchone()["cnt"]
print(f"\nAGENT TRACE LOGS for NVDA: {trace_count} entries")

if trace_count > 0:
    cur.execute("SELECT agent_role, tool_calls_json FROM agent_trace_logs WHERE ticker='NVDA' ORDER BY id DESC LIMIT 5")
    for row in cur.fetchall():
        import json
        tools = json.loads(row["tool_calls_json"]) if row["tool_calls_json"] else []
        tool_names = [t.get("tool", "?") for t in tools]
        print(f"  [{row['agent_role']}] Tools called by LLM: {tool_names}")

conn.close()
print("\nTest complete.")
