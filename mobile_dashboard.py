"""
📱 MOBILE SIGNAL DASHBOARD
A lightweight web server that serves a phone-optimized trading dashboard.

USAGE:
    1. pip install flask
    2. python mobile_dashboard.py
    3. Open http://localhost:5000 on your phone (same WiFi)
    4. OR use ngrok for remote access

FOR REMOTE ACCESS (access from anywhere):
    pip install pyngrok
    python mobile_dashboard.py --ngrok
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

# =============================================================================
# DATA STORE (In production, this reads from your signal agent)
# =============================================================================

@dataclass
class Signal:
    asset: str
    direction: str
    entry: float
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    leverage: int
    confidence: int
    rationale: List[str]
    liquidation_price: float
    timestamp: str
    status: str = "ACTIVE"  # ACTIVE, TP1, TP2, TP3, SL, EXPIRED
    current_price: float = 0.0
    pnl_pct: float = 0.0

@dataclass
class TradeHistory:
    asset: str
    direction: str
    entry: float
    exit_price: float
    pnl_pct: float
    exit_reason: str
    date: str
    confidence: int

# Demo data — replace with live data from your agent
ACTIVE_SIGNALS = [
    Signal(
        asset="BTC/USDT", direction="LONG", entry=67340.0,
        entry_zone_low=67200.0, entry_zone_high=67480.0,
        stop_loss=66800.0, tp1=68450.0, tp2=69200.0, tp3=70100.0,
        leverage=5, confidence=82,
        rationale=["4H EMA Bullish", "RSI Bullish Divergence", "Near Support", "Volume Spike"],
        liquidation_price=56200.0, timestamp="2026-08-04 18:00",
        status="ACTIVE", current_price=68120.0, pnl_pct=1.16
    ),
    Signal(
        asset="XAU/USD", direction="SHORT", entry=2450.20,
        entry_zone_low=2445.0, entry_zone_high=2455.0,
        stop_loss=2468.0, tp1=2420.0, tp2=2395.0, tp3=2370.0,
        leverage=10, confidence=76,
        rationale=["4H EMA Bearish", "Lower Highs + Lows", "High Volatility"],
        liquidation_price=2695.0, timestamp="2026-08-04 14:00",
        status="ACTIVE", current_price=2438.50, pnl_pct=0.48
    ),
]

HISTORY = [
    TradeHistory("ETH/USDT", "LONG", 3250.0, 3380.0, 3.24, "TP2", "2026-08-03", 79),
    TradeHistory("SOL/USDT", "SHORT", 142.5, 145.35, -2.00, "SL", "2026-08-02", 71),
    TradeHistory("BTC/USDT", "LONG", 64500.0, 67800.0, 5.12, "TP3", "2026-08-01", 85),
    TradeHistory("XAU/USD", "SHORT", 2485.0, 2439.0, 1.84, "TP1", "2026-07-31", 74),
    TradeHistory("ETH/USDT", "LONG", 3100.0, 3068.0, -2.00, "SL", "2026-07-29", 68),
    TradeHistory("BTC/USDT", "SHORT", 68500.0, 67200.0, 1.90, "TP1", "2026-07-28", 73),
]

PORTFOLIO = {
    "total_equity": 12847.32,
    "initial_balance": 10000.0,
    "total_return_pct": 28.47,
    "win_rate": 54.2,
    "profit_factor": 1.68,
    "max_drawdown": -14.3,
    "total_trades": 87,
    "wins": 47,
    "losses": 40,
    "avg_trade": 0.84,
}

# =============================================================================
# HTML TEMPLATE — Mobile-First Design
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="theme-color" content="#0a0e1a">
    <title>Signal Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
        body {
            background: #0a0e1a;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 430px;
            margin: 0 auto;
            min-height: 100vh;
            padding-bottom: 80px;
        }
        .status-bar {
            background: #0f172a;
            padding: 8px 16px;
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #64748b;
        }
        .header {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            padding: 20px 16px 16px;
        }
        .header-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        .header h1 {
            font-size: 22px;
            font-weight: 800;
            color: #fbbf24;
            margin: 0;
        }
        .header p {
            margin: 4px 0 0;
            font-size: 12px;
            color: #94a3b8;
        }
        .badge {
            background: #22c55e;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
        }
        .portfolio-card {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 16px;
            margin-top: 16px;
        }
        .portfolio-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 12px;
        }
        .portfolio-label {
            font-size: 11px;
            color: #94a3b8;
        }
        .portfolio-value {
            font-size: 24px;
            font-weight: 800;
            color: white;
        }
        .portfolio-return {
            font-size: 18px;
            font-weight: 700;
            color: #22c55e;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
        }
        .stat-box {
            background: #0f172a;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
        }
        .stat-label {
            font-size: 10px;
            color: #94a3b8;
        }
        .stat-value {
            font-size: 16px;
            font-weight: 700;
        }
        .green { color: #22c55e; }
        .blue { color: #3b82f6; }
        .orange { color: #f59e0b; }
        .red { color: #ef4444; }

        .section {
            padding: 16px;
        }
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .section-header h2 {
            font-size: 16px;
            font-weight: 700;
            margin: 0;
        }
        .section-count {
            font-size: 11px;
            color: #64748b;
        }

        .signal-card {
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 12px;
            border: 1px solid;
        }
        .signal-card.long {
            background: linear-gradient(135deg, #14532d 0%, #0f172a 100%);
            border-color: #22c55e;
        }
        .signal-card.short {
            background: linear-gradient(135deg, #7f1d1d 0%, #0f172a 100%);
            border-color: #ef4444;
        }
        .signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .signal-badge {
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 700;
            color: white;
        }
        .signal-badge.long { background: #22c55e; }
        .signal-badge.short { background: #ef4444; }
        .signal-asset {
            font-size: 16px;
            font-weight: 700;
        }
        .signal-conf {
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
        }
        .signal-conf.high {
            background: rgba(34, 197, 94, 0.2);
            color: #22c55e;
        }
        .signal-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 12px;
        }
        .signal-cell {
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 8px;
        }
        .signal-cell-label {
            font-size: 10px;
            color: #94a3b8;
        }
        .signal-cell-value {
            font-size: 14px;
            font-weight: 700;
        }
        .signal-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .signal-meta {
            font-size: 11px;
            color: #94a3b8;
        }
        .signal-pnl {
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            color: white;
        }
        .signal-pnl.positive { background: #22c55e; }
        .signal-pnl.negative { background: #ef4444; }

        .history-list {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            overflow: hidden;
        }
        .history-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 14px;
            border-bottom: 1px solid #1e293b;
        }
        .history-item:last-child { border-bottom: none; }
        .history-left {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .history-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        .history-dot.win { background: #22c55e; }
        .history-dot.loss { background: #ef4444; }
        .history-name {
            font-size: 13px;
            font-weight: 600;
        }
        .history-date {
            font-size: 10px;
            color: #64748b;
        }
        .history-right {
            text-align: right;
        }
        .history-pnl {
            font-size: 13px;
            font-weight: 700;
        }
        .history-reason {
            font-size: 10px;
            color: #64748b;
        }

        .bottom-nav {
            position: fixed;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100%;
            max-width: 430px;
            background: #0f172a;
            border-top: 1px solid #1e293b;
            padding: 12px 16px;
            display: flex;
            justify-content: space-around;
        }
        .nav-item {
            text-align: center;
            color: #64748b;
            text-decoration: none;
        }
        .nav-item.active {
            color: #fbbf24;
        }
        .nav-icon {
            font-size: 20px;
        }
        .nav-label {
            font-size: 10px;
            font-weight: 600;
        }

        .risk-calc {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 14px;
        }
        .calc-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 12px;
        }
        .calc-input-group label {
            font-size: 10px;
            color: #94a3b8;
            display: block;
            margin-bottom: 4px;
        }
        .calc-input-group input {
            width: 100%;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px;
            color: white;
            font-size: 14px;
            outline: none;
        }
        .calc-input-group input:focus {
            border-color: #3b82f6;
        }
        .calc-btn {
            width: 100%;
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
        }
        .calc-results {
            margin-top: 12px;
            background: #1e293b;
            border-radius: 8px;
            padding: 10px;
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
            text-align: center;
        }
        .calc-result-label {
            font-size: 10px;
            color: #94a3b8;
        }
        .calc-result-value {
            font-size: 13px;
            font-weight: 700;
        }

        /* Animations */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .live-indicator {
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="status-bar">
        <span class="live-indicator">📡 Live</span>
        <span>🔋 100%</span>
    </div>

    <div class="header">
        <div class="header-top">
            <div>
                <h1>🎯 Signal Agent</h1>
                <p>Crypto & Gold · Multi-Factor Engine</p>
            </div>
            <div class="badge">● Active</div>
        </div>

        <div class="portfolio-card">
            <div class="portfolio-row">
                <div>
                    <div class="portfolio-label">Total Equity</div>
                    <div class="portfolio-value">${{ "%.2f"|format(portfolio.total_equity) }}</div>
                </div>
                <div style="text-align: right;">
                    <div class="portfolio-label">Total Return</div>
                    <div class="portfolio-return">+{{ "%.2f"|format(portfolio.total_return_pct) }}%</div>
                </div>
            </div>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Win Rate</div>
                    <div class="stat-value green">{{ "%.1f"|format(portfolio.win_rate) }}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Profit Factor</div>
                    <div class="stat-value blue">{{ "%.2f"|format(portfolio.profit_factor) }}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Max DD</div>
                    <div class="stat-value orange">{{ "%.1f"|format(portfolio.max_drawdown) }}%</div>
                </div>
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <h2>🔥 Active Signals</h2>
            <span class="section-count">{{ active_signals|length }} active</span>
        </div>

        {% for signal in active_signals %}
        <div class="signal-card {{ 'long' if signal.direction == 'LONG' else 'short' }}">
            <div class="signal-header">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div class="signal-badge {{ 'long' if signal.direction == 'LONG' else 'short' }}">
                        {{ signal.direction }}
                    </div>
                    <span class="signal-asset">{{ signal.asset }}</span>
                </div>
                <div class="signal-conf high">{{ signal.confidence }}% Conf</div>
            </div>

            <div class="signal-grid">
                <div class="signal-cell">
                    <div class="signal-cell-label">Entry</div>
                    <div class="signal-cell-value">${{ "%.2f"|format(signal.entry) }}</div>
                </div>
                <div class="signal-cell">
                    <div class="signal-cell-label">Current</div>
                    <div class="signal-cell-value {{ 'green' if signal.pnl_pct > 0 else 'red' }}">
                        ${{ "%.2f"|format(signal.current_price) }} {{ "▲" if signal.pnl_pct > 0 else "▼" }}
                    </div>
                </div>
                <div class="signal-cell">
                    <div class="signal-cell-label">SL</div>
                    <div class="signal-cell-value red">${{ "%.2f"|format(signal.stop_loss) }}</div>
                </div>
                <div class="signal-cell">
                    <div class="signal-cell-label">TP1</div>
                    <div class="signal-cell-value green">${{ "%.2f"|format(signal.tp1) }}</div>
                </div>
            </div>

            <div class="signal-footer">
                <div class="signal-meta">{{ signal.leverage }}x Lev · {{ signal.timestamp }}</div>
                <div class="signal-pnl {{ 'positive' if signal.pnl_pct > 0 else 'negative' }}">
                    {{ "+%.2f"|format(signal.pnl_pct) if signal.pnl_pct > 0 else "%.2f"|format(signal.pnl_pct) }}%
                </div>
            </div>
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <div class="section-header">
            <h2>📋 Recent Signals</h2>
            <span class="section-count">Last 7 days</span>
        </div>

        <div class="history-list">
            {% for trade in history %}
            <div class="history-item">
                <div class="history-left">
                    <div class="history-dot {{ 'win' if trade.pnl_pct > 0 else 'loss' }}"></div>
                    <div>
                        <div class="history-name">{{ trade.asset }} {{ trade.direction }}</div>
                        <div class="history-date">{{ trade.date }} · {{ trade.confidence }}% conf</div>
                    </div>
                </div>
                <div class="history-right">
                    <div class="history-pnl {{ 'green' if trade.pnl_pct > 0 else 'red' }}">
                        {{ "+%.2f"|format(trade.pnl_pct) if trade.pnl_pct > 0 else "%.2f"|format(trade.pnl_pct) }}%
                    </div>
                    <div class="history-reason">{{ trade.exit_reason }}</div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

    <div class="section">
        <h2 style="margin-bottom: 12px; font-size: 16px;">⚡ Quick Risk Calc</h2>
        <div class="risk-calc">
            <div class="calc-grid">
                <div class="calc-input-group">
                    <label>Balance ($)</label>
                    <input type="number" id="balance" value="10000" oninput="calculateRisk()">
                </div>
                <div class="calc-input-group">
                    <label>Entry ($)</label>
                    <input type="number" id="entry" value="67300" oninput="calculateRisk()">
                </div>
                <div class="calc-input-group">
                    <label>Stop Loss ($)</label>
                    <input type="number" id="stop" value="66800" oninput="calculateRisk()">
                </div>
                <div class="calc-input-group">
                    <label>Leverage (x)</label>
                    <input type="number" id="leverage" value="5" oninput="calculateRisk()">
                </div>
            </div>
            <button class="calc-btn" onclick="calculateRisk()">Calculate Position Size</button>
            <div class="calc-results" id="calcResults">
                <div>
                    <div class="calc-result-label">Position</div>
                    <div class="calc-result-value" id="posSize">0.04 BTC</div>
                </div>
                <div>
                    <div class="calc-result-label">Margin</div>
                    <div class="calc-result-value" id="margin">$538.40</div>
                </div>
                <div>
                    <div class="calc-result-label">Liq Price</div>
                    <div class="calc-result-value orange" id="liqPrice">~$56,200</div>
                </div>
            </div>
        </div>
    </div>

    <div class="bottom-nav">
        <a href="/" class="nav-item active">
            <div class="nav-icon">🏠</div>
            <div class="nav-label">Signals</div>
        </a>
        <a href="/portfolio" class="nav-item">
            <div class="nav-icon">📊</div>
            <div class="nav-label">Portfolio</div>
        </a>
        <a href="/history" class="nav-item">
            <div class="nav-icon">📈</div>
            <div class="nav-label">History</div>
        </a>
        <a href="/settings" class="nav-item">
            <div class="nav-icon">⚙️</div>
            <div class="nav-label">Settings</div>
        </a>
    </div>

    <script>
        function calculateRisk() {
            const balance = parseFloat(document.getElementById('balance').value) || 10000;
            const entry = parseFloat(document.getElementById('entry').value) || 67300;
            const stop = parseFloat(document.getElementById('stop').value) || 66800;
            const leverage = parseFloat(document.getElementById('leverage').value) || 5;

            const riskAmount = balance * 0.02;
            const stopDistance = Math.abs(entry - stop);
            const positionSize = riskAmount / stopDistance;
            const margin = (positionSize * entry) / leverage;
            const liq = entry > stop 
                ? entry * (1 - 0.9/leverage) 
                : entry * (1 + 0.9/leverage);

            document.getElementById('posSize').textContent = positionSize.toFixed(4) + ' units';
            document.getElementById('margin').textContent = '$' + margin.toFixed(2);
            document.getElementById('liqPrice').textContent = '~$' + liq.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        }

        // Auto-calculate on load
        calculateRisk();

        // Auto-refresh every 60 seconds
        setInterval(() => {
            fetch('/api/signals')
                .then(r => r.json())
                .then(data => {
                    // In a real app, this would update the DOM
                    console.log('Signals refreshed:', data);
                });
        }, 60000);
    </script>
</body>
</html>
"""

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML, 
                                   active_signals=ACTIVE_SIGNALS,
                                   history=HISTORY,
                                   portfolio=PORTFOLIO)

@app.route("/api/signals")
def api_signals():
    """JSON API for live data updates."""
    return jsonify({
        "active_signals": [asdict(s) for s in ACTIVE_SIGNALS],
        "portfolio": PORTFOLIO,
        "last_update": datetime.now().isoformat()
    })

@app.route("/api/history")
def api_history():
    return jsonify([asdict(t) for t in HISTORY])

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngrok", action="store_true", help="Expose via ngrok tunnel")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    args = parser.parse_args()

    if args.ngrok:
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(args.port)
            print(f"\n🌐 PUBLIC URL: {public_url}")
            print("📱 Open this URL on your phone from ANYWHERE\n")
        except ImportError:
            print("⚠️ pyngrok not installed. Run: pip install pyngrok")
            print("   Or use --ngrok after installing.\n")

    print(f"🚀 Dashboard running at: http://localhost:{args.port}")
    print(f"📱 Same WiFi? Open http://YOUR_COMPUTER_IP:{args.port} on phone")
    print("=" * 50)

    app.run(host="0.0.0.0", port=args.port, debug=False)
