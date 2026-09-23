"""
📊 SIGNAL AGENT BACKTESTER
Tests signal accuracy on historical data before live deployment.

USAGE:
    python backtester.py

OUTPUT:
    - Performance metrics (win rate, profit factor, max drawdown, etc.)
    - Equity curve chart
    - Trade log CSV
    - Monthly breakdown
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from scipy.signal import argrelextrema
import json

# For charts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# =============================================================================
# CONFIGURATION
# =============================================================================

BACKTEST_CONFIG = {
    # Assets to backtest
    "assets": {
        "BTC/USDT": {"yf": "BTC-USD", "start": "2022-01-01", "end": "2026-07-01", "is_gold": False},
        "ETH/USDT": {"yf": "ETH-USD", "start": "2022-01-01", "end": "2026-07-01", "is_gold": False},
        "XAU/USD":  {"yf": "GC=F",    "start": "2022-01-01", "end": "2026-07-01", "is_gold": True},
    },

    # Account settings
    "initial_balance": 10000,
    "risk_per_trade_pct": 2.0,
    "max_leverage_crypto": 5,
    "max_leverage_gold": 10,
    "min_confidence": 70,

    # Signal settings (same as live agent)
    "scan_interval_hours": 4,
    "max_open_positions": 3,

    # Slippage & fees (be realistic)
    "slippage_pct": 0.05,      # 0.05% slippage on entry/exit
    "maker_fee_pct": 0.02,     # 0.02% maker fee
    "taker_fee_pct": 0.05,     # 0.05% taker fee
}


# =============================================================================
# DATA FETCHER
# =============================================================================

class BacktestDataFetcher:
    """Fetches historical data for backtesting."""

    def fetch(self, yf_symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch 1-hour data for the full backtest period."""
        print(f"  📥 Fetching {yf_symbol} from {start} to {end}...")

        try:
            data = yf.download(yf_symbol, start=start, end=end, interval="1h", progress=False)

            if data.empty:
                print(f"  ⚠️ No data returned for {yf_symbol}")
                return pd.DataFrame()

            # Flatten multi-index
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.reset_index()

            # Standardize columns
            rename_map = {
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Adj Close': 'close', 'Volume': 'volume',
                'Datetime': 'timestamp', 'Date': 'timestamp'
            }
            data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

            data['timestamp'] = pd.to_datetime(data['timestamp'])

            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    data[col] = data[col].astype(float)

            # Create 4h resampled data as well
            data_4h = data.set_index('timestamp').resample('4h').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna().reset_index()

            print(f"  ✅ Loaded {len(data)} 1h candles, {len(data_4h)} 4h candles")
            return data, data_4h

        except Exception as e:
            print(f"  ❌ Error fetching {yf_symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame()


# =============================================================================
# TECHNICAL INDICATORS (Same as live agent)
# =============================================================================

class TechnicalIndicators:
    @staticmethod
    def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(closes: pd.Series, period: int) -> pd.Series:
        return closes.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, min_periods=period).mean()

    @staticmethod
    def swing_points(series: pd.Series, order: int = 3) -> Tuple[pd.Series, pd.Series]:
        highs_idx = argrelextrema(series.values, np.greater, order=order)[0]
        lows_idx = argrelextrema(series.values, np.less, order=order)[0]
        highs = pd.Series(False, index=series.index)
        lows = pd.Series(False, index=series.index)
        if len(highs_idx) > 0:
            highs.iloc[highs_idx] = True
        if len(lows_idx) > 0:
            lows.iloc[lows_idx] = True
        return highs, lows

    @staticmethod
    def detect_divergence(price: pd.Series, indicator: pd.Series, 
                          lookback: int = 50, type: str = "bullish") -> bool:
        price_recent = price.tail(lookback)
        ind_recent = indicator.tail(lookback)

        if type == "bullish":
            p_lows = argrelextrema(price_recent.values, np.less, order=5)[0]
            i_lows = argrelextrema(ind_recent.values, np.less, order=5)[0]
            if len(p_lows) < 2 or len(i_lows) < 2:
                return False
            p_ll = price_recent.iloc[p_lows[-1]] < price_recent.iloc[p_lows[-2]]
            i_hl = ind_recent.iloc[i_lows[-1]] > ind_recent.iloc[i_lows[-2]]
            oversold = ind_recent.iloc[i_lows[-1]] < 40
            return p_ll and i_hl and oversold
        else:
            p_highs = argrelextrema(price_recent.values, np.greater, order=5)[0]
            i_highs = argrelextrema(ind_recent.values, np.greater, order=5)[0]
            if len(p_highs) < 2 or len(i_highs) < 2:
                return False
            p_hh = price_recent.iloc[p_highs[-1]] > price_recent.iloc[p_highs[-2]]
            i_lh = ind_recent.iloc[i_highs[-1]] < ind_recent.iloc[i_highs[-2]]
            overbought = ind_recent.iloc[i_highs[-1]] > 60
            return p_hh and i_lh and overbought


# =============================================================================
# SIGNAL GENERATOR (Same logic as live agent)
# =============================================================================

@dataclass
class Signal:
    asset: str
    direction: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    leverage: int
    confidence: int
    rationale: List[str]
    timestamp: datetime


class SignalGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.indicators = TechnicalIndicators()

    def generate(self, asset: str, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                 current_price: float, is_gold: bool, timestamp: datetime) -> Optional[Signal]:
        """Generate signal at a specific point in time (walk-forward)."""

        if df_4h.empty or df_1h.empty or current_price == 0:
            return None

        score = 0
        factors = []
        direction = None

        # === FACTOR 1: EMA Trend (4H) ===
        ema_50_4h = self.indicators.ema(df_4h['close'], 50)
        ema_200_4h = self.indicators.ema(df_4h['close'], 200)

        price_above_ema50 = current_price > ema_50_4h.iloc[-1]
        price_above_ema200 = current_price > ema_200_4h.iloc[-1]
        ema_bullish = ema_50_4h.iloc[-1] > ema_200_4h.iloc[-1]

        if price_above_ema50 and price_above_ema200 and ema_bullish:
            score += 15
            factors.append("4H EMA Bullish")
            direction = "LONG"
        elif not price_above_ema50 and not price_above_ema200 and not ema_bullish:
            score += 15
            factors.append("4H EMA Bearish")
            direction = "SHORT"

        # === FACTOR 2: RSI Divergence (1H) ===
        rsi_1h = self.indicators.rsi(df_1h['close'])

        if self.indicators.detect_divergence(df_1h['close'], rsi_1h, type="bullish"):
            score += 20
            factors.append("RSI Bullish Divergence")
            if not direction:
                direction = "LONG"
        elif self.indicators.detect_divergence(df_1h['close'], rsi_1h, type="bearish"):
            score += 20
            factors.append("RSI Bearish Divergence")
            if not direction:
                direction = "SHORT"

        # === FACTOR 3: Swing Structure (4H) ===
        swing_highs, swing_lows = self.indicators.swing_points(df_4h['close'], order=3)
        recent_highs = df_4h['close'][swing_highs].tail(3)
        recent_lows = df_4h['close'][swing_lows].tail(3)

        if len(recent_lows) >= 2 and len(recent_highs) >= 2:
            higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-2]
            higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-2]

            if higher_lows and higher_highs:
                score += 15
                factors.append("Higher Highs + Lows")
                if not direction:
                    direction = "LONG"
            elif not higher_lows and not higher_highs:
                score += 15
                factors.append("Lower Highs + Lows")
                if not direction:
                    direction = "SHORT"

        # === FACTOR 4: Volume Confirmation ===
        vol_avg = df_1h['volume'].tail(20).mean()
        vol_current = df_1h['volume'].iloc[-1]
        if vol_current > vol_avg * 1.5:
            score += 10
            factors.append(f"Volume Spike")

        # === FACTOR 5: ATR Volatility ===
        atr_4h = self.indicators.atr(df_4h)
        atr_current = atr_4h.iloc[-1]
        atr_avg = atr_4h.tail(20).mean()
        if atr_current > atr_avg * 1.2:
            score += 10
            factors.append("High Volatility")

        # === FACTOR 6: S/R Proximity ===
        if direction == "LONG" and len(recent_lows) > 0:
            nearest_support = recent_lows.iloc[-1]
            if abs(current_price - nearest_support) / current_price < 0.02:
                score += 10
                factors.append("Near Support")

        if direction == "SHORT" and len(recent_highs) > 0:
            nearest_resistance = recent_highs.iloc[-1]
            if abs(current_price - nearest_resistance) / current_price < 0.02:
                score += 10
                factors.append("Near Resistance")

        # === CHECK CONFIDENCE ===
        if score < self.config["min_confidence"] or not direction:
            return None

        # === CALCULATE LEVELS ===
        atr_1h = self.indicators.atr(df_1h).iloc[-1]

        if direction == "LONG":
            if len(recent_lows) > 0:
                sl = min(recent_lows.iloc[-1] * 0.998, current_price - atr_1h * 1.5)
            else:
                sl = current_price - atr_1h * 1.5
            sl = min(sl, current_price * 0.985)
            risk_dist = current_price - sl
            tp1 = current_price + risk_dist * 2
            tp2 = current_price + risk_dist * 4
            tp3 = current_price + risk_dist * 6
        else:
            if len(recent_highs) > 0:
                sl = max(recent_highs.iloc[-1] * 1.002, current_price + atr_1h * 1.5)
            else:
                sl = current_price + atr_1h * 1.5
            sl = max(sl, current_price * 1.015)
            risk_dist = sl - current_price
            tp1 = current_price - risk_dist * 2
            tp2 = current_price - risk_dist * 4
            tp3 = current_price - risk_dist * 6

        leverage = self.config["max_leverage_gold"] if is_gold else self.config["max_leverage_crypto"]

        # Safety check: liquidation distance
        if direction == "LONG":
            liq = current_price * (1 - (0.9 / leverage))
        else:
            liq = current_price * (1 + (0.9 / leverage))

        liq_dist = abs(current_price - liq)
        if liq_dist < abs(current_price - sl) * 3:
            return None

        return Signal(
            asset=asset,
            direction=direction,
            entry=current_price,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            leverage=leverage,
            confidence=min(score, 100),
            rationale=factors,
            timestamp=timestamp
        )


# =============================================================================
# TRADE SIMULATOR
# =============================================================================

@dataclass
class Trade:
    asset: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    leverage: int
    confidence: int
    entry_time: datetime
    exit_time: datetime
    exit_reason: str  # "SL", "TP1", "TP2", "TP3", "TIMEOUT"
    pnl_pct: float    # Percentage return on margin
    pnl_dollar: float # Dollar return
    risk_reward: float
    holding_hours: float


class TradeSimulator:
    """Simulates what happens after a signal is generated."""

    def __init__(self, config: dict):
        self.config = config
        self.slippage = config["slippage_pct"] / 100
        self.taker_fee = config["taker_fee_pct"] / 100

    def simulate(self, signal: Signal, future_data: pd.DataFrame) -> Optional[Trade]:
        """
        Simulate the trade from signal time forward.
        future_data: DataFrame of 1h candles AFTER the signal timestamp.
        """
        if future_data.empty:
            return None

        # Apply slippage to entry
        if signal.direction == "LONG":
            actual_entry = signal.entry * (1 + self.slippage)
        else:
            actual_entry = signal.entry * (1 - self.slippage)

        # Calculate position size based on risk
        account_balance = self.config["initial_balance"]  # Simplified
        risk_amount = account_balance * (self.config["risk_per_trade_pct"] / 100)
        stop_distance = abs(actual_entry - signal.stop_loss)
        position_size = risk_amount / stop_distance
        margin = (position_size * actual_entry) / signal.leverage

        # Walk forward through future candles
        for idx, row in future_data.iterrows():
            high = row['high']
            low = row['low']
            close = row['close']
            timestamp = row['timestamp']

            exit_price = None
            exit_reason = None

            if signal.direction == "LONG":
                # Check SL
                if low <= signal.stop_loss:
                    exit_price = signal.stop_loss * (1 - self.slippage)
                    exit_reason = "SL"
                # Check TP1 (close 50%)
                elif high >= signal.tp1 and exit_reason is None:
                    # For simplicity, we track if TP1 was hit and continue for remainder
                    # In real trading you'd partial close. Here we simulate full close at TP1
                    # or continue to TP2/TP3. Let's use a weighted approach:
                    # 50% at TP1, 25% at TP2, 25% at TP3
                    # Weighted exit price:
                    exit_price = signal.tp1 * 0.5 + signal.tp2 * 0.25 + signal.tp3 * 0.25
                    exit_price = exit_price * (1 - self.slippage)
                    exit_reason = "TP"

            else:  # SHORT
                if high >= signal.stop_loss:
                    exit_price = signal.stop_loss * (1 + self.slippage)
                    exit_reason = "SL"
                elif low <= signal.tp1 and exit_reason is None:
                    exit_price = signal.tp1 * 0.5 + signal.tp2 * 0.25 + signal.tp3 * 0.25
                    exit_price = exit_price * (1 + self.slippage)
                    exit_reason = "TP"

            if exit_reason:
                # Calculate P&L
                if signal.direction == "LONG":
                    price_change = exit_price - actual_entry
                else:
                    price_change = actual_entry - exit_price

                gross_pnl = price_change * position_size
                # Fees: entry + exit
                fees = (actual_entry * position_size * self.taker_fee * 2)
                net_pnl = gross_pnl - fees

                # P&L as % of margin
                pnl_pct = (net_pnl / margin) * 100

                # Risk:Reward
                risk = abs(actual_entry - signal.stop_loss)
                reward = abs(exit_price - actual_entry)
                r_r = reward / risk if risk > 0 else 0

                holding_hours = (timestamp - signal.timestamp).total_seconds() / 3600

                return Trade(
                    asset=signal.asset,
                    direction=signal.direction,
                    entry_price=actual_entry,
                    exit_price=exit_price,
                    stop_loss=signal.stop_loss,
                    tp1=signal.tp1,
                    tp2=signal.tp2,
                    tp3=signal.tp3,
                    leverage=signal.leverage,
                    confidence=signal.confidence,
                    entry_time=signal.timestamp,
                    exit_time=timestamp,
                    exit_reason=exit_reason,
                    pnl_pct=pnl_pct,
                    pnl_dollar=net_pnl,
                    risk_reward=r_r,
                    holding_hours=holding_hours
                )

        # If no exit by end of data, close at last price
        last_row = future_data.iloc[-1]
        exit_price = last_row['close']

        if signal.direction == "LONG":
            price_change = exit_price - actual_entry
        else:
            price_change = actual_entry - exit_price

        gross_pnl = price_change * position_size
        fees = (actual_entry * position_size * self.taker_fee * 2)
        net_pnl = gross_pnl - fees
        pnl_pct = (net_pnl / margin) * 100

        risk = abs(actual_entry - signal.stop_loss)
        reward = abs(exit_price - actual_entry)
        r_r = reward / risk if risk > 0 else 0

        holding_hours = (last_row['timestamp'] - signal.timestamp).total_seconds() / 3600

        return Trade(
            asset=signal.asset,
            direction=signal.direction,
            entry_price=actual_entry,
            exit_price=exit_price,
            stop_loss=signal.stop_loss,
            tp1=signal.tp1,
            tp2=signal.tp2,
            tp3=signal.tp3,
            leverage=signal.leverage,
            confidence=signal.confidence,
            entry_time=signal.timestamp,
            exit_time=last_row['timestamp'],
            exit_reason="TIMEOUT",
            pnl_pct=pnl_pct,
            pnl_dollar=net_pnl,
            risk_reward=r_r,
            holding_hours=holding_hours
        )


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """Main backtesting orchestrator."""

    def __init__(self, config: dict):
        self.config = config
        self.data_fetcher = BacktestDataFetcher()
        self.signal_generator = SignalGenerator(config)
        self.simulator = TradeSimulator(config)
        self.trades: List[Trade] = []
        self.signals_generated = 0
        self.signals_skipped = 0

    def run(self):
        """Run backtest on all configured assets."""
        print("\n" + "=" * 60)
        print("📊 SIGNAL AGENT BACKTEST")
        print("=" * 60)
        print(f"Period: 2022-01-01 to 2026-07-01")
        print(f"Assets: {list(self.config['assets'].keys())}")
        print(f"Min Confidence: {self.config['min_confidence']}%")
        print(f"Risk/Trade: {self.config['risk_per_trade_pct']}%")
        print("=" * 60 + "\n")

        for asset_name, asset_config in self.config["assets"].items():
            self._backtest_asset(asset_name, asset_config)

        self._generate_report()

    def _backtest_asset(self, asset_name: str, asset_config: dict):
        """Backtest a single asset."""
        print(f"\n🔷 Backtesting {asset_name}...")

        # Fetch data
        df_1h, df_4h = self.data_fetcher.fetch(
            asset_config["yf"], 
            asset_config["start"], 
            asset_config["end"]
        )

        if df_1h.empty or df_4h.empty:
            return

        # Walk forward: check for signals every N hours
        scan_interval = self.config["scan_interval_hours"]

        # Start after enough history (need 200 4h candles = ~33 days)
        start_idx = 200 * 4  # 200 4h candles = 800 hours of 1h data

        for i in range(start_idx, len(df_1h) - 48, scan_interval):  # -48 to ensure exit data
            current_time = df_1h.iloc[i]['timestamp']
            current_price = df_1h.iloc[i]['close']

            # Build lookback windows (NO look-ahead bias)
            hist_1h = df_1h.iloc[:i+1].copy()
            hist_4h = df_4h[df_4h['timestamp'] <= current_time].copy()

            # Future data for simulation
            future_1h = df_1h.iloc[i+1:].copy()

            # Generate signal
            signal = self.signal_generator.generate(
                asset=asset_name,
                df_1h=hist_1h,
                df_4h=hist_4h,
                current_price=current_price,
                is_gold=asset_config["is_gold"],
                timestamp=current_time
            )

            if signal:
                self.signals_generated += 1

                # Simulate trade
                trade = self.simulator.simulate(signal, future_1h.head(168))  # Max 1 week hold

                if trade:
                    self.trades.append(trade)
                    emoji = "✅" if trade.pnl_pct > 0 else "❌"
                    print(f"  {emoji} {trade.direction} | Entry: ${trade.entry_price:,.2f} | "
                          f"Exit: ${trade.exit_price:,.2f} | P&L: {trade.pnl_pct:+.2f}% | "
                          f"{trade.exit_reason} | R:R {trade.risk_reward:.2f}")
            else:
                self.signals_skipped += 1

        print(f"  📊 Done. Signals: {self.signals_generated}, Trades: {len(self.trades)}")

    def _generate_report(self):
        """Generate performance report."""
        if not self.trades:
            print("\n❌ No trades generated. Check your signal logic.")
            return

        trades_df = pd.DataFrame([{
            'asset': t.asset,
            'direction': t.direction,
            'entry': t.entry_price,
            'exit': t.exit_price,
            'pnl_pct': t.pnl_pct,
            'pnl_dollar': t.pnl_dollar,
            'exit_reason': t.exit_reason,
            'r_r': t.risk_reward,
            'confidence': t.confidence,
            'holding_hours': t.holding_hours,
            'entry_time': t.entry_time,
            'leverage': t.leverage
        } for t in self.trades])

        # ===== METRICS =====
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['pnl_pct'] > 0])
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean() if losses > 0 else 0

        gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy
        win_pct = win_rate / 100
        loss_pct = 1 - win_pct
        expectancy = (win_pct * avg_win) - (loss_pct * abs(avg_loss))

        # Max drawdown (simplified equity curve)
        equity = [self.config["initial_balance"]]
        for pnl in trades_df['pnl_dollar']:
            equity.append(equity[-1] + pnl)

        equity_series = pd.Series(equity)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        avg_holding = trades_df['holding_hours'].mean()
        avg_r_r = trades_df['r_r'].mean()

        # By asset
        by_asset = trades_df.groupby('asset').agg({
            'pnl_pct': ['count', 'mean', 'sum'],
            'exit_reason': lambda x: (x == 'TP').sum() / len(x) * 100
        }).round(2)

        # By direction
        by_direction = trades_df.groupby('direction')['pnl_pct'].agg(['count', 'mean', 'sum']).round(2)

        # ===== PRINT REPORT =====
        print("\n" + "=" * 60)
        print("📈 BACKTEST RESULTS")
        print("=" * 60)
        print(f"\nTotal Signals Generated: {self.signals_generated}")
        print(f"Trades Taken:            {total_trades}")
        print(f"Win Rate:                {win_rate:.1f}% ({wins}W / {losses}L)")
        print(f"Profit Factor:           {profit_factor:.2f}")
        print(f"Expectancy:              {expectancy:.2f}% per trade")
        print(f"Avg Win:                 {avg_win:+.2f}%")
        print(f"Avg Loss:                {avg_loss:.2f}%")
        print(f"Avg R:R:                 {avg_r_r:.2f}")
        print(f"Max Drawdown:            {max_drawdown:.2f}%")
        print(f"Avg Holding Time:        {avg_holding:.1f} hours")
        print(f"Total P&L:               ${sum(t.pnl_dollar for t in self.trades):,.2f}")
        print(f"Final Equity:            ${equity[-1]:,.2f}")
        print(f"Return:                  {((equity[-1] / self.config['initial_balance']) - 1) * 100:+.2f}%")

        print(f"\n📊 By Asset:")
        print(by_asset.to_string())

        print(f"\n📊 By Direction:")
        print(by_direction.to_string())

        print(f"\n📊 Exit Reasons:")
        print(trades_df['exit_reason'].value_counts().to_string())

        # ===== SAVE DATA =====
        trades_df.to_csv('backtest_trades.csv', index=False)
        print("\n💾 Trade log saved to: backtest_trades.csv")

        # ===== CHARTS =====
        self._plot_results(trades_df, equity, max_drawdown)

    def _plot_results(self, trades_df: pd.DataFrame, equity: list, max_dd: float):
        """Generate performance charts."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity, color='#22c55e', linewidth=1.5)
        ax1.axhline(y=self.config["initial_balance"], color='gray', linestyle='--', alpha=0.5)
        ax1.fill_between(range(len(equity)), equity, self.config["initial_balance"], 
                         where=[e >= self.config["initial_balance"] for e in equity],
                         alpha=0.2, color='green')
        ax1.fill_between(range(len(equity)), equity, self.config["initial_balance"],
                         where=[e < self.config["initial_balance"] for e in equity],
                         alpha=0.2, color='red')
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Account Balance ($)')
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        equity_series = pd.Series(equity)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.4)
        ax2.plot(drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True, alpha=0.3)

        # 3. P&L Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        colors = ['green' if x > 0 else 'red' for x in trades_df['pnl_pct']]
        ax3.bar(range(len(trades_df)), trades_df['pnl_pct'], color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linewidth=0.5)
        ax3.set_title('Trade P&L Distribution (%)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('P&L %')
        ax3.set_xlabel('Trade #')
        ax3.grid(True, alpha=0.3)

        # 4. Monthly Returns
        ax4 = fig.add_subplot(gs[2, 0])
        trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
        monthly = trades_df.groupby('month')['pnl_pct'].sum()
        colors = ['green' if x > 0 else 'red' for x in monthly]
        ax4.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linewidth=0.5)
        ax4.set_title('Monthly Returns (%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Return %')
        ax4.set_xticks(range(0, len(monthly), max(1, len(monthly)//6)))
        ax4.set_xticklabels([str(m) for m in monthly.index[::max(1, len(monthly)//6)]], rotation=45)
        ax4.grid(True, alpha=0.3)

        # 5. Win Rate by Confidence
        ax5 = fig.add_subplot(gs[2, 1])
        trades_df['conf_bucket'] = pd.cut(trades_df['confidence'], bins=[70, 75, 80, 85, 90, 100])
        conf_stats = trades_df.groupby('conf_bucket').agg({
            'pnl_pct': ['count', lambda x: (x > 0).sum() / len(x) * 100]
        })
        conf_stats.columns = ['count', 'win_rate']
        conf_stats = conf_stats.dropna()

        if not conf_stats.empty:
            ax5.bar(range(len(conf_stats)), conf_stats['win_rate'], color='steelblue', alpha=0.7)
            ax5.set_title('Win Rate by Confidence Bucket', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Win Rate %')
            ax5.set_xticks(range(len(conf_stats)))
            ax5.set_xticklabels([str(c) for c in conf_stats.index], rotation=45)
            ax5.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
            ax5.grid(True, alpha=0.3)
            ax5.legend()

        plt.suptitle('Signal Agent Backtest Results', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print("📊 Charts saved to: backtest_results.png")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    engine = BacktestEngine(BACKTEST_CONFIG)
    engine.run()
