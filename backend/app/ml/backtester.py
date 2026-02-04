"""
Pattern Backtester - Outcome Tracking for Smart Money Patterns

Runs SmartMoneyAnalyzer on historical data windows and tracks
forward price outcomes for each detected pattern. This creates
the labeled dataset needed for ML training (Tier 3+).
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .data_cache import get_data_cache
from ..services.smart_money_analyzer import SmartMoneyAnalyzer, SmartMoneyAnalysisResult

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


@dataclass
class PatternOutcome:
    """Forward outcome for a single detected pattern."""
    pattern_type: str
    detection_index: int
    detection_price: float
    direction: str  # 'bullish' or 'bearish'
    # Forward returns (percentage)
    return_10bar: float = 0.0
    return_20bar: float = 0.0
    return_50bar: float = 0.0
    # Did price reach target?
    hit_target: bool = False
    hit_stop: bool = False
    # Risk/Reward achieved
    max_favorable: float = 0.0  # Max favorable excursion (%)
    max_adverse: float = 0.0    # Max adverse excursion (%)
    risk_reward: float = 0.0
    # Metadata
    pattern_price_high: float = 0.0
    pattern_price_low: float = 0.0
    timestamp: Optional[str] = None


@dataclass
class PatternStats:
    """Aggregated statistics for a pattern type."""
    pattern_type: str
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_return_10bar: float = 0.0
    avg_return_20bar: float = 0.0
    avg_return_50bar: float = 0.0
    avg_rr: float = 0.0
    profit_factor: float = 0.0
    avg_max_favorable: float = 0.0
    avg_max_adverse: float = 0.0
    max_drawdown_pct: float = 0.0
    sample_count: int = 0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    symbol: str
    timeframe: str
    lookback_days: int
    total_bars: int = 0
    windows_analyzed: int = 0
    pattern_stats: Dict[str, PatternStats] = field(default_factory=dict)
    outcomes: List[PatternOutcome] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class PatternBacktester:
    """
    Backtests Smart Money patterns on historical data.

    Slides a window across historical OHLCV data, runs SmartMoneyAnalyzer
    on each window, and tracks what happened after each detected pattern.
    """

    def __init__(self, window_size: int = 100, step_size: int = 10):
        """
        Args:
            window_size: Number of candles per analysis window
            step_size: How many candles to advance between windows
        """
        self.window_size = window_size
        self.step_size = step_size
        self.data_cache = get_data_cache()
        self.analyzer = SmartMoneyAnalyzer(use_ml=False)  # Rule-based for backtesting

    def backtest_patterns(
        self,
        symbol: str,
        timeframe: str = 'D1',
        lookback_days: int = 365,
        forward_bars: int = 50,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe to analyze
            lookback_days: How far back to look
            forward_bars: How many bars forward to track outcome

        Returns:
            BacktestResult with per-pattern statistics
        """
        logger.info(f"Starting backtest: {symbol} {timeframe} {lookback_days}d")

        # Fetch data
        data = self.data_cache.get_ohlcv(symbol, timeframe, lookback_days)
        if len(data) < self.window_size + forward_bars:
            logger.warning(f"Insufficient data: {len(data)} bars (need {self.window_size + forward_bars})")
            return BacktestResult(
                symbol=symbol, timeframe=timeframe,
                lookback_days=lookback_days, total_bars=len(data)
            )

        result = BacktestResult(
            symbol=symbol, timeframe=timeframe,
            lookback_days=lookback_days, total_bars=len(data)
        )

        all_outcomes: List[PatternOutcome] = []
        windows_analyzed = 0

        # Slide window across data
        max_start = len(data) - self.window_size - forward_bars
        for start_idx in range(0, max_start, self.step_size):
            end_idx = start_idx + self.window_size
            window = data.iloc[start_idx:end_idx]
            forward = data.iloc[end_idx:end_idx + forward_bars]

            if len(window) < self.window_size or len(forward) < 10:
                continue

            try:
                analysis = self.analyzer.analyze(window)
                outcomes = self._evaluate_patterns(analysis, window, forward)
                all_outcomes.extend(outcomes)
                windows_analyzed += 1
            except Exception as e:
                logger.debug(f"Window {start_idx} failed: {e}")
                continue

        result.windows_analyzed = windows_analyzed
        result.outcomes = all_outcomes

        # Aggregate stats per pattern type
        result.pattern_stats = self._aggregate_stats(all_outcomes)

        logger.info(
            f"Backtest complete: {windows_analyzed} windows, "
            f"{len(all_outcomes)} pattern outcomes, "
            f"{len(result.pattern_stats)} pattern types"
        )

        return result

    def _evaluate_patterns(
        self,
        analysis: SmartMoneyAnalysisResult,
        window: pd.DataFrame,
        forward: pd.DataFrame,
    ) -> List[PatternOutcome]:
        """Evaluate forward outcomes for all patterns in an analysis."""
        outcomes = []
        detection_price = float(window['close'].iloc[-1])
        forward_closes = forward['close'].values
        forward_highs = forward['high'].values
        forward_lows = forward['low'].values

        # Order Blocks
        for ob in analysis.order_blocks:
            outcome = self._compute_outcome(
                pattern_type='order_block',
                direction=ob.type,
                detection_price=detection_price,
                detection_index=ob.start_index,
                pattern_high=ob.high,
                pattern_low=ob.low,
                forward_closes=forward_closes,
                forward_highs=forward_highs,
                forward_lows=forward_lows,
                target_zone=(ob.low, ob.high),
            )
            outcomes.append(outcome)

        # Fair Value Gaps
        for fvg in analysis.fair_value_gaps:
            outcome = self._compute_outcome(
                pattern_type='fvg',
                direction=fvg.type,
                detection_price=detection_price,
                detection_index=fvg.index,
                pattern_high=fvg.high,
                pattern_low=fvg.low,
                forward_closes=forward_closes,
                forward_highs=forward_highs,
                forward_lows=forward_lows,
                target_zone=(fvg.low, fvg.high),
            )
            outcomes.append(outcome)

        # Displacements
        for disp in analysis.displacements:
            outcome = self._compute_outcome(
                pattern_type='displacement',
                direction=disp['type'],
                detection_price=detection_price,
                detection_index=disp['index'],
                pattern_high=disp['high'],
                pattern_low=disp['low'],
                forward_closes=forward_closes,
                forward_highs=forward_highs,
                forward_lows=forward_lows,
            )
            outcomes.append(outcome)

        # OTE Zones
        for ote in analysis.ote_zones:
            outcome = self._compute_outcome(
                pattern_type='optimal_trade_entry',
                direction=ote['type'],
                detection_price=detection_price,
                detection_index=0,
                pattern_high=ote['ote_high'],
                pattern_low=ote['ote_low'],
                forward_closes=forward_closes,
                forward_highs=forward_highs,
                forward_lows=forward_lows,
                target_zone=(ote['ote_low'], ote['ote_high']),
            )
            outcomes.append(outcome)

        # Breaker Blocks
        for bb in analysis.breaker_blocks:
            outcome = self._compute_outcome(
                pattern_type='breaker_block',
                direction='bullish' if bb['type'] == 'bullish_breaker' else 'bearish',
                detection_price=detection_price,
                detection_index=bb['index'],
                pattern_high=bb['high'],
                pattern_low=bb['low'],
                forward_closes=forward_closes,
                forward_highs=forward_highs,
                forward_lows=forward_lows,
            )
            outcomes.append(outcome)

        # Buy/Sell Stops
        for bs in analysis.buy_sell_stops.get('buy_stops', []):
            outcome = self._compute_outcome(
                pattern_type='buy_stops',
                direction='bullish',
                detection_price=detection_price,
                detection_index=0,
                pattern_high=bs['level'],
                pattern_low=bs['level'] * 0.99,
                forward_closes=forward_closes,
                forward_highs=forward_highs,
                forward_lows=forward_lows,
            )
            outcomes.append(outcome)

        for ss in analysis.buy_sell_stops.get('sell_stops', []):
            outcome = self._compute_outcome(
                pattern_type='sell_stops',
                direction='bearish',
                detection_price=detection_price,
                detection_index=0,
                pattern_high=ss['level'] * 1.01,
                pattern_low=ss['level'],
                forward_closes=forward_closes,
                forward_highs=forward_highs,
                forward_lows=forward_lows,
            )
            outcomes.append(outcome)

        return outcomes

    def _compute_outcome(
        self,
        pattern_type: str,
        direction: str,
        detection_price: float,
        detection_index: int,
        pattern_high: float,
        pattern_low: float,
        forward_closes: np.ndarray,
        forward_highs: np.ndarray,
        forward_lows: np.ndarray,
        target_zone: Optional[Tuple[float, float]] = None,
    ) -> PatternOutcome:
        """Compute forward outcome for a single pattern."""
        outcome = PatternOutcome(
            pattern_type=pattern_type,
            detection_index=detection_index,
            detection_price=detection_price,
            direction=direction,
            pattern_price_high=pattern_high,
            pattern_price_low=pattern_low,
        )

        if len(forward_closes) == 0 or detection_price == 0:
            return outcome

        # Forward returns at different horizons
        if len(forward_closes) >= 10:
            raw_return = (forward_closes[9] - detection_price) / detection_price * 100
            outcome.return_10bar = raw_return if direction == 'bullish' else -raw_return

        if len(forward_closes) >= 20:
            raw_return = (forward_closes[19] - detection_price) / detection_price * 100
            outcome.return_20bar = raw_return if direction == 'bullish' else -raw_return

        if len(forward_closes) >= 50:
            raw_return = (forward_closes[49] - detection_price) / detection_price * 100
            outcome.return_50bar = raw_return if direction == 'bullish' else -raw_return

        # Max favorable/adverse excursion
        if direction == 'bullish':
            max_high = float(np.max(forward_highs))
            max_low = float(np.min(forward_lows))
            outcome.max_favorable = (max_high - detection_price) / detection_price * 100
            outcome.max_adverse = (detection_price - max_low) / detection_price * 100
        else:
            max_high = float(np.max(forward_highs))
            max_low = float(np.min(forward_lows))
            outcome.max_favorable = (detection_price - max_low) / detection_price * 100
            outcome.max_adverse = (max_high - detection_price) / detection_price * 100

        # Risk/Reward
        if outcome.max_adverse > 0:
            outcome.risk_reward = outcome.max_favorable / outcome.max_adverse
        else:
            outcome.risk_reward = outcome.max_favorable * 10  # Capped

        # Target hit check
        if target_zone:
            target_low, target_high = target_zone
            if direction == 'bullish':
                # For bullish: did price reach above target zone?
                outcome.hit_target = float(np.max(forward_highs)) >= target_high
                outcome.hit_stop = float(np.min(forward_lows)) < target_low * 0.98
            else:
                # For bearish: did price drop below target zone?
                outcome.hit_target = float(np.min(forward_lows)) <= target_low
                outcome.hit_stop = float(np.max(forward_highs)) > target_high * 1.02
        else:
            # Default: consider profitable if 20-bar return > 0
            outcome.hit_target = outcome.return_20bar > 0
            outcome.hit_stop = outcome.return_20bar < -2.0  # 2% adverse

        return outcome

    def _aggregate_stats(self, outcomes: List[PatternOutcome]) -> Dict[str, PatternStats]:
        """Aggregate outcomes into per-pattern statistics."""
        by_type: Dict[str, List[PatternOutcome]] = {}
        for o in outcomes:
            by_type.setdefault(o.pattern_type, []).append(o)

        stats = {}
        for ptype, type_outcomes in by_type.items():
            n = len(type_outcomes)
            wins = sum(1 for o in type_outcomes if o.return_20bar > 0)
            losses = n - wins

            gross_profit = sum(o.return_20bar for o in type_outcomes if o.return_20bar > 0)
            gross_loss = abs(sum(o.return_20bar for o in type_outcomes if o.return_20bar <= 0))

            # Compute equity curve for max drawdown
            returns = [o.return_20bar for o in type_outcomes]
            cumulative = np.cumsum(returns) if returns else np.array([0])
            peak = np.maximum.accumulate(cumulative) if len(cumulative) > 0 else np.array([0])
            drawdown = peak - cumulative
            max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

            stats[ptype] = PatternStats(
                pattern_type=ptype,
                total_signals=n,
                wins=wins,
                losses=losses,
                win_rate=wins / n * 100 if n > 0 else 0,
                avg_return_10bar=float(np.mean([o.return_10bar for o in type_outcomes])) if n > 0 else 0,
                avg_return_20bar=float(np.mean([o.return_20bar for o in type_outcomes])) if n > 0 else 0,
                avg_return_50bar=float(np.mean([o.return_50bar for o in type_outcomes])) if n > 0 else 0,
                avg_rr=float(np.mean([o.risk_reward for o in type_outcomes])) if n > 0 else 0,
                profit_factor=gross_profit / gross_loss if gross_loss > 0 else gross_profit * 10,
                avg_max_favorable=float(np.mean([o.max_favorable for o in type_outcomes])) if n > 0 else 0,
                avg_max_adverse=float(np.mean([o.max_adverse for o in type_outcomes])) if n > 0 else 0,
                max_drawdown_pct=max_dd,
                sample_count=n,
            )

        return stats

    def save_results(self, result: BacktestResult) -> None:
        """Save backtest results to SQLite."""
        from ..database import Database
        db = Database()

        with db.get_connection() as conn:
            cursor = conn.cursor()
            for ptype, stats in result.pattern_stats.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO backtest_results
                    (symbol, timeframe, pattern_type, total_signals, win_rate,
                     avg_return_pct, profit_factor, avg_rr, max_drawdown_pct,
                     sample_count, lookback_days, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.symbol, result.timeframe, ptype,
                    stats.total_signals, stats.win_rate,
                    stats.avg_return_20bar, stats.profit_factor,
                    stats.avg_rr, stats.max_drawdown_pct,
                    stats.sample_count, result.lookback_days,
                    result.created_at,
                ))

    def get_saved_results(self, symbol: str, timeframe: str = None) -> List[Dict]:
        """Load saved backtest results from SQLite."""
        from ..database import Database
        db = Database()

        with db.get_connection() as conn:
            cursor = conn.cursor()
            if timeframe:
                cursor.execute(
                    "SELECT * FROM backtest_results WHERE symbol = ? AND timeframe = ? ORDER BY created_at DESC",
                    (symbol, timeframe)
                )
            else:
                cursor.execute(
                    "SELECT * FROM backtest_results WHERE symbol = ? ORDER BY created_at DESC",
                    (symbol,)
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_aggregate_performance(self) -> Dict:
        """Get aggregate performance across all symbols."""
        from ..database import Database
        db = Database()

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pattern_type,
                       COUNT(*) as entries,
                       AVG(win_rate) as avg_win_rate,
                       AVG(profit_factor) as avg_pf,
                       AVG(avg_rr) as avg_rr,
                       SUM(sample_count) as total_samples
                FROM backtest_results
                GROUP BY pattern_type
                ORDER BY avg_win_rate DESC
            """)
            results = {}
            for row in cursor.fetchall():
                results[row[0]] = {
                    'pattern_type': row[0],
                    'backtest_entries': row[1],
                    'avg_win_rate': round(row[2], 2) if row[2] else 0,
                    'avg_profit_factor': round(row[3], 2) if row[3] else 0,
                    'avg_rr': round(row[4], 2) if row[4] else 0,
                    'total_samples': row[5] or 0,
                }
            return results

    def to_dict(self, result: BacktestResult) -> Dict:
        """Convert BacktestResult to serializable dict."""
        return {
            'symbol': result.symbol,
            'timeframe': result.timeframe,
            'lookback_days': result.lookback_days,
            'total_bars': result.total_bars,
            'windows_analyzed': result.windows_analyzed,
            'pattern_stats': {
                k: asdict(v) for k, v in result.pattern_stats.items()
            },
            'total_outcomes': len(result.outcomes),
            'created_at': result.created_at,
        }


# Singleton
_backtester_instance = None

def get_backtester() -> PatternBacktester:
    global _backtester_instance
    if _backtester_instance is None:
        _backtester_instance = PatternBacktester()
    return _backtester_instance
