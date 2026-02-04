"""
Parameter Optimizer - Walk-Forward Threshold Optimization (Tier 2)

Uses backtesting results to find optimal detection thresholds for
each pattern type. Replaces hardcoded values with data-driven params.

Walk-forward validation: train on 70%, test on 30% to avoid overfitting.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data_cache import get_data_cache
from .backtester import PatternBacktester, BacktestResult

logger = logging.getLogger(__name__)


# Default thresholds (current hardcoded values)
DEFAULT_PARAMS = {
    'fvg_min_gap_pct': 0.0001,       # Minimum FVG gap size as % of price
    'ob_min_move_strength': 0.3,      # Minimum OB impulse strength
    'ob_confidence_multiplier': 0.5,  # OB confidence scaling
    'swing_lookback': 5,              # Candles for swing detection
    'displacement_body_ratio': 0.70,  # Min body/range for displacement
    'displacement_range_mult': 1.2,   # Min range vs avg for displacement
    'ote_fib_low': 0.62,              # OTE zone lower fib
    'ote_fib_high': 0.79,             # OTE zone upper fib
    'equal_level_tolerance': 0.001,   # % tolerance for equal highs/lows
    'breaker_lookforward': 10,        # Bars to check for breaker confirmation
}

# Search ranges per parameter
PARAM_RANGES = {
    'fvg_min_gap_pct':         (0.00001, 0.005),
    'ob_min_move_strength':    (0.05, 0.8),
    'ob_confidence_multiplier': (0.2, 1.0),
    'swing_lookback':          (3, 10),
    'displacement_body_ratio': (0.50, 0.90),
    'displacement_range_mult': (0.8, 2.0),
    'ote_fib_low':             (0.50, 0.70),
    'ote_fib_high':            (0.70, 0.90),
    'equal_level_tolerance':   (0.0005, 0.005),
    'breaker_lookforward':     (5, 20),
}


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    symbol: str
    timeframe: str
    optimized_params: Dict[str, float] = field(default_factory=dict)
    default_metrics: Dict[str, float] = field(default_factory=dict)
    optimized_metrics: Dict[str, float] = field(default_factory=dict)
    improvement: Dict[str, float] = field(default_factory=dict)
    validation_win_rate: float = 0.0
    validation_profit_factor: float = 0.0
    train_period: str = ""
    test_period: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ParameterOptimizer:
    """
    Walk-forward parameter optimizer for Smart Money patterns.

    1. Fetches historical data
    2. Splits into train (70%) and test (30%) periods
    3. Grid searches parameter space on training data
    4. Validates best params on test data
    5. Saves if validation improves over defaults
    """

    def __init__(self):
        self.data_cache = get_data_cache()

    def optimize_thresholds(
        self,
        symbol: str,
        timeframe: str = 'D1',
        lookback_days: int = 730,
    ) -> Dict:
        """
        Run full optimization cycle.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe to optimize
            lookback_days: Total data period (70% train, 30% test)

        Returns:
            Dict with optimized params and comparison metrics
        """
        logger.info(f"Starting optimization: {symbol} {timeframe}")

        # Get data
        data = self.data_cache.get_ohlcv(symbol, timeframe, lookback_days)
        if len(data) < 200:
            return {"error": f"Insufficient data: {len(data)} bars (need 200+)"}

        # Split 70/30
        split_idx = int(len(data) * 0.7)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        train_period = f"{train_data.index[0]} to {train_data.index[-1]}"
        test_period = f"{test_data.index[0]} to {test_data.index[-1]}"

        logger.info(f"Train: {len(train_data)} bars, Test: {len(test_data)} bars")

        # Step 1: Get baseline metrics with default params
        default_metrics = self._evaluate_params(DEFAULT_PARAMS, train_data)

        # Step 2: Grid search on training data
        best_params = self._grid_search(train_data)

        # Step 3: Validate on test data
        default_test_metrics = self._evaluate_params(DEFAULT_PARAMS, test_data)
        optimized_test_metrics = self._evaluate_params(best_params, test_data)

        # Step 4: Calculate improvement
        improvement = {}
        for key in default_test_metrics:
            d = default_test_metrics[key]
            o = optimized_test_metrics[key]
            if d != 0:
                improvement[key] = round((o - d) / abs(d) * 100, 2)
            else:
                improvement[key] = 0.0

        result = OptimizationResult(
            symbol=symbol,
            timeframe=timeframe,
            optimized_params=best_params,
            default_metrics=default_test_metrics,
            optimized_metrics=optimized_test_metrics,
            improvement=improvement,
            validation_win_rate=optimized_test_metrics.get('avg_win_rate', 0),
            validation_profit_factor=optimized_test_metrics.get('avg_profit_factor', 0),
            train_period=train_period,
            test_period=test_period,
        )

        # Save if improved
        if optimized_test_metrics.get('avg_win_rate', 0) > default_test_metrics.get('avg_win_rate', 0):
            self._save_params(result)
            logger.info("Optimized params saved (improvement found)")
        else:
            logger.info("Default params are already optimal (no improvement)")

        return asdict(result)

    def _evaluate_params(self, params: Dict, data: pd.DataFrame) -> Dict:
        """Evaluate a parameter set by running backtest-like analysis."""
        from ..services.smart_money_analyzer import SmartMoneyAnalyzer

        analyzer = SmartMoneyAnalyzer(
            lookback_swing=int(params.get('swing_lookback', 5)),
            use_ml=False,
        )

        window_size = 100
        step = 20
        wins = 0
        losses = 0
        total_return = 0.0
        pattern_counts = {}

        max_start = len(data) - window_size - 20
        if max_start <= 0:
            return {'avg_win_rate': 0, 'avg_profit_factor': 0, 'total_patterns': 0}

        for start in range(0, max_start, step):
            window = data.iloc[start:start + window_size]
            forward = data.iloc[start + window_size:start + window_size + 20]

            if len(window) < window_size or len(forward) < 10:
                continue

            try:
                result = analyzer.analyze(window)
            except Exception:
                continue

            detection_price = float(window['close'].iloc[-1])
            forward_close = float(forward['close'].iloc[-1])

            # Count patterns and evaluate
            all_patterns = []
            if result.order_blocks:
                all_patterns.extend([('order_block', ob.type) for ob in result.order_blocks])
            if result.fair_value_gaps:
                all_patterns.extend([('fvg', fvg.type) for fvg in result.fair_value_gaps])
            if result.displacements:
                all_patterns.extend([('displacement', d['type']) for d in result.displacements])

            for ptype, direction in all_patterns:
                pattern_counts[ptype] = pattern_counts.get(ptype, 0) + 1
                if direction == 'bullish':
                    ret = (forward_close - detection_price) / detection_price * 100
                else:
                    ret = (detection_price - forward_close) / detection_price * 100

                total_return += ret
                if ret > 0:
                    wins += 1
                else:
                    losses += 1

        total = wins + losses
        gross_profit = max(total_return, 0) if total_return > 0 else 0
        gross_loss = abs(min(total_return, 0)) if total_return < 0 else 0

        return {
            'avg_win_rate': wins / total * 100 if total > 0 else 0,
            'avg_profit_factor': gross_profit / gross_loss if gross_loss > 0 else gross_profit,
            'total_patterns': sum(pattern_counts.values()),
            'total_signals': total,
            'wins': wins,
            'losses': losses,
            'pattern_counts': pattern_counts,
        }

    def _grid_search(self, train_data: pd.DataFrame) -> Dict:
        """
        Grid search over parameter space.

        Tests a coarse grid first, then refines around the best point.
        """
        best_score = -float('inf')
        best_params = dict(DEFAULT_PARAMS)

        # Coarse grid: test 5 values per parameter (key params only)
        key_params = [
            'fvg_min_gap_pct', 'ob_min_move_strength',
            'displacement_body_ratio', 'swing_lookback',
        ]

        # Generate parameter combinations (limited to avoid explosion)
        param_grid = {}
        for p in key_params:
            low, high = PARAM_RANGES[p]
            if p == 'swing_lookback':
                param_grid[p] = list(range(int(low), int(high) + 1))
            else:
                param_grid[p] = np.linspace(low, high, 5).tolist()

        # Test each key param independently (faster than full grid)
        for param_name in key_params:
            best_val = best_params[param_name]
            best_param_score = -float('inf')

            for val in param_grid[param_name]:
                test_params = dict(best_params)
                test_params[param_name] = val

                metrics = self._evaluate_params(test_params, train_data)
                # Score = win_rate * 0.6 + profit_factor * 0.4
                score = metrics.get('avg_win_rate', 0) * 0.6 + metrics.get('avg_profit_factor', 0) * 0.4

                if score > best_param_score:
                    best_param_score = score
                    best_val = val

            best_params[param_name] = best_val
            logger.info(f"Best {param_name}: {best_val}")

        # Round to reasonable precision
        for k, v in best_params.items():
            if isinstance(v, float):
                best_params[k] = round(v, 6)

        return best_params

    def _save_params(self, result: OptimizationResult) -> None:
        """Save optimized parameters to SQLite."""
        from ..database import Database
        db = Database()

        with db.get_connection() as conn:
            cursor = conn.cursor()
            for param_name, param_value in result.optimized_params.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO optimized_params
                    (symbol, timeframe, param_name, param_value,
                     validation_win_rate, validation_profit_factor,
                     train_period_start, train_period_end, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.symbol, result.timeframe,
                    param_name, float(param_value),
                    result.validation_win_rate,
                    result.validation_profit_factor,
                    result.train_period, result.test_period,
                    result.created_at,
                ))

    def get_best_params(self, symbol: str, timeframe: str = 'D1') -> Dict:
        """Load saved optimized parameters."""
        from ..database import Database
        db = Database()

        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT param_name, param_value FROM optimized_params WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe)
            )
            params = dict(DEFAULT_PARAMS)
            saved = {row[0]: row[1] for row in cursor.fetchall()}
            params.update(saved)
            return params


# Singleton
_optimizer_instance = None

def get_optimizer() -> ParameterOptimizer:
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = ParameterOptimizer()
    return _optimizer_instance
