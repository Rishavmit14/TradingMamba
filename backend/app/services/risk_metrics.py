"""
Risk Metrics Engine - Institutional-Grade Risk Analysis

This module provides hedge fund-level risk metrics using Empyrical (FREE).
It calculates the same metrics used by professional traders and risk managers.

Key Metrics:
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk only)
- Calmar Ratio (return vs max drawdown)
- Maximum Drawdown (worst case scenario)
- Value at Risk (VaR)
- Conditional VaR (Expected Shortfall)

100% FREE - Uses Empyrical library.

Install: pip install empyrical
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Try to import empyrical
try:
    import empyrical as ep
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False
    print("Empyrical not installed. Install with: pip install empyrical")

# Try to import pyfolio for tearsheets
try:
    import pyfolio as pf
    PYFOLIO_AVAILABLE = True
except ImportError:
    PYFOLIO_AVAILABLE = False


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"           # Conservative, safe
    MODERATE = "moderate" # Balanced risk/reward
    HIGH = "high"         # Aggressive
    EXTREME = "extreme"   # Dangerous


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a strategy/pattern"""
    # Identification
    strategy_name: str
    period_start: str
    period_end: str
    total_trades: int

    # Return Metrics
    total_return: float
    annualized_return: float
    cumulative_return: float

    # Risk Metrics
    volatility: float              # Annualized standard deviation
    downside_volatility: float     # Downside deviation only
    max_drawdown: float            # Maximum peak-to-trough decline
    max_drawdown_duration_days: int

    # Risk-Adjusted Returns
    sharpe_ratio: float            # (Return - Rf) / Volatility
    sortino_ratio: float           # (Return - Rf) / Downside Vol
    calmar_ratio: float            # Annual Return / Max DD
    omega_ratio: float             # Probability-weighted ratio

    # Tail Risk
    var_95: float                  # 95% Value at Risk
    var_99: float                  # 99% Value at Risk
    cvar_95: float                 # Conditional VaR (Expected Shortfall)

    # Stability
    stability: float               # R-squared of cumulative returns
    tail_ratio: float              # Right tail / Left tail

    # Win/Loss Analysis
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float

    # Risk Assessment
    risk_level: str
    risk_score: float              # 0-100 score

    # Timestamp
    calculated_at: str = ""

    def __post_init__(self):
        if not self.calculated_at:
            self.calculated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)

    def is_acceptable_risk(self) -> bool:
        """Check if risk metrics are within acceptable bounds"""
        return (
            self.sharpe_ratio > 0.5 and
            self.max_drawdown > -0.30 and  # Max 30% drawdown
            self.win_rate > 0.40 and
            self.expectancy > 0
        )


@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation based on risk"""
    symbol: str
    account_size: float
    risk_per_trade_pct: float
    recommended_position_size: float
    recommended_shares: int
    stop_loss_price: float
    max_loss_amount: float
    risk_reward_ratio: float
    kelly_criterion: float
    optimal_f: float


class RiskMetricsEngine:
    """
    Calculates institutional-grade risk metrics.

    This is what separates a retail trader from a professional:
    - Retail: "I made 50% this year!"
    - Professional: "I made 50% with 1.8 Sharpe, 15% max DD, and 1.2 Sortino"
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent.parent / "data"
        self.metrics_dir = self.data_dir / "risk_metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Risk-free rate (annual)
        self.risk_free_rate = 0.05  # 5% (current T-bill rate)

    def calculate_metrics(
        self,
        returns: List[float],
        strategy_name: str = "unnamed",
        period_start: str = None,
        period_end: str = None
    ) -> Optional[RiskMetrics]:
        """
        Calculate comprehensive risk metrics from a series of returns.

        Args:
            returns: List of period returns (e.g., daily returns as decimals)
            strategy_name: Name for identification
            period_start: Start date
            period_end: End date

        Returns:
            RiskMetrics object with all calculations
        """
        if not returns or len(returns) < 10:
            print("Need at least 10 returns to calculate meaningful metrics")
            return None

        returns_arr = np.array(returns)
        returns_series = pd.Series(returns_arr)

        # Basic return metrics
        total_return = float(np.prod(1 + returns_arr) - 1)
        cumulative_return = total_return

        # Calculate periods per year (assume daily if not specified)
        periods_per_year = 252

        # Annualized return
        n_periods = len(returns)
        annualized_return = float((1 + total_return) ** (periods_per_year / n_periods) - 1)

        # Volatility
        volatility = float(np.std(returns_arr) * np.sqrt(periods_per_year))
        downside_returns = returns_arr[returns_arr < 0]
        downside_volatility = float(np.std(downside_returns) * np.sqrt(periods_per_year)) if len(downside_returns) > 0 else 0

        # Maximum Drawdown
        max_dd, max_dd_duration = self._calculate_max_drawdown(returns_arr)

        # Risk-Adjusted Returns
        if EMPYRICAL_AVAILABLE:
            sharpe_ratio = float(ep.sharpe_ratio(returns_series, risk_free=self.risk_free_rate / periods_per_year))
            sortino_ratio = float(ep.sortino_ratio(returns_series, risk_free=self.risk_free_rate / periods_per_year))
            calmar_ratio = float(ep.calmar_ratio(returns_series)) if max_dd != 0 else 0
            omega_ratio = float(ep.omega_ratio(returns_series, risk_free=self.risk_free_rate / periods_per_year))
            stability = float(ep.stability_of_timeseries(returns_series))
            tail_ratio = float(ep.tail_ratio(returns_series))
        else:
            sharpe_ratio = self._calculate_sharpe_manual(returns_arr, periods_per_year)
            sortino_ratio = self._calculate_sortino_manual(returns_arr, periods_per_year)
            calmar_ratio = annualized_return / abs(max_dd) if max_dd != 0 else 0
            omega_ratio = self._calculate_omega_manual(returns_arr)
            stability = self._calculate_stability_manual(returns_arr)
            tail_ratio = self._calculate_tail_ratio_manual(returns_arr)

        # Value at Risk
        var_95 = float(np.percentile(returns_arr, 5))
        var_99 = float(np.percentile(returns_arr, 1))
        cvar_95 = float(np.mean(returns_arr[returns_arr <= var_95]))

        # Win/Loss Analysis
        wins = returns_arr[returns_arr > 0]
        losses = returns_arr[returns_arr <= 0]
        win_rate = len(wins) / len(returns_arr) if len(returns_arr) > 0 else 0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
        profit_factor = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else float('inf')
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

        # Risk Assessment
        risk_level, risk_score = self._assess_risk_level(
            sharpe_ratio, max_dd, volatility, win_rate, expectancy
        )

        return RiskMetrics(
            strategy_name=strategy_name,
            period_start=period_start or datetime.now().strftime("%Y-%m-%d"),
            period_end=period_end or datetime.now().strftime("%Y-%m-%d"),
            total_trades=len(returns),
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return,
            volatility=volatility,
            downside_volatility=downside_volatility,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            stability=stability,
            tail_ratio=tail_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_win=avg_win,
            avg_loss=avg_loss,
            risk_level=risk_level.value,
            risk_score=risk_score
        )

    def _calculate_max_drawdown(self, returns: np.ndarray) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        max_dd = float(np.min(drawdowns))

        # Calculate duration
        in_drawdown = drawdowns < 0
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_dd, max_duration

    def _calculate_sharpe_manual(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Manual Sharpe ratio calculation"""
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        if np.std(excess_returns) == 0:
            return 0
        return float(np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns))

    def _calculate_sortino_manual(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Manual Sortino ratio calculation"""
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        downside = returns[returns < 0]
        if len(downside) == 0 or np.std(downside) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0
        return float(np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(downside))

    def _calculate_omega_manual(self, returns: np.ndarray, threshold: float = 0) -> float:
        """Manual Omega ratio calculation"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        if np.sum(losses) == 0:
            return float('inf')
        return float(np.sum(gains) / np.sum(losses))

    def _calculate_stability_manual(self, returns: np.ndarray) -> float:
        """Manual stability calculation (R-squared of cumulative returns)"""
        cumulative = np.cumprod(1 + returns)
        x = np.arange(len(cumulative))
        correlation = np.corrcoef(x, cumulative)[0, 1]
        return float(correlation ** 2) if not np.isnan(correlation) else 0

    def _calculate_tail_ratio_manual(self, returns: np.ndarray) -> float:
        """Manual tail ratio calculation"""
        right_tail = np.percentile(returns, 95)
        left_tail = abs(np.percentile(returns, 5))
        if left_tail == 0:
            return float('inf')
        return float(right_tail / left_tail)

    def _assess_risk_level(
        self,
        sharpe: float,
        max_dd: float,
        volatility: float,
        win_rate: float,
        expectancy: float
    ) -> Tuple[RiskLevel, float]:
        """Assess overall risk level and score"""
        score = 50  # Start at neutral

        # Sharpe contribution (±20 points)
        if sharpe > 2:
            score += 20
        elif sharpe > 1:
            score += 10
        elif sharpe > 0.5:
            score += 5
        elif sharpe < 0:
            score -= 15

        # Max Drawdown contribution (±20 points)
        if max_dd > -0.10:
            score += 20
        elif max_dd > -0.20:
            score += 10
        elif max_dd > -0.30:
            score -= 5
        elif max_dd > -0.50:
            score -= 15
        else:
            score -= 25

        # Win rate contribution (±15 points)
        if win_rate > 0.60:
            score += 15
        elif win_rate > 0.50:
            score += 10
        elif win_rate > 0.40:
            score += 0
        else:
            score -= 10

        # Expectancy contribution (±15 points)
        if expectancy > 0.02:
            score += 15
        elif expectancy > 0:
            score += 5
        else:
            score -= 15

        # Volatility contribution (±10 points)
        if volatility < 0.15:
            score += 10
        elif volatility < 0.25:
            score += 5
        elif volatility > 0.40:
            score -= 10

        # Clamp score
        score = max(0, min(100, score))

        # Determine risk level
        if score >= 75:
            level = RiskLevel.LOW
        elif score >= 50:
            level = RiskLevel.MODERATE
        elif score >= 25:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.EXTREME

        return level, score

    def calculate_position_size(
        self,
        account_size: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        win_rate: float = 0.5,
        risk_per_trade_pct: float = 0.02  # 2% risk per trade
    ) -> PositionSizeRecommendation:
        """
        Calculate optimal position size based on risk management.

        Uses multiple methods:
        1. Fixed percentage risk
        2. Kelly Criterion
        3. Optimal f
        """
        # Calculate risk and reward per share
        risk_per_share = abs(entry_price - stop_loss_price)
        reward_per_share = abs(take_profit_price - entry_price)
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

        # Method 1: Fixed percentage risk
        max_loss_amount = account_size * risk_per_trade_pct
        fixed_pct_shares = int(max_loss_amount / risk_per_share) if risk_per_share > 0 else 0

        # Method 2: Kelly Criterion
        # Kelly % = W - [(1-W) / R] where W = win rate, R = win/loss ratio
        if risk_reward_ratio > 0:
            kelly = win_rate - ((1 - win_rate) / risk_reward_ratio)
            kelly = max(0, min(kelly, 0.25))  # Cap at 25% for safety
        else:
            kelly = 0

        kelly_shares = int((account_size * kelly) / entry_price)

        # Method 3: Optimal f (simplified)
        # Use half-Kelly for safety
        optimal_f = kelly / 2
        optimal_f_shares = int((account_size * optimal_f) / entry_price)

        # Use the most conservative of the three
        recommended_shares = min(
            fixed_pct_shares,
            kelly_shares if kelly_shares > 0 else fixed_pct_shares,
            optimal_f_shares if optimal_f_shares > 0 else fixed_pct_shares
        )

        recommended_position_size = recommended_shares * entry_price

        return PositionSizeRecommendation(
            symbol="",
            account_size=account_size,
            risk_per_trade_pct=risk_per_trade_pct,
            recommended_position_size=recommended_position_size,
            recommended_shares=recommended_shares,
            stop_loss_price=stop_loss_price,
            max_loss_amount=recommended_shares * risk_per_share,
            risk_reward_ratio=risk_reward_ratio,
            kelly_criterion=kelly,
            optimal_f=optimal_f
        )

    def compare_strategies(self, metrics_list: List[RiskMetrics]) -> pd.DataFrame:
        """Compare multiple strategies side by side"""
        if not metrics_list:
            return pd.DataFrame()

        data = []
        for m in metrics_list:
            data.append({
                'Strategy': m.strategy_name,
                'Return': f"{m.total_return:.1%}",
                'Sharpe': f"{m.sharpe_ratio:.2f}",
                'Sortino': f"{m.sortino_ratio:.2f}",
                'Max DD': f"{m.max_drawdown:.1%}",
                'Win Rate': f"{m.win_rate:.1%}",
                'Expectancy': f"{m.expectancy:.4f}",
                'Risk Level': m.risk_level,
                'Score': f"{m.risk_score:.0f}/100"
            })

        return pd.DataFrame(data)

    def save_metrics(self, metrics: RiskMetrics) -> str:
        """Save metrics to file"""
        filename = f"{metrics.strategy_name}_{metrics.calculated_at[:10]}.json"
        filepath = self.metrics_dir / filename

        with open(filepath, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        return str(filepath)

    def load_metrics(self, strategy_name: str = None) -> List[RiskMetrics]:
        """Load saved metrics"""
        results = []

        for filepath in self.metrics_dir.glob("*.json"):
            if strategy_name and strategy_name not in filepath.name:
                continue

            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(RiskMetrics(**data))

        return results


# Singleton instance
_risk_engine: Optional[RiskMetricsEngine] = None


def get_risk_engine() -> RiskMetricsEngine:
    """Get or create the singleton risk engine instance"""
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskMetricsEngine()
    return _risk_engine


def calculate_pattern_risk(
    returns: List[float],
    pattern_type: str
) -> Optional[RiskMetrics]:
    """
    Calculate risk metrics for a specific pattern type.

    Args:
        returns: List of returns from pattern trades
        pattern_type: Pattern type name

    Returns:
        RiskMetrics object
    """
    engine = get_risk_engine()
    return engine.calculate_metrics(returns, strategy_name=pattern_type)


def get_position_size(
    account_size: float,
    entry: float,
    stop_loss: float,
    take_profit: float,
    win_rate: float = 0.5
) -> PositionSizeRecommendation:
    """
    Get recommended position size based on risk management.
    """
    engine = get_risk_engine()
    return engine.calculate_position_size(
        account_size=account_size,
        entry_price=entry,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        win_rate=win_rate
    )


# Test function
def test_risk_metrics():
    """Test the risk metrics engine"""
    print("=" * 60)
    print("RISK METRICS ENGINE TEST")
    print("=" * 60)

    engine = get_risk_engine()

    # Generate sample returns (simulating a trading strategy)
    np.random.seed(42)
    # Mix of winning and losing trades
    returns = []
    for _ in range(100):
        if np.random.random() < 0.6:  # 60% win rate
            returns.append(np.random.uniform(0.005, 0.03))  # Wins
        else:
            returns.append(np.random.uniform(-0.02, -0.005))  # Losses

    print("\n📊 CALCULATING RISK METRICS...")
    metrics = engine.calculate_metrics(returns, strategy_name="FVG_Strategy")

    if metrics:
        print(f"\n{'='*40}")
        print(f"📈 RETURN METRICS")
        print(f"{'='*40}")
        print(f"  Total Return: {metrics.total_return:.2%}")
        print(f"  Annualized Return: {metrics.annualized_return:.2%}")
        print(f"  Volatility: {metrics.volatility:.2%}")

        print(f"\n{'='*40}")
        print(f"📉 RISK METRICS")
        print(f"{'='*40}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"  Drawdown Duration: {metrics.max_drawdown_duration_days} days")
        print(f"  VaR (95%): {metrics.var_95:.2%}")
        print(f"  CVaR (95%): {metrics.cvar_95:.2%}")

        print(f"\n{'='*40}")
        print(f"⚖️ RISK-ADJUSTED RETURNS")
        print(f"{'='*40}")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
        print(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")
        print(f"  Omega Ratio: {metrics.omega_ratio:.2f}")

        print(f"\n{'='*40}")
        print(f"🎯 TRADE ANALYSIS")
        print(f"{'='*40}")
        print(f"  Win Rate: {metrics.win_rate:.1%}")
        print(f"  Profit Factor: {metrics.profit_factor:.2f}")
        print(f"  Expectancy: {metrics.expectancy:.4f}")
        print(f"  Avg Win: {metrics.avg_win:.2%}")
        print(f"  Avg Loss: {metrics.avg_loss:.2%}")

        print(f"\n{'='*40}")
        print(f"🚦 RISK ASSESSMENT")
        print(f"{'='*40}")
        print(f"  Risk Level: {metrics.risk_level.upper()}")
        print(f"  Risk Score: {metrics.risk_score:.0f}/100")
        print(f"  Acceptable Risk: {'✅ YES' if metrics.is_acceptable_risk() else '❌ NO'}")

        # Save metrics
        filepath = engine.save_metrics(metrics)
        print(f"\nMetrics saved to: {filepath}")

    # Test position sizing
    print(f"\n{'='*40}")
    print(f"💰 POSITION SIZING")
    print(f"{'='*40}")

    position = engine.calculate_position_size(
        account_size=100000,
        entry_price=500,
        stop_loss_price=495,
        take_profit_price=510,
        win_rate=0.6
    )

    print(f"  Account Size: ${position.account_size:,.2f}")
    print(f"  Risk per Trade: {position.risk_per_trade_pct:.1%}")
    print(f"  Recommended Shares: {position.recommended_shares}")
    print(f"  Position Size: ${position.recommended_position_size:,.2f}")
    print(f"  Max Loss: ${position.max_loss_amount:,.2f}")
    print(f"  Risk/Reward: {position.risk_reward_ratio:.2f}")
    print(f"  Kelly Criterion: {position.kelly_criterion:.2%}")
    print(f"  Optimal f: {position.optimal_f:.2%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_risk_metrics()
