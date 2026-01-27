"""
Performance Tracking System

Tracks and analyzes the performance of trading signals over time.
Enables continuous improvement of the ML model.

100% FREE - Uses JSON file storage
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """Record of a generated signal"""
    signal_id: str
    symbol: str
    timeframe: str
    direction: str  # 'BUY', 'SELL', 'WAIT'
    confidence: float
    entry_price: float
    entry_zone: Tuple[float, float]
    stop_loss: float
    take_profit: List[float]
    risk_reward: float
    factors: List[str]
    concepts_used: List[str]
    kill_zone: Optional[str]
    created_at: str

    # Outcome tracking
    status: str = "pending"  # pending, triggered, tp_hit, sl_hit, expired, cancelled
    actual_entry: Optional[float] = None
    actual_exit: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pips: Optional[float] = None
    pnl_percent: Optional[float] = None
    closed_at: Optional[str] = None


@dataclass
class DailyStats:
    """Daily performance statistics"""
    date: str
    total_signals: int
    buy_signals: int
    sell_signals: int
    triggered: int
    wins: int
    losses: int
    expired: int
    win_rate: float
    avg_pnl_pips: float
    total_pnl_pips: float
    best_signal: Optional[str]
    worst_signal: Optional[str]
    concepts_used: Dict[str, int]
    kill_zones_used: Dict[str, int]


class PerformanceTracker:
    """
    Tracks and analyzes signal performance.
    Enables model improvement through outcome feedback.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent.parent / "data"
        self.signals_path = self.data_dir / "signal_history.json"
        self.stats_path = self.data_dir / "performance_stats.json"

        # In-memory storage
        self.signals: Dict[str, SignalRecord] = {}
        self.daily_stats: Dict[str, DailyStats] = {}

        # Load existing data
        self.load()

    def load(self):
        """Load data from disk"""
        # Load signals
        if self.signals_path.exists():
            try:
                with open(self.signals_path) as f:
                    data = json.load(f)
                    for sig_id, sig_data in data.get('signals', {}).items():
                        self.signals[sig_id] = SignalRecord(**sig_data)
                logger.info(f"Loaded {len(self.signals)} signal records")
            except Exception as e:
                logger.warning(f"Error loading signals: {e}")

        # Load stats
        if self.stats_path.exists():
            try:
                with open(self.stats_path) as f:
                    data = json.load(f)
                    for date, stats_data in data.get('daily_stats', {}).items():
                        self.daily_stats[date] = DailyStats(**stats_data)
            except Exception as e:
                logger.warning(f"Error loading stats: {e}")

    def save(self):
        """Save data to disk"""
        # Save signals
        signals_data = {
            'signals': {sig_id: asdict(sig) for sig_id, sig in self.signals.items()},
            'last_saved': datetime.utcnow().isoformat()
        }
        with open(self.signals_path, 'w') as f:
            json.dump(signals_data, f, indent=2, default=str)

        # Save stats
        stats_data = {
            'daily_stats': {date: asdict(stats) for date, stats in self.daily_stats.items()},
            'last_saved': datetime.utcnow().isoformat()
        }
        with open(self.stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2, default=str)

        logger.info("Performance data saved")

    def record_signal(self, signal_data: Dict) -> str:
        """Record a new signal"""
        signal_id = signal_data.get('signal_id', str(datetime.utcnow().timestamp()))

        record = SignalRecord(
            signal_id=signal_id,
            symbol=signal_data.get('symbol', ''),
            timeframe=signal_data.get('timeframe', 'H1'),
            direction=signal_data.get('direction', 'WAIT'),
            confidence=signal_data.get('confidence', 0),
            entry_price=signal_data.get('entry_price', 0),
            entry_zone=tuple(signal_data.get('entry_zone', [0, 0])),
            stop_loss=signal_data.get('stop_loss', 0),
            take_profit=signal_data.get('take_profit', []),
            risk_reward=signal_data.get('risk_reward', 0),
            factors=signal_data.get('factors', []),
            concepts_used=signal_data.get('concepts', []),
            kill_zone=signal_data.get('kill_zone'),
            created_at=datetime.utcnow().isoformat(),
            status="pending"
        )

        self.signals[signal_id] = record
        self.save()

        return signal_id

    def update_signal_outcome(self, signal_id: str, outcome: Dict) -> bool:
        """Update signal with actual outcome"""
        if signal_id not in self.signals:
            logger.warning(f"Signal {signal_id} not found")
            return False

        signal = self.signals[signal_id]

        # Update fields
        signal.status = outcome.get('status', signal.status)
        signal.actual_entry = outcome.get('actual_entry')
        signal.actual_exit = outcome.get('actual_exit')
        signal.exit_reason = outcome.get('exit_reason')
        signal.pnl_pips = outcome.get('pnl_pips')
        signal.pnl_percent = outcome.get('pnl_percent')
        signal.closed_at = datetime.utcnow().isoformat()

        # Recalculate daily stats
        self._update_daily_stats(signal)

        self.save()
        return True

    def _update_daily_stats(self, signal: SignalRecord):
        """Update daily statistics based on signal outcome"""
        date = signal.created_at[:10]  # YYYY-MM-DD

        if date not in self.daily_stats:
            self.daily_stats[date] = DailyStats(
                date=date,
                total_signals=0,
                buy_signals=0,
                sell_signals=0,
                triggered=0,
                wins=0,
                losses=0,
                expired=0,
                win_rate=0,
                avg_pnl_pips=0,
                total_pnl_pips=0,
                best_signal=None,
                worst_signal=None,
                concepts_used={},
                kill_zones_used={}
            )

        stats = self.daily_stats[date]

        # Recompute stats from all signals for this day
        day_signals = [s for s in self.signals.values() if s.created_at.startswith(date)]

        stats.total_signals = len(day_signals)
        stats.buy_signals = len([s for s in day_signals if s.direction == 'BUY'])
        stats.sell_signals = len([s for s in day_signals if s.direction == 'SELL'])

        closed = [s for s in day_signals if s.status in ['tp_hit', 'sl_hit']]
        stats.triggered = len([s for s in day_signals if s.status != 'pending'])
        stats.wins = len([s for s in closed if s.pnl_pips and s.pnl_pips > 0])
        stats.losses = len([s for s in closed if s.pnl_pips and s.pnl_pips <= 0])
        stats.expired = len([s for s in day_signals if s.status == 'expired'])

        # Win rate
        if len(closed) > 0:
            stats.win_rate = stats.wins / len(closed)
        else:
            stats.win_rate = 0

        # PnL stats
        pnls = [s.pnl_pips for s in closed if s.pnl_pips is not None]
        if pnls:
            stats.total_pnl_pips = sum(pnls)
            stats.avg_pnl_pips = stats.total_pnl_pips / len(pnls)

            # Best/worst
            if pnls:
                best_idx = pnls.index(max(pnls))
                worst_idx = pnls.index(min(pnls))
                stats.best_signal = closed[best_idx].signal_id
                stats.worst_signal = closed[worst_idx].signal_id

        # Concept usage
        concept_counts = defaultdict(int)
        kz_counts = defaultdict(int)

        for s in day_signals:
            for concept in s.concepts_used:
                concept_counts[concept] += 1
            if s.kill_zone:
                kz_counts[s.kill_zone] += 1

        stats.concepts_used = dict(concept_counts)
        stats.kill_zones_used = dict(kz_counts)

    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary for last N days"""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        relevant_signals = [s for s in self.signals.values() if s.created_at >= cutoff]
        closed_signals = [s for s in relevant_signals if s.status in ['tp_hit', 'sl_hit']]

        # Calculate metrics
        total = len(relevant_signals)
        wins = len([s for s in closed_signals if s.pnl_pips and s.pnl_pips > 0])
        losses = len([s for s in closed_signals if s.pnl_pips and s.pnl_pips <= 0])

        pnls = [s.pnl_pips for s in closed_signals if s.pnl_pips is not None]
        total_pnl = sum(pnls) if pnls else 0
        avg_pnl = total_pnl / len(pnls) if pnls else 0

        # Win rate
        win_rate = wins / len(closed_signals) if closed_signals else 0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0) if pnls else 0
        gross_loss = abs(sum(p for p in pnls if p < 0)) if pnls else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # By direction
        buy_signals = [s for s in closed_signals if s.direction == 'BUY']
        sell_signals = [s for s in closed_signals if s.direction == 'SELL']

        buy_wins = len([s for s in buy_signals if s.pnl_pips and s.pnl_pips > 0])
        sell_wins = len([s for s in sell_signals if s.pnl_pips and s.pnl_pips > 0])

        # By confidence level
        high_conf = [s for s in closed_signals if s.confidence >= 0.75]
        med_conf = [s for s in closed_signals if 0.5 <= s.confidence < 0.75]
        low_conf = [s for s in closed_signals if s.confidence < 0.5]

        # By concept
        concept_performance = self._analyze_concept_performance(closed_signals)

        # By kill zone
        kz_performance = self._analyze_kill_zone_performance(closed_signals)

        return {
            'period_days': days,
            'total_signals': total,
            'closed_signals': len(closed_signals),
            'pending_signals': len(relevant_signals) - len(closed_signals),

            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 4),

            'total_pnl_pips': round(total_pnl, 2),
            'avg_pnl_pips': round(avg_pnl, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'N/A',

            'by_direction': {
                'BUY': {
                    'total': len(buy_signals),
                    'wins': buy_wins,
                    'win_rate': round(buy_wins / len(buy_signals), 4) if buy_signals else 0
                },
                'SELL': {
                    'total': len(sell_signals),
                    'wins': sell_wins,
                    'win_rate': round(sell_wins / len(sell_signals), 4) if sell_signals else 0
                }
            },

            'by_confidence': {
                'high_75+': {
                    'total': len(high_conf),
                    'win_rate': self._calc_win_rate(high_conf)
                },
                'medium_50-75': {
                    'total': len(med_conf),
                    'win_rate': self._calc_win_rate(med_conf)
                },
                'low_under_50': {
                    'total': len(low_conf),
                    'win_rate': self._calc_win_rate(low_conf)
                }
            },

            'by_concept': concept_performance,
            'by_kill_zone': kz_performance,

            'generated_at': datetime.utcnow().isoformat()
        }

    def _calc_win_rate(self, signals: List[SignalRecord]) -> float:
        """Calculate win rate for a list of signals"""
        if not signals:
            return 0
        wins = len([s for s in signals if s.pnl_pips and s.pnl_pips > 0])
        return round(wins / len(signals), 4)

    def _analyze_concept_performance(self, signals: List[SignalRecord]) -> Dict:
        """Analyze performance by Smart Money concept"""
        concept_results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})

        for signal in signals:
            is_win = signal.pnl_pips and signal.pnl_pips > 0
            pnl = signal.pnl_pips or 0

            for concept in signal.concepts_used:
                if is_win:
                    concept_results[concept]['wins'] += 1
                else:
                    concept_results[concept]['losses'] += 1
                concept_results[concept]['total_pnl'] += pnl

        # Calculate win rates
        result = {}
        for concept, data in concept_results.items():
            total = data['wins'] + data['losses']
            result[concept] = {
                'total': total,
                'wins': data['wins'],
                'win_rate': round(data['wins'] / total, 4) if total > 0 else 0,
                'total_pnl': round(data['total_pnl'], 2)
            }

        return dict(sorted(result.items(), key=lambda x: x[1]['total'], reverse=True))

    def _analyze_kill_zone_performance(self, signals: List[SignalRecord]) -> Dict:
        """Analyze performance by kill zone"""
        kz_results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})

        for signal in signals:
            kz = signal.kill_zone or 'No Kill Zone'
            is_win = signal.pnl_pips and signal.pnl_pips > 0
            pnl = signal.pnl_pips or 0

            if is_win:
                kz_results[kz]['wins'] += 1
            else:
                kz_results[kz]['losses'] += 1
            kz_results[kz]['total_pnl'] += pnl

        result = {}
        for kz, data in kz_results.items():
            total = data['wins'] + data['losses']
            result[kz] = {
                'total': total,
                'wins': data['wins'],
                'win_rate': round(data['wins'] / total, 4) if total > 0 else 0,
                'total_pnl': round(data['total_pnl'], 2)
            }

        return result

    def get_model_feedback(self) -> Dict:
        """
        Generate feedback for model improvement.
        Identifies what's working and what's not.
        """
        summary = self.get_performance_summary(days=30)

        recommendations = []

        # Win rate feedback
        if summary['win_rate'] < 0.5:
            recommendations.append({
                'type': 'warning',
                'message': f"Win rate is below 50% ({summary['win_rate']:.1%}). Consider raising confidence threshold."
            })

        # Concept feedback
        for concept, data in summary['by_concept'].items():
            if data['total'] >= 5:  # Minimum sample size
                if data['win_rate'] < 0.4:
                    recommendations.append({
                        'type': 'reduce_weight',
                        'concept': concept,
                        'message': f"Concept '{concept}' has low win rate ({data['win_rate']:.1%}). Consider reducing weight."
                    })
                elif data['win_rate'] > 0.7:
                    recommendations.append({
                        'type': 'increase_weight',
                        'concept': concept,
                        'message': f"Concept '{concept}' has high win rate ({data['win_rate']:.1%}). Consider increasing weight."
                    })

        # Kill zone feedback
        for kz, data in summary['by_kill_zone'].items():
            if data['total'] >= 5:
                if data['win_rate'] > 0.6:
                    recommendations.append({
                        'type': 'info',
                        'message': f"'{kz}' kill zone is performing well ({data['win_rate']:.1%} win rate)"
                    })

        # Direction bias
        buy_wr = summary['by_direction']['BUY']['win_rate']
        sell_wr = summary['by_direction']['SELL']['win_rate']

        if abs(buy_wr - sell_wr) > 0.15:
            if buy_wr > sell_wr:
                recommendations.append({
                    'type': 'info',
                    'message': f"BUY signals outperforming SELL ({buy_wr:.1%} vs {sell_wr:.1%}). Market may be bullish."
                })
            else:
                recommendations.append({
                    'type': 'info',
                    'message': f"SELL signals outperforming BUY ({sell_wr:.1%} vs {buy_wr:.1%}). Market may be bearish."
                })

        return {
            'summary': summary,
            'recommendations': recommendations,
            'generated_at': datetime.utcnow().isoformat()
        }

    def export_to_csv(self, filepath: str = None) -> str:
        """Export signal history to CSV"""
        if filepath is None:
            filepath = str(self.data_dir / "signal_history_export.csv")

        try:
            import pandas as pd

            records = [asdict(s) for s in self.signals.values()]
            df = pd.DataFrame(records)
            df.to_csv(filepath, index=False)

            return filepath
        except ImportError:
            logger.warning("pandas not available for CSV export")
            return ""

    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """Get most recent signals"""
        sorted_signals = sorted(
            self.signals.values(),
            key=lambda x: x.created_at,
            reverse=True
        )

        return [asdict(s) for s in sorted_signals[:limit]]

    def get_signal(self, signal_id: str) -> Optional[Dict]:
        """Get a specific signal"""
        if signal_id in self.signals:
            return asdict(self.signals[signal_id])
        return None


# Singleton instance
_tracker = None


def get_tracker(data_dir: str = None) -> PerformanceTracker:
    """Get or create performance tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker(data_dir)
    return _tracker


if __name__ == "__main__":
    # Test
    tracker = PerformanceTracker()

    print("=" * 50)
    print("PERFORMANCE TRACKER TEST")
    print("=" * 50)

    # Record test signal
    signal_id = tracker.record_signal({
        'symbol': 'EURUSD',
        'timeframe': 'H1',
        'direction': 'BUY',
        'confidence': 0.75,
        'entry_price': 1.0850,
        'entry_zone': [1.0845, 1.0855],
        'stop_loss': 1.0820,
        'take_profit': [1.0900, 1.0950],
        'risk_reward': 2.5,
        'factors': ['order_block', 'fvg', 'premium_discount'],
        'concepts': ['order_block', 'fair_value_gap', 'market_structure'],
        'kill_zone': 'London Kill Zone'
    })

    print(f"\nRecorded signal: {signal_id}")

    # Update with outcome
    tracker.update_signal_outcome(signal_id, {
        'status': 'tp_hit',
        'actual_entry': 1.0850,
        'actual_exit': 1.0900,
        'exit_reason': 'TP1 hit',
        'pnl_pips': 50,
        'pnl_percent': 0.46
    })

    print(f"Updated signal outcome")

    # Get summary
    summary = tracker.get_performance_summary(days=30)
    print(f"\nPerformance Summary:")
    print(f"  Total Signals: {summary['total_signals']}")
    print(f"  Win Rate: {summary['win_rate']:.1%}")
    print(f"  Total PnL: {summary['total_pnl_pips']} pips")

    # Get feedback
    feedback = tracker.get_model_feedback()
    print(f"\nModel Feedback:")
    for rec in feedback['recommendations']:
        print(f"  [{rec['type']}] {rec['message']}")
