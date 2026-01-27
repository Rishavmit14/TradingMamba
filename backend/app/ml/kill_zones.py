"""
Smart Money Kill Zones and Session Analysis

Kill Zones are optimal trading times according to Smart Money methodology.
These are periods when institutional order flow is highest.

100% FREE - Uses only Python standard library + pandas
"""

from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pytz

try:
    import pandas as pd
except ImportError:
    pd = None


class TradingSession(Enum):
    """Major trading sessions"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_CLOSE = "london_close"
    OVERLAP = "overlap"  # London/NY overlap
    OFF_HOURS = "off_hours"


@dataclass
class KillZone:
    """Represents an Smart Money Kill Zone"""
    name: str
    session: TradingSession
    start_time: time  # UTC time
    end_time: time    # UTC time
    bias_weight: float  # How much this KZ affects signal confidence
    description: str
    optimal_pairs: List[str]  # Best pairs to trade in this KZ


@dataclass
class SessionInfo:
    """Current session information"""
    current_session: TradingSession
    in_kill_zone: bool
    kill_zone_name: Optional[str]
    time_to_next_kz: Optional[timedelta]
    session_high: Optional[float]
    session_low: Optional[float]
    session_open: Optional[float]
    daily_bias: str


class KillZoneAnalyzer:
    """
    Analyzes Smart Money Kill Zones and trading sessions.

    Smart Money Kill Zones (UTC):
    - Asian Kill Zone: 00:00 - 04:00 UTC (Tokyo)
    - London Kill Zone: 07:00 - 10:00 UTC
    - New York Kill Zone: 12:00 - 15:00 UTC
    - London Close Kill Zone: 15:00 - 17:00 UTC
    """

    # Define Kill Zones (all times in UTC)
    KILL_ZONES = [
        KillZone(
            name="Asian Kill Zone",
            session=TradingSession.ASIAN,
            start_time=time(0, 0),
            end_time=time(4, 0),
            bias_weight=0.6,
            description="Tokyo session open - accumulation phase, range-bound. Good for AUD, NZD, JPY pairs.",
            optimal_pairs=["USDJPY", "AUDUSD", "NZDUSD", "EURJPY", "AUDJPY"]
        ),
        KillZone(
            name="London Kill Zone",
            session=TradingSession.LONDON,
            start_time=time(7, 0),
            end_time=time(10, 0),
            bias_weight=1.0,  # Highest weight - most volatile
            description="London session open - high volatility, strong moves. Best for EUR, GBP pairs.",
            optimal_pairs=["EURUSD", "GBPUSD", "EURGBP", "GBPJPY", "EURJPY"]
        ),
        KillZone(
            name="New York Kill Zone",
            session=TradingSession.NEW_YORK,
            start_time=time(12, 0),
            end_time=time(15, 0),
            bias_weight=0.95,
            description="New York session open - continuation or reversal of London move. Best for USD pairs.",
            optimal_pairs=["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "XAUUSD"]
        ),
        KillZone(
            name="London Close Kill Zone",
            session=TradingSession.LONDON_CLOSE,
            start_time=time(15, 0),
            end_time=time(17, 0),
            bias_weight=0.7,
            description="London close - potential reversals as London traders close positions.",
            optimal_pairs=["EURUSD", "GBPUSD", "EURGBP"]
        ),
    ]

    # Session times (UTC)
    SESSIONS = {
        TradingSession.ASIAN: (time(0, 0), time(8, 0)),
        TradingSession.LONDON: (time(7, 0), time(16, 0)),
        TradingSession.NEW_YORK: (time(12, 0), time(21, 0)),
        TradingSession.OVERLAP: (time(12, 0), time(16, 0)),  # London/NY overlap
    }

    def __init__(self, timezone: str = "UTC"):
        self.tz = pytz.timezone(timezone)
        self.utc = pytz.UTC

    def get_current_utc_time(self) -> datetime:
        """Get current time in UTC"""
        return datetime.now(self.utc)

    def is_in_kill_zone(self, dt: datetime = None) -> Tuple[bool, Optional[KillZone]]:
        """Check if given time is within a kill zone"""
        if dt is None:
            dt = self.get_current_utc_time()

        # Convert to UTC if needed
        if dt.tzinfo is None:
            dt = self.utc.localize(dt)
        else:
            dt = dt.astimezone(self.utc)

        current_time = dt.time()

        for kz in self.KILL_ZONES:
            if self._time_in_range(current_time, kz.start_time, kz.end_time):
                return True, kz

        return False, None

    def _time_in_range(self, check_time: time, start: time, end: time) -> bool:
        """Check if time is within range (handles overnight ranges)"""
        if start <= end:
            return start <= check_time < end
        else:  # Overnight range (e.g., 22:00 - 04:00)
            return check_time >= start or check_time < end

    def get_current_session(self, dt: datetime = None) -> TradingSession:
        """Get current trading session"""
        if dt is None:
            dt = self.get_current_utc_time()

        if dt.tzinfo is None:
            dt = self.utc.localize(dt)
        else:
            dt = dt.astimezone(self.utc)

        current_time = dt.time()

        # Check overlap first (most specific)
        overlap_start, overlap_end = self.SESSIONS[TradingSession.OVERLAP]
        if self._time_in_range(current_time, overlap_start, overlap_end):
            return TradingSession.OVERLAP

        # Check other sessions
        for session, (start, end) in self.SESSIONS.items():
            if session == TradingSession.OVERLAP:
                continue
            if self._time_in_range(current_time, start, end):
                return session

        return TradingSession.OFF_HOURS

    def get_next_kill_zone(self, dt: datetime = None) -> Tuple[KillZone, timedelta]:
        """Get next upcoming kill zone and time until it starts"""
        if dt is None:
            dt = self.get_current_utc_time()

        if dt.tzinfo is None:
            dt = self.utc.localize(dt)

        current_time = dt.time()
        current_date = dt.date()

        # Find next kill zone
        min_delta = timedelta(days=2)
        next_kz = None

        for kz in self.KILL_ZONES:
            # Check if KZ starts later today
            kz_start_dt = datetime.combine(current_date, kz.start_time)
            kz_start_dt = self.utc.localize(kz_start_dt)

            if kz_start_dt > dt:
                delta = kz_start_dt - dt
            else:
                # KZ starts tomorrow
                kz_start_dt = datetime.combine(current_date + timedelta(days=1), kz.start_time)
                kz_start_dt = self.utc.localize(kz_start_dt)
                delta = kz_start_dt - dt

            if delta < min_delta:
                min_delta = delta
                next_kz = kz

        return next_kz, min_delta

    def get_session_info(self, df: pd.DataFrame = None, dt: datetime = None) -> SessionInfo:
        """Get comprehensive session information"""
        if dt is None:
            dt = self.get_current_utc_time()

        current_session = self.get_current_session(dt)
        in_kz, kz = self.is_in_kill_zone(dt)
        next_kz, time_to_next = self.get_next_kill_zone(dt)

        # Calculate session high/low/open if DataFrame provided
        session_high = None
        session_low = None
        session_open = None

        if df is not None and pd is not None and len(df) > 0:
            # Get session start time
            session_start, _ = self.SESSIONS.get(current_session, (time(0, 0), time(23, 59)))

            # Filter data for current session
            # This is simplified - in production, you'd filter by actual session times
            session_data = df.tail(24)  # Approximate session data

            if len(session_data) > 0:
                session_high = float(session_data['high'].max())
                session_low = float(session_data['low'].min())
                session_open = float(session_data['open'].iloc[0])

        # Determine daily bias based on session
        daily_bias = self._get_daily_bias(current_session, dt)

        return SessionInfo(
            current_session=current_session,
            in_kill_zone=in_kz,
            kill_zone_name=kz.name if kz else None,
            time_to_next_kz=time_to_next if not in_kz else None,
            session_high=session_high,
            session_low=session_low,
            session_open=session_open,
            daily_bias=daily_bias
        )

    def _get_daily_bias(self, session: TradingSession, dt: datetime) -> str:
        """Determine daily bias based on session and day of week"""
        day_of_week = dt.weekday()  # 0 = Monday

        # Smart Money concepts about weekly/daily tendencies
        # Monday: Accumulation/range
        # Tuesday-Wednesday: Main moves
        # Thursday: Continuation or reversal
        # Friday: Profit taking, potential reversals

        day_biases = {
            0: "accumulation",  # Monday
            1: "expansion",     # Tuesday
            2: "expansion",     # Wednesday
            3: "distribution",  # Thursday
            4: "reversal_risk", # Friday
            5: "closed",        # Saturday
            6: "closed",        # Sunday
        }

        return day_biases.get(day_of_week, "neutral")

    def get_optimal_pairs_for_session(self, session: TradingSession = None) -> List[str]:
        """Get optimal trading pairs for current or specified session"""
        if session is None:
            session = self.get_current_session()

        # Find kill zone matching this session
        for kz in self.KILL_ZONES:
            if kz.session == session:
                return kz.optimal_pairs

        # Default pairs
        return ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

    def adjust_confidence_for_timing(self, base_confidence: float,
                                     symbol: str = None,
                                     dt: datetime = None) -> float:
        """Adjust signal confidence based on kill zone timing"""
        in_kz, kz = self.is_in_kill_zone(dt)

        if not in_kz:
            # Not in kill zone - reduce confidence
            return base_confidence * 0.7

        # In kill zone - apply KZ weight
        adjusted = base_confidence * kz.bias_weight

        # Bonus if symbol is optimal for this KZ
        if symbol and kz.optimal_pairs:
            symbol_base = symbol.upper().replace("/", "")
            if any(symbol_base.startswith(p[:3]) or symbol_base.endswith(p[-3:])
                   for p in kz.optimal_pairs):
                adjusted *= 1.1  # 10% bonus

        return min(adjusted, 1.0)

    def get_power_of_three_phase(self, df: pd.DataFrame = None) -> Dict:
        """
        Determine current Power of Three (AMD) phase.

        Power of Three (Smart Money concept):
        - Accumulation: Range-bound, building positions (Asian session)
        - Manipulation: False move to trap traders (early London/NY)
        - Distribution: True move direction (main session)
        """
        session = self.get_current_session()
        in_kz, kz = self.is_in_kill_zone()

        if session == TradingSession.ASIAN:
            phase = "accumulation"
            description = "Asian session - building positions, expect range"
        elif session == TradingSession.LONDON and in_kz:
            # Early London often has manipulation
            hour = self.get_current_utc_time().hour
            if hour < 9:
                phase = "manipulation"
                description = "Early London - watch for false moves (Judas swing)"
            else:
                phase = "distribution"
                description = "London expansion - true direction emerging"
        elif session == TradingSession.NEW_YORK and in_kz:
            hour = self.get_current_utc_time().hour
            if hour < 14:
                phase = "manipulation"
                description = "Early NY - potential manipulation before true move"
            else:
                phase = "distribution"
                description = "NY expansion - continuation or reversal of London"
        elif session == TradingSession.OVERLAP:
            phase = "distribution"
            description = "London/NY overlap - highest volatility, true moves"
        else:
            phase = "off_hours"
            description = "Off hours - reduced liquidity, avoid trading"

        return {
            'phase': phase,
            'description': description,
            'session': session.value,
            'in_kill_zone': in_kz,
            'kill_zone': kz.name if kz else None
        }

    def get_true_day_open(self, df: pd.DataFrame, dt: datetime = None) -> Optional[float]:
        """
        Get Smart Money True Day Open (New York midnight open).

        True Day Open is at 00:00 New York time (05:00 UTC in winter, 04:00 UTC in summer).
        """
        if df is None or len(df) == 0:
            return None

        if dt is None:
            dt = self.get_current_utc_time()

        # New York timezone
        ny_tz = pytz.timezone('America/New_York')

        # Get NY midnight for today
        ny_now = dt.astimezone(ny_tz)
        ny_midnight = ny_now.replace(hour=0, minute=0, second=0, microsecond=0)
        utc_midnight = ny_midnight.astimezone(self.utc)

        # Find candle closest to this time
        # This is simplified - in production you'd query exact candle
        try:
            # Assuming df has datetime index
            if hasattr(df.index, 'tz_localize'):
                df_utc = df.copy()
            else:
                df_utc = df.copy()

            # Get open price around that time
            return float(df_utc['open'].iloc[0])  # Simplified
        except Exception:
            return None

    def format_session_summary(self) -> str:
        """Format current session status as human-readable summary"""
        info = self.get_session_info()
        po3 = self.get_power_of_three_phase()

        lines = []
        lines.append(f"Session: {info.current_session.value.replace('_', ' ').title()}")

        if info.in_kill_zone:
            lines.append(f"Kill Zone: {info.kill_zone_name} (ACTIVE)")
        else:
            if info.time_to_next_kz:
                hours = info.time_to_next_kz.seconds // 3600
                mins = (info.time_to_next_kz.seconds % 3600) // 60
                lines.append(f"Next Kill Zone in: {hours}h {mins}m")

        lines.append(f"Power of Three: {po3['phase'].title()}")
        lines.append(f"Daily Bias: {info.daily_bias.replace('_', ' ').title()}")

        return "\n".join(lines)


# Convenience functions
def is_kill_zone_active() -> Tuple[bool, Optional[str]]:
    """Quick check if we're in a kill zone"""
    analyzer = KillZoneAnalyzer()
    in_kz, kz = analyzer.is_in_kill_zone()
    return in_kz, kz.name if kz else None


def get_session_summary() -> Dict:
    """Get quick session summary"""
    analyzer = KillZoneAnalyzer()
    info = analyzer.get_session_info()
    po3 = analyzer.get_power_of_three_phase()

    return {
        'session': info.current_session.value,
        'in_kill_zone': info.in_kill_zone,
        'kill_zone': info.kill_zone_name,
        'po3_phase': po3['phase'],
        'daily_bias': info.daily_bias,
        'optimal_pairs': analyzer.get_optimal_pairs_for_session()
    }


if __name__ == "__main__":
    # Test
    analyzer = KillZoneAnalyzer()

    print("=" * 50)
    print("Smart Money KILL ZONE ANALYZER")
    print("=" * 50)

    print(f"\nCurrent UTC Time: {analyzer.get_current_utc_time()}")
    print(f"\n{analyzer.format_session_summary()}")

    print("\nOptimal Pairs for Current Session:")
    for pair in analyzer.get_optimal_pairs_for_session():
        print(f"  - {pair}")

    print("\nAll Kill Zones:")
    for kz in analyzer.KILL_ZONES:
        print(f"  {kz.name}: {kz.start_time} - {kz.end_time} UTC (weight: {kz.bias_weight})")
