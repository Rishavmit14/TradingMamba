"""Data models for the TradingMamba ICT/SMC detection engine.

Every detector takes Candle arrays as input and returns typed model instances.
These are pure data containers — no business logic.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class Direction(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class SwingType(Enum):
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"


class SwingClassification(Enum):
    HH = "HH"   # Higher High
    HL = "HL"   # Higher Low
    LH = "LH"   # Lower High
    LL = "LL"   # Lower Low
    UNCLASSIFIED = "unclassified"


class TrendState(Enum):
    BULLISH = "bullish"     # HH + HL sequence
    BEARISH = "bearish"     # LH + LL sequence
    RANGING = "ranging"     # No clear direction


class IDMStatus(Enum):
    ACTIVE = "active"       # IDM exists, not yet taken
    TAKEN = "taken"         # IDM has been swept
    TRANSFERRED = "transferred"  # IDM transferred from previous swing


class LiquidityType(Enum):
    BUY_SIDE = "buy_side"     # Liquidity above (buy stops, sell limits)
    SELL_SIDE = "sell_side"   # Liquidity below (sell stops, buy limits)


class LiquiditySource(Enum):
    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"
    SWING_EXTREME = "swing_extreme"
    TRENDLINE = "trendline"
    IDM_LEVEL = "idm_level"


class LiquidityEvent(Enum):
    SWEEP = "sweep"   # Wick sweeps but doesn't close beyond
    GRAB = "grab"     # Body closes beyond, traps traders, reverses


class ZoneType(Enum):
    PREMIUM = "premium"
    DISCOUNT = "discount"
    EQUILIBRIUM = "equilibrium"


class EntryMethod(Enum):
    MSS = "mss"                    # Market Structure Shift
    SCOB = "scob"                  # Sweep-based Change of Bias
    PULLBACK_BREAK = "pullback_break"  # Valid pullback break
    SBC = "sbc"                    # Sweep Based Change of Character


class SignalGrade(Enum):
    A = "A"  # 3+ confluences, trend-aligned, kill zone, multi-TF
    B = "B"  # 2 confluences, trend-aligned
    C = "C"  # 1 confluence or weak alignment
    D = "D"  # Counter-trend or climax warning


# ──────────────────────────────────────────────
# Core Data Models
# ──────────────────────────────────────────────

@dataclass
class Candle:
    """Single OHLCV candlestick."""
    timestamp: int        # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    index: int = 0        # Position in the candle array

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def body_top(self) -> float:
        return max(self.open, self.close)

    @property
    def body_bottom(self) -> float:
        return min(self.open, self.close)

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def total_range(self) -> float:
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        return self.high - self.body_top

    @property
    def lower_wick(self) -> float:
        return self.body_bottom - self.low


@dataclass
class SwingPoint:
    """A structural swing high or low."""
    candle_index: int
    price: float
    swing_type: SwingType
    classification: SwingClassification = SwingClassification.UNCLASSIFIED
    is_valid_smc: bool = True       # False = inducement/liquidity zone (marked X)
    is_strong: bool = False         # Strong vs weak swing (V09-V10 concept)
    idm_taken: bool = False         # Was inducement taken before this swing?
    candle_closed_properly: bool = False  # Body close condition met?


@dataclass
class Inducement:
    """An inducement (IDM) level — first valid pullback on left side of swing."""
    candle_index: int
    price: float
    parent_swing_index: int   # The swing this IDM belongs to
    status: IDMStatus = IDMStatus.ACTIVE
    taken_at_candle: Optional[int] = None  # Candle index where IDM was swept
    body_closed: bool = False  # V04: body closed beyond IDM (not just wick)
    is_major: bool = True      # V08: major IDM (deepest pullback, 80-85% probability)


@dataclass
class LiquidityPool:
    """A cluster of liquidity at a price level."""
    price_level: float
    pool_type: LiquidityType
    source: LiquiditySource
    candle_indices: list[int] = field(default_factory=list)  # Candles forming this pool
    swept: bool = False
    swept_at_candle: Optional[int] = None
    event_type: Optional[LiquidityEvent] = None


@dataclass
class BOS:
    """Break of Structure event."""
    candle_index: int          # Candle that confirmed BOS
    direction: Direction
    broken_swing_index: int    # Which swing point was broken
    broken_price: float        # The price level that was broken
    valid: bool = True
    invalidation_reason: Optional[str] = None
    idm_body_closed: bool = False  # V06: IDM taken with body close (Swing HH tier)


@dataclass
class CHoCH:
    """Change of Character event — potential trend reversal."""
    candle_index: int
    direction: Direction       # New direction (bullish_to_bearish or bearish_to_bullish)
    broken_swing_index: int    # Which major HL/LH was broken
    broken_price: float
    confidence: float = 0.0   # 0-1 confidence score
    has_climax_confluence: bool = False
    is_fake: bool = False      # Filtered by V09 fake CHoCH rules
    confirmed: bool = False    # V10 confirmation (follow-through)
    model: str = ""            # V10/V15: "swing" (MSS) or "sweep" (SBC)


@dataclass
class FVG:
    """Fair Value Gap — price imbalance zone."""
    candle_index: int          # Middle candle of the 3-candle pattern
    upper_price: float         # Top of gap (wick-to-wick)
    lower_price: float         # Bottom of gap (wick-to-wick)
    direction: Direction       # Bullish FVG = gap up, bearish FVG = gap down
    valid: bool = True
    from_extreme_candle: bool = False   # Is this from the extreme candle? (V13 rule)
    mitigated: bool = False    # Has price returned to fill this gap?
    mitigated_at_candle: Optional[int] = None

    @property
    def midpoint(self) -> float:
        return (self.upper_price + self.lower_price) / 2

    @property
    def size(self) -> float:
        return self.upper_price - self.lower_price


@dataclass
class OrderBlock:
    """Order Block — institutional order placement zone."""
    candle_index_start: int    # First candle of OB
    candle_index_end: int      # Last candle of OB
    upper_price: float
    lower_price: float
    direction: Direction       # Bullish OB (bearish candle before buy) or bearish OB
    valid: bool = True
    has_fvg: bool = False      # Rule 2: FVG must exist below/above
    swept_liquidity: bool = False  # Rule 1: Must sweep previous candle liquidity
    is_trap: bool = False      # Part of inducement/engineered liquidity
    mitigated: bool = False    # Has price returned to this OB?
    mitigated_at_candle: Optional[int] = None

    @property
    def midpoint(self) -> float:
        return (self.upper_price + self.lower_price) / 2

    @property
    def size(self) -> float:
        return self.upper_price - self.lower_price


@dataclass
class PremiumDiscount:
    """Premium/Discount zone calculation for a given range."""
    swing_high: float
    swing_low: float
    equilibrium: float         # 50% level
    zone: ZoneType
    depth_pct: float           # How deep into premium/discount (0-100%)

    @classmethod
    def calculate(cls, swing_high: float, swing_low: float, current_price: float) -> PremiumDiscount:
        eq = (swing_high + swing_low) / 2
        total_range = swing_high - swing_low
        if total_range == 0:
            return cls(swing_high, swing_low, eq, ZoneType.EQUILIBRIUM, 50.0)

        if current_price > eq:
            zone = ZoneType.PREMIUM
            depth_pct = ((current_price - eq) / (swing_high - eq)) * 100
        elif current_price < eq:
            zone = ZoneType.DISCOUNT
            depth_pct = ((eq - current_price) / (eq - swing_low)) * 100
        else:
            zone = ZoneType.EQUILIBRIUM
            depth_pct = 50.0

        return cls(swing_high, swing_low, eq, zone, min(depth_pct, 100.0))


@dataclass
class Session:
    """Current market session context."""
    name: str                  # "asian", "london", "ny", "late"
    is_kill_zone: bool
    volatility_expectation: str  # "low", "medium", "high"


@dataclass
class TradingSignal:
    """Complete trading signal output from the Master Checklist."""
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    confidence_score: float    # 0-100
    grade: SignalGrade
    confluences: list[str] = field(default_factory=list)
    timeframe: str = ""
    entry_method: Optional[EntryMethod] = None
    pattern_type: str = ""     # e.g., "FVG+OB trend continuation"
    timestamp: int = 0
    # Context
    w1_trend: Optional[TrendState] = None
    d1_trend: Optional[TrendState] = None
    session: Optional[Session] = None
    climax_warning: bool = False
    is_counter_trend: bool = False
