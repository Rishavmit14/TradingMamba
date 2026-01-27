"""
Technical Analysis Module

Provides technical indicators and pattern detection for ICT analysis.
Uses pandas-ta (100% FREE) for indicator calculations.

Includes ICT-specific analysis like:
- Order Block detection
- Fair Value Gap identification
- Liquidity level mapping
- Market structure analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    import pandas as pd
except ImportError:
    raise ImportError("Install pandas: pip install pandas")

# Try to import pandas-ta, fall back to manual calculations if not available
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logging.warning("pandas-ta not installed. Using manual indicator calculations.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderBlock:
    """Represents an ICT Order Block"""
    type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    timestamp: datetime
    timeframe: str
    mitigated: bool = False
    strength: float = 1.0  # Based on move after OB


@dataclass
class FairValueGap:
    """Represents an ICT Fair Value Gap"""
    type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    timestamp: datetime
    timeframe: str
    filled: bool = False
    size_atr: float = 0  # Size relative to ATR


@dataclass
class LiquidityLevel:
    """Represents a liquidity level (equal highs/lows, swing points)"""
    type: str  # 'buy_side' or 'sell_side'
    price: float
    timestamp: datetime
    touches: int = 1
    swept: bool = False


@dataclass
class MarketStructure:
    """Represents market structure (HH, HL, LH, LL, BOS, CHoCH)"""
    type: str  # 'hh', 'hl', 'lh', 'll', 'bos', 'choch'
    price: float
    timestamp: datetime
    bias: str  # 'bullish', 'bearish', 'neutral'


class TechnicalIndicators:
    """
    Calculate technical indicators.
    Uses pandas-ta if available, otherwise manual calculations.
    """

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        if HAS_PANDAS_TA:
            return ta.atr(df['high'], df['low'], df['close'], length=period)

        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        if HAS_PANDAS_TA:
            return ta.rsi(df['close'], length=period)

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        if HAS_PANDAS_TA:
            return ta.ema(series, length=period)
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        if HAS_PANDAS_TA:
            return ta.sma(series, length=period)
        return series.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        if HAS_PANDAS_TA:
            bb = ta.bbands(df['close'], length=period, std=std)
            return bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]  # lower, mid, upper

        mid = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()
        upper = mid + (std_dev * std)
        lower = mid - (std_dev * std)
        return lower, mid, upper

    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD"""
        if HAS_PANDAS_TA:
            macd_df = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
            return macd_df.iloc[:, 0], macd_df.iloc[:, 1], macd_df.iloc[:, 2]

        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        if HAS_PANDAS_TA:
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=k_period, d=d_period)
            return stoch.iloc[:, 0], stoch.iloc[:, 1]

        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()

        k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return k, d


class ICTStructureAnalyzer:
    """
    Analyzes market structure according to ICT methodology.
    Identifies swing points, BOS, CHoCH, and trend direction.
    """

    def __init__(self, swing_lookback: int = 5):
        self.swing_lookback = swing_lookback

    def find_swing_points(self, df: pd.DataFrame) -> List[Dict]:
        """Find swing highs and swing lows"""
        swing_points = []
        n = self.swing_lookback

        for i in range(n, len(df) - n):
            # Swing High: Higher than n bars on each side
            if df['high'].iloc[i] == df['high'].iloc[i-n:i+n+1].max():
                swing_points.append({
                    'type': 'swing_high',
                    'price': float(df['high'].iloc[i]),
                    'index': i,
                    'timestamp': df.index[i] if hasattr(df.index[i], 'isoformat') else str(df.index[i])
                })

            # Swing Low: Lower than n bars on each side
            if df['low'].iloc[i] == df['low'].iloc[i-n:i+n+1].min():
                swing_points.append({
                    'type': 'swing_low',
                    'price': float(df['low'].iloc[i]),
                    'index': i,
                    'timestamp': df.index[i] if hasattr(df.index[i], 'isoformat') else str(df.index[i])
                })

        return sorted(swing_points, key=lambda x: x['index'])

    def analyze_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure (HH, HL, LH, LL, BOS, CHoCH)"""
        swings = self.find_swing_points(df)

        if len(swings) < 4:
            return {'bias': 'neutral', 'structure': [], 'swings': swings}

        structure = []
        highs = [s for s in swings if s['type'] == 'swing_high']
        lows = [s for s in swings if s['type'] == 'swing_low']

        # Analyze highs
        for i in range(1, len(highs)):
            if highs[i]['price'] > highs[i-1]['price']:
                structure.append({
                    'type': 'higher_high',
                    'price': highs[i]['price'],
                    'index': highs[i]['index']
                })
            else:
                structure.append({
                    'type': 'lower_high',
                    'price': highs[i]['price'],
                    'index': highs[i]['index']
                })

        # Analyze lows
        for i in range(1, len(lows)):
            if lows[i]['price'] > lows[i-1]['price']:
                structure.append({
                    'type': 'higher_low',
                    'price': lows[i]['price'],
                    'index': lows[i]['index']
                })
            else:
                structure.append({
                    'type': 'lower_low',
                    'price': lows[i]['price'],
                    'index': lows[i]['index']
                })

        # Determine bias
        recent_structure = sorted(structure, key=lambda x: x['index'])[-4:]
        hh_count = sum(1 for s in recent_structure if s['type'] == 'higher_high')
        hl_count = sum(1 for s in recent_structure if s['type'] == 'higher_low')
        lh_count = sum(1 for s in recent_structure if s['type'] == 'lower_high')
        ll_count = sum(1 for s in recent_structure if s['type'] == 'lower_low')

        if hh_count >= 1 and hl_count >= 1:
            bias = 'bullish'
        elif lh_count >= 1 and ll_count >= 1:
            bias = 'bearish'
        else:
            bias = 'neutral'

        # Detect BOS (Break of Structure)
        bos_events = []
        if len(highs) >= 2 and len(lows) >= 2:
            last_swing_high = highs[-1]['price']
            last_swing_low = lows[-1]['price']
            current_price = float(df['close'].iloc[-1])

            if current_price > last_swing_high:
                bos_events.append({'type': 'bullish_bos', 'level': last_swing_high})
            elif current_price < last_swing_low:
                bos_events.append({'type': 'bearish_bos', 'level': last_swing_low})

        return {
            'bias': bias,
            'structure': structure,
            'swings': swings,
            'bos_events': bos_events,
            'swing_high': highs[-1]['price'] if highs else None,
            'swing_low': lows[-1]['price'] if lows else None,
        }


class ICTConceptDetector:
    """
    Detects ICT-specific concepts in price data:
    - Order Blocks
    - Fair Value Gaps
    - Liquidity Levels
    - Premium/Discount Zones
    """

    def __init__(self):
        self.indicators = TechnicalIndicators()

    def find_order_blocks(self, df: pd.DataFrame, lookback: int = 50) -> List[OrderBlock]:
        """
        Find Order Blocks.

        Bullish OB: Last down candle before a strong up move
        Bearish OB: Last up candle before a strong down move
        """
        order_blocks = []
        atr = self.indicators.atr(df).iloc[-1] if len(df) > 14 else df['high'].iloc[-14:].mean() - df['low'].iloc[-14:].mean()

        for i in range(lookback, len(df) - 3):
            # Bullish Order Block
            if df['close'].iloc[i] < df['open'].iloc[i]:  # Down candle
                # Check for strong up move after
                future_high = df['high'].iloc[i+1:i+4].max()
                move = future_high - df['high'].iloc[i]

                if move > atr * 1.5:  # Strong move
                    ob = OrderBlock(
                        type='bullish',
                        high=float(df['high'].iloc[i]),
                        low=float(df['low'].iloc[i]),
                        timestamp=df.index[i],
                        timeframe='',  # Will be set by caller
                        strength=float(move / atr)
                    )
                    order_blocks.append(ob)

            # Bearish Order Block
            if df['close'].iloc[i] > df['open'].iloc[i]:  # Up candle
                # Check for strong down move after
                future_low = df['low'].iloc[i+1:i+4].min()
                move = df['low'].iloc[i] - future_low

                if move > atr * 1.5:  # Strong move
                    ob = OrderBlock(
                        type='bearish',
                        high=float(df['high'].iloc[i]),
                        low=float(df['low'].iloc[i]),
                        timestamp=df.index[i],
                        timeframe='',
                        strength=float(move / atr)
                    )
                    order_blocks.append(ob)

        # Check if OBs have been mitigated
        current_price = float(df['close'].iloc[-1])
        for ob in order_blocks:
            if ob.type == 'bullish' and current_price < ob.low:
                ob.mitigated = True
            elif ob.type == 'bearish' and current_price > ob.high:
                ob.mitigated = True

        return order_blocks

    def find_fair_value_gaps(self, df: pd.DataFrame, lookback: int = 50) -> List[FairValueGap]:
        """
        Find Fair Value Gaps (FVGs).

        Bullish FVG: Gap between candle 1's high and candle 3's low
        Bearish FVG: Gap between candle 1's low and candle 3's high
        """
        fvgs = []
        atr = self.indicators.atr(df).iloc[-1] if len(df) > 14 else 0.001

        for i in range(lookback, len(df) - 2):
            # Bullish FVG (gap up)
            if df['low'].iloc[i+2] > df['high'].iloc[i]:
                gap_size = df['low'].iloc[i+2] - df['high'].iloc[i]
                if gap_size > atr * 0.2:  # Minimum gap size
                    fvg = FairValueGap(
                        type='bullish',
                        high=float(df['low'].iloc[i+2]),
                        low=float(df['high'].iloc[i]),
                        timestamp=df.index[i+1],
                        timeframe='',
                        size_atr=float(gap_size / atr)
                    )
                    fvgs.append(fvg)

            # Bearish FVG (gap down)
            if df['high'].iloc[i+2] < df['low'].iloc[i]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i+2]
                if gap_size > atr * 0.2:
                    fvg = FairValueGap(
                        type='bearish',
                        high=float(df['low'].iloc[i]),
                        low=float(df['high'].iloc[i+2]),
                        timestamp=df.index[i+1],
                        timeframe='',
                        size_atr=float(gap_size / atr)
                    )
                    fvgs.append(fvg)

        # Check if FVGs have been filled
        for fvg in fvgs:
            fvg_idx = df.index.get_loc(fvg.timestamp) if fvg.timestamp in df.index else -1
            if fvg_idx > 0:
                future_data = df.iloc[fvg_idx:]
                if fvg.type == 'bullish':
                    if future_data['low'].min() <= fvg.low:
                        fvg.filled = True
                else:
                    if future_data['high'].max() >= fvg.high:
                        fvg.filled = True

        return fvgs

    def find_liquidity_levels(self, df: pd.DataFrame, tolerance: float = 0.001) -> List[LiquidityLevel]:
        """
        Find liquidity levels (equal highs/lows, obvious swing points).

        These are areas where stop losses cluster.
        """
        liquidity_levels = []

        # Find equal highs
        highs = df['high'].values
        for i in range(len(highs) - 10):
            level = highs[i]
            touches = 1
            for j in range(i + 3, min(i + 30, len(highs))):
                if abs(highs[j] - level) / level < tolerance:
                    touches += 1

            if touches >= 2:
                liquidity_levels.append(LiquidityLevel(
                    type='buy_side',
                    price=float(level),
                    timestamp=df.index[i],
                    touches=touches
                ))

        # Find equal lows
        lows = df['low'].values
        for i in range(len(lows) - 10):
            level = lows[i]
            touches = 1
            for j in range(i + 3, min(i + 30, len(lows))):
                if abs(lows[j] - level) / level < tolerance:
                    touches += 1

            if touches >= 2:
                liquidity_levels.append(LiquidityLevel(
                    type='sell_side',
                    price=float(level),
                    timestamp=df.index[i],
                    touches=touches
                ))

        # Deduplicate close levels
        unique_levels = []
        for ll in liquidity_levels:
            is_duplicate = False
            for ul in unique_levels:
                if abs(ll.price - ul.price) / ul.price < tolerance * 2:
                    ul.touches += ll.touches - 1
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_levels.append(ll)

        # Check if levels have been swept
        current_price = float(df['close'].iloc[-1])
        recent_high = df['high'].tail(5).max()
        recent_low = df['low'].tail(5).min()

        for ll in unique_levels:
            if ll.type == 'buy_side' and recent_high > ll.price:
                ll.swept = True
            elif ll.type == 'sell_side' and recent_low < ll.price:
                ll.swept = True

        return unique_levels

    def calculate_premium_discount(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Calculate premium/discount zones based on recent range.

        Premium: Above 50% (equilibrium)
        Discount: Below 50%
        """
        recent = df.tail(lookback)
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        current = float(df['close'].iloc[-1])

        range_size = swing_high - swing_low
        equilibrium = (swing_high + swing_low) / 2

        # Calculate position in range (0 = low, 1 = high)
        if range_size > 0:
            position = (current - swing_low) / range_size
        else:
            position = 0.5

        # Zone boundaries
        premium_zone = (equilibrium + range_size * 0.2, swing_high)
        discount_zone = (swing_low, equilibrium - range_size * 0.2)

        if current > equilibrium:
            zone = 'premium'
        else:
            zone = 'discount'

        return {
            'current_price': current,
            'swing_high': float(swing_high),
            'swing_low': float(swing_low),
            'equilibrium': float(equilibrium),
            'position': float(position),
            'zone': zone,
            'premium_zone': premium_zone,
            'discount_zone': discount_zone,
            'optimal_buy_zone': discount_zone,
            'optimal_sell_zone': premium_zone,
        }


class FullICTAnalysis:
    """
    Performs complete ICT analysis on market data.
    Combines all ICT concepts for comprehensive view.
    """

    def __init__(self):
        self.structure_analyzer = ICTStructureAnalyzer()
        self.concept_detector = ICTConceptDetector()
        self.indicators = TechnicalIndicators()

    def analyze(self, df: pd.DataFrame, timeframe: str = 'H1') -> Dict:
        """Run full ICT analysis"""
        if df is None or df.empty:
            return {'error': 'No data provided'}

        analysis = {
            'timeframe': timeframe,
            'timestamp': datetime.utcnow().isoformat(),
            'candles_analyzed': len(df),
        }

        # Market Structure
        structure = self.structure_analyzer.analyze_structure(df)
        analysis['market_structure'] = structure

        # Order Blocks
        order_blocks = self.concept_detector.find_order_blocks(df)
        for ob in order_blocks:
            ob.timeframe = timeframe
        analysis['order_blocks'] = {
            'bullish': [ob.__dict__ for ob in order_blocks if ob.type == 'bullish' and not ob.mitigated][-3:],
            'bearish': [ob.__dict__ for ob in order_blocks if ob.type == 'bearish' and not ob.mitigated][-3:],
            'total_found': len(order_blocks),
        }

        # Fair Value Gaps
        fvgs = self.concept_detector.find_fair_value_gaps(df)
        for fvg in fvgs:
            fvg.timeframe = timeframe
        analysis['fair_value_gaps'] = {
            'bullish': [fvg.__dict__ for fvg in fvgs if fvg.type == 'bullish' and not fvg.filled][-3:],
            'bearish': [fvg.__dict__ for fvg in fvgs if fvg.type == 'bearish' and not fvg.filled][-3:],
            'total_found': len(fvgs),
        }

        # Liquidity Levels
        liquidity = self.concept_detector.find_liquidity_levels(df)
        analysis['liquidity'] = {
            'buy_side': [ll.__dict__ for ll in liquidity if ll.type == 'buy_side' and not ll.swept][-5:],
            'sell_side': [ll.__dict__ for ll in liquidity if ll.type == 'sell_side' and not ll.swept][-5:],
        }

        # Premium/Discount
        pd_analysis = self.concept_detector.calculate_premium_discount(df)
        analysis['premium_discount'] = pd_analysis

        # Technical Indicators
        analysis['indicators'] = {
            'rsi': float(self.indicators.rsi(df).iloc[-1]) if len(df) > 14 else None,
            'atr': float(self.indicators.atr(df).iloc[-1]) if len(df) > 14 else None,
        }

        # Generate Summary
        bias = structure['bias']
        zone = pd_analysis['zone']

        if bias == 'bullish' and zone == 'discount':
            summary = 'STRONG_BUY - Bullish structure in discount zone'
        elif bias == 'bearish' and zone == 'premium':
            summary = 'STRONG_SELL - Bearish structure in premium zone'
        elif bias == 'bullish':
            summary = 'BUY - Bullish structure (wait for discount)'
        elif bias == 'bearish':
            summary = 'SELL - Bearish structure (wait for premium)'
        else:
            summary = 'NEUTRAL - No clear structure'

        analysis['summary'] = summary
        analysis['detected_concepts'] = self._list_detected_concepts(analysis)

        return analysis

    def _list_detected_concepts(self, analysis: Dict) -> List[str]:
        """List all detected ICT concepts"""
        concepts = []

        if analysis.get('market_structure', {}).get('bias') != 'neutral':
            concepts.append('market_structure')

        if analysis.get('order_blocks', {}).get('total_found', 0) > 0:
            concepts.append('order_block')

        if analysis.get('fair_value_gaps', {}).get('total_found', 0) > 0:
            concepts.append('fair_value_gap')

        if analysis.get('liquidity', {}).get('buy_side') or analysis.get('liquidity', {}).get('sell_side'):
            concepts.append('liquidity')

        concepts.append('premium_discount')  # Always calculated

        return concepts


def test_technical_analysis():
    """Test the technical analysis module"""
    print("=" * 60)
    print("ICT TECHNICAL ANALYSIS TEST")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')

    # Simulate trending market
    trend = np.cumsum(np.random.randn(200) * 0.3 + 0.05)
    df = pd.DataFrame({
        'open': 100 + trend,
        'high': 100 + trend + abs(np.random.randn(200) * 0.4),
        'low': 100 + trend - abs(np.random.randn(200) * 0.4),
        'close': 100 + trend + np.random.randn(200) * 0.2,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)

    # Run analysis
    analyzer = FullICTAnalysis()
    result = analyzer.analyze(df, timeframe='H1')

    print(f"\n📊 Analysis Results:")
    print(f"   Timeframe: {result['timeframe']}")
    print(f"   Candles: {result['candles_analyzed']}")

    print(f"\n📈 Market Structure:")
    print(f"   Bias: {result['market_structure']['bias']}")
    print(f"   Swing High: {result['market_structure'].get('swing_high', 'N/A')}")
    print(f"   Swing Low: {result['market_structure'].get('swing_low', 'N/A')}")

    print(f"\n🎯 Order Blocks:")
    print(f"   Bullish OBs: {len(result['order_blocks']['bullish'])}")
    print(f"   Bearish OBs: {len(result['order_blocks']['bearish'])}")

    print(f"\n📊 Fair Value Gaps:")
    print(f"   Bullish FVGs: {len(result['fair_value_gaps']['bullish'])}")
    print(f"   Bearish FVGs: {len(result['fair_value_gaps']['bearish'])}")

    print(f"\n💧 Liquidity:")
    print(f"   Buy Side Levels: {len(result['liquidity']['buy_side'])}")
    print(f"   Sell Side Levels: {len(result['liquidity']['sell_side'])}")

    print(f"\n💰 Premium/Discount:")
    print(f"   Current Zone: {result['premium_discount']['zone']}")
    print(f"   Position: {result['premium_discount']['position']:.2%}")

    print(f"\n🎯 Summary: {result['summary']}")
    print(f"   Detected Concepts: {result['detected_concepts']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_technical_analysis()
