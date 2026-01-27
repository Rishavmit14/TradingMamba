"""
ICT Chart Generator
Generates annotated trading charts with ICT patterns
Uses matplotlib - 100% FREE, no TradingView subscription needed!
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import io
import base64


class ICTChartGenerator:
    """
    Generate professional ICT-annotated charts

    Features:
    - Candlestick charts
    - Order Block zones
    - Fair Value Gap highlights
    - Liquidity levels
    - Entry/SL/TP markers
    - Premium/Discount zones
    """

    def __init__(self, style: str = 'dark'):
        """
        Initialize chart generator

        Parameters:
        - style: 'dark' or 'light'
        """
        self.style = style
        self.setup_style()

        # Chart directory
        self.chart_dir = Path(__file__).parent.parent.parent / "data" / "charts"
        self.chart_dir.mkdir(parents=True, exist_ok=True)

    def setup_style(self):
        """Set up chart style"""
        if self.style == 'dark':
            plt.style.use('dark_background')
            self.colors = {
                'background': '#1a1a2e',
                'text': '#e0e0e0',
                'grid': '#2a2a3e',
                'bullish': '#26a69a',
                'bearish': '#ef5350',
                'bullish_ob': 'rgba(38, 166, 154, 0.3)',
                'bearish_ob': 'rgba(239, 83, 80, 0.3)',
                'bullish_fvg': '#26a69a',
                'bearish_fvg': '#ef5350',
                'liquidity': '#ffd700',
                'entry': '#4fc3f7',
                'stop_loss': '#f44336',
                'take_profit': '#66bb6a',
                'premium': '#ffcdd2',
                'discount': '#c8e6c9',
                'equilibrium': '#fff59d'
            }
        else:
            plt.style.use('seaborn-whitegrid')
            self.colors = {
                'background': '#ffffff',
                'text': '#333333',
                'grid': '#e0e0e0',
                'bullish': '#26a69a',
                'bearish': '#ef5350',
                'bullish_ob': 'rgba(38, 166, 154, 0.2)',
                'bearish_ob': 'rgba(239, 83, 80, 0.2)',
                'bullish_fvg': '#26a69a',
                'bearish_fvg': '#ef5350',
                'liquidity': '#ff9800',
                'entry': '#2196f3',
                'stop_loss': '#f44336',
                'take_profit': '#4caf50',
                'premium': '#ffebee',
                'discount': '#e8f5e9',
                'equilibrium': '#fffde7'
            }

    def generate_candlestick_chart(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        signal: Optional[Dict] = None,
        patterns: Optional[List[Dict]] = None,
        analysis: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a complete ICT-annotated candlestick chart

        Parameters:
        - data: OHLCV DataFrame
        - symbol: Trading symbol
        - timeframe: Chart timeframe
        - signal: Optional signal data with entry/SL/TP
        - patterns: Optional list of detected patterns
        - analysis: Optional ICT analysis results

        Returns:
        - Path to saved chart or base64 encoded image
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        if self.style == 'dark':
            fig.patch.set_facecolor(self.colors['background'])
            ax.set_facecolor(self.colors['background'])

        # Draw candlesticks
        self._draw_candlesticks(ax, data)

        # Draw patterns if provided
        if patterns:
            self._draw_patterns(ax, data, patterns)

        # Draw analysis zones if provided
        if analysis:
            self._draw_analysis(ax, data, analysis)

        # Draw signal levels if provided
        if signal:
            self._draw_signal_levels(ax, data, signal)

        # Formatting
        self._format_chart(ax, data, symbol, timeframe)

        # Add legend
        self._add_legend(ax, signal, patterns)

        plt.tight_layout()

        # Save or return
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            plt.close(fig)
            return save_path
        else:
            # Return as base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            return f"data:image/png;base64,{image_base64}"

    def _draw_candlesticks(self, ax, data: pd.DataFrame):
        """Draw candlestick chart"""
        for i in range(len(data)):
            open_price = data['open'].iloc[i]
            close_price = data['close'].iloc[i]
            high_price = data['high'].iloc[i]
            low_price = data['low'].iloc[i]

            # Determine color
            if close_price >= open_price:
                color = self.colors['bullish']
                body_bottom = open_price
            else:
                color = self.colors['bearish']
                body_bottom = close_price

            body_height = abs(close_price - open_price)

            # Draw wick
            ax.plot([i, i], [low_price, high_price],
                   color=color, linewidth=1)

            # Draw body
            rect = Rectangle((i - 0.4, body_bottom), 0.8, body_height,
                            facecolor=color, edgecolor=color)
            ax.add_patch(rect)

    def _draw_patterns(self, ax, data: pd.DataFrame, patterns: List[Dict]):
        """Draw detected ICT patterns"""
        for pattern in patterns:
            pattern_type = pattern.get('pattern_type', '')
            start_idx = pattern.get('start_index', 0)
            end_idx = pattern.get('end_index', start_idx)
            high = pattern.get('price_high', 0)
            low = pattern.get('price_low', 0)

            # Ensure indices are within bounds
            if start_idx >= len(data) or end_idx >= len(data):
                continue

            # Order Blocks
            if 'order_block' in pattern_type:
                color = self.colors['bullish'] if 'bullish' in pattern_type else self.colors['bearish']
                alpha = 0.3
                rect = Rectangle(
                    (start_idx - 0.5, low),
                    end_idx - start_idx + 1,
                    high - low,
                    facecolor=color,
                    edgecolor=color,
                    alpha=alpha,
                    linewidth=2
                )
                ax.add_patch(rect)

                # Label
                label_y = high if 'bullish' in pattern_type else low
                ax.annotate('OB', xy=(start_idx, label_y),
                           fontsize=8, color=color, fontweight='bold')

            # Fair Value Gaps
            elif 'fvg' in pattern_type:
                color = self.colors['bullish_fvg'] if 'bullish' in pattern_type else self.colors['bearish_fvg']
                # Draw horizontal lines for FVG
                ax.axhspan(low, high, xmin=(start_idx - 0.5) / len(data),
                          xmax=1, alpha=0.2, color=color)

                # Add marker
                ax.annotate('FVG', xy=(start_idx, (high + low) / 2),
                           fontsize=8, color=color, fontweight='bold')

            # Liquidity levels
            elif 'liquidity' in pattern_type or 'equal' in pattern_type:
                level = pattern.get('details', {}).get('level', (high + low) / 2)
                ax.axhline(y=level, color=self.colors['liquidity'],
                          linestyle='--', linewidth=1, alpha=0.7)

                label = 'EQH' if 'high' in pattern_type else 'EQL' if 'low' in pattern_type else 'LIQ'
                ax.annotate(label, xy=(len(data) - 1, level),
                           fontsize=8, color=self.colors['liquidity'])

            # BOS/CHoCH
            elif 'bos' in pattern_type or 'choch' in pattern_type:
                level = pattern.get('details', {}).get('break_level', high)
                color = self.colors['bullish'] if 'bullish' in pattern_type else self.colors['bearish']

                ax.axhline(y=level, color=color, linestyle=':',
                          linewidth=1.5, alpha=0.8)

                label = 'BOS' if 'bos' in pattern_type else 'CHoCH'
                ax.annotate(label, xy=(end_idx, level),
                           fontsize=9, color=color, fontweight='bold')

    def _draw_analysis(self, ax, data: pd.DataFrame, analysis: Dict):
        """Draw ICT analysis zones"""
        # Premium/Discount zones
        if 'premium_discount' in analysis:
            pd_info = analysis['premium_discount']
            range_high = pd_info.get('range_high', 0)
            range_low = pd_info.get('range_low', 0)
            equilibrium = pd_info.get('equilibrium', 0)

            if range_high and range_low:
                # Premium zone (top 30%)
                premium_low = equilibrium + (range_high - equilibrium) * 0.4
                ax.axhspan(premium_low, range_high, alpha=0.1,
                          color=self.colors['bearish'], label='Premium Zone')

                # Discount zone (bottom 30%)
                discount_high = equilibrium - (equilibrium - range_low) * 0.4
                ax.axhspan(range_low, discount_high, alpha=0.1,
                          color=self.colors['bullish'], label='Discount Zone')

                # Equilibrium line
                ax.axhline(y=equilibrium, color=self.colors['equilibrium'],
                          linestyle='-', linewidth=2, alpha=0.8, label='Equilibrium')

        # Draw order blocks from analysis
        if 'order_blocks' in analysis:
            for ob in analysis['order_blocks'][:5]:  # Top 5
                if hasattr(ob, 'high'):
                    high, low = ob.high, ob.low
                    ob_type = ob.type
                else:
                    high = ob.get('high', 0)
                    low = ob.get('low', 0)
                    ob_type = ob.get('type', 'bullish')

                color = self.colors['bullish'] if ob_type == 'bullish' else self.colors['bearish']
                ax.axhspan(low, high, alpha=0.15, color=color)

    def _draw_signal_levels(self, ax, data: pd.DataFrame, signal: Dict):
        """Draw entry, stop loss, and take profit levels"""
        # Entry level
        entry = signal.get('entry_price', 0)
        if entry:
            ax.axhline(y=entry, color=self.colors['entry'],
                      linestyle='-', linewidth=2, label=f'Entry: {entry:.5f}')

        # Entry zone
        entry_zone = signal.get('entry_zone', (0, 0))
        if isinstance(entry_zone, (list, tuple)) and len(entry_zone) >= 2:
            if entry_zone[0] and entry_zone[1]:
                ax.axhspan(entry_zone[0], entry_zone[1], alpha=0.2,
                          color=self.colors['entry'])

        # Stop loss
        stop_loss = signal.get('stop_loss', 0)
        if stop_loss:
            ax.axhline(y=stop_loss, color=self.colors['stop_loss'],
                      linestyle='--', linewidth=2, label=f'SL: {stop_loss:.5f}')

        # Take profit levels
        take_profits = signal.get('take_profit', [])
        if isinstance(take_profits, list):
            for i, tp in enumerate(take_profits[:3]):
                if tp:
                    ax.axhline(y=tp, color=self.colors['take_profit'],
                              linestyle='--', linewidth=1.5,
                              label=f'TP{i+1}: {tp:.5f}')

        # Direction arrow
        direction = signal.get('direction', '')
        current_price = data['close'].iloc[-1]

        if direction == 'BUY':
            ax.annotate('', xy=(len(data) - 1, current_price * 1.005),
                       xytext=(len(data) - 1, current_price),
                       arrowprops=dict(arrowstyle='->', color=self.colors['bullish'],
                                      lw=3))
        elif direction == 'SELL':
            ax.annotate('', xy=(len(data) - 1, current_price * 0.995),
                       xytext=(len(data) - 1, current_price),
                       arrowprops=dict(arrowstyle='->', color=self.colors['bearish'],
                                      lw=3))

    def _format_chart(self, ax, data: pd.DataFrame, symbol: str, timeframe: str):
        """Format chart appearance"""
        # Title
        current_price = data['close'].iloc[-1]
        change = ((current_price - data['close'].iloc[0]) / data['close'].iloc[0]) * 100
        change_color = self.colors['bullish'] if change >= 0 else self.colors['bearish']

        ax.set_title(f'{symbol} | {timeframe} | ${current_price:.5f} ({change:+.2f}%)',
                    fontsize=14, fontweight='bold', color=self.colors['text'])

        # Grid
        ax.grid(True, alpha=0.3, color=self.colors['grid'])

        # Labels
        ax.set_xlabel('Time', fontsize=10, color=self.colors['text'])
        ax.set_ylabel('Price', fontsize=10, color=self.colors['text'])

        # X-axis formatting
        ax.set_xlim(-1, len(data))

        # Y-axis padding
        y_range = data['high'].max() - data['low'].min()
        ax.set_ylim(data['low'].min() - y_range * 0.05,
                   data['high'].max() + y_range * 0.05)

        # Tick colors
        ax.tick_params(colors=self.colors['text'])

    def _add_legend(self, ax, signal: Optional[Dict], patterns: Optional[List[Dict]]):
        """Add chart legend"""
        handles = []
        labels = []

        # Signal info
        if signal:
            direction = signal.get('direction', 'WAIT')
            confidence = signal.get('confidence', 0)
            color = self.colors['bullish'] if direction == 'BUY' else \
                    self.colors['bearish'] if direction == 'SELL' else self.colors['text']

            handles.append(mpatches.Patch(color=color, alpha=0.5))
            labels.append(f'{direction} ({confidence:.0%})')

        # Pattern count
        if patterns:
            bullish = sum(1 for p in patterns if 'bullish' in p.get('pattern_type', ''))
            bearish = sum(1 for p in patterns if 'bearish' in p.get('pattern_type', ''))

            handles.append(mpatches.Patch(color=self.colors['bullish'], alpha=0.5))
            labels.append(f'Bullish: {bullish}')

            handles.append(mpatches.Patch(color=self.colors['bearish'], alpha=0.5))
            labels.append(f'Bearish: {bearish}')

        if handles:
            ax.legend(handles, labels, loc='upper left', fontsize=9,
                     facecolor=self.colors['background'], edgecolor=self.colors['grid'])

    def generate_signal_chart(self, symbol: str, timeframe: str,
                              signal: Dict, data: pd.DataFrame) -> str:
        """
        Generate a chart specifically for a trading signal

        Returns base64 encoded image or file path
        """
        # Generate filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol.replace('/', '')}_{timeframe}_{timestamp}.png"
        save_path = self.chart_dir / filename

        return self.generate_candlestick_chart(
            data=data,
            symbol=symbol,
            timeframe=timeframe,
            signal=signal,
            save_path=str(save_path)
        )


# Singleton instance
chart_generator = ICTChartGenerator()


# Quick test function
def test_chart():
    """Generate a test chart"""
    import numpy as np

    # Generate sample data
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='H')

    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.0010)
    high = close + np.abs(np.random.randn(n) * 0.0005)
    low = close - np.abs(np.random.randn(n) * 0.0005)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]

    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)

    # Sample signal
    signal = {
        'direction': 'BUY',
        'confidence': 0.75,
        'entry_price': close[-1],
        'entry_zone': (close[-1] * 0.999, close[-1] * 1.001),
        'stop_loss': close[-1] * 0.995,
        'take_profit': [close[-1] * 1.01, close[-1] * 1.02, close[-1] * 1.03]
    }

    # Generate chart
    generator = ICTChartGenerator()
    result = generator.generate_candlestick_chart(
        data=data,
        symbol='EUR/USD',
        timeframe='H1',
        signal=signal
    )

    print(f"Chart generated: {result[:50]}...")
    return result


if __name__ == "__main__":
    test_chart()
