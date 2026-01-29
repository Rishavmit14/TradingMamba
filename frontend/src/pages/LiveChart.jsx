import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';
import {
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Target,
  Shield,
  Crosshair,
  BarChart2,
  Layers,
  Maximize2,
  AlertCircle,
  Eye,
  ChevronDown,
  X,
  Wifi,
  WifiOff
} from 'lucide-react';
import { getLiveOHLCV, getSignalAnalysis, getWebSocketUrl } from '../services/api';

// Available symbols organized by category
const SYMBOL_CATEGORIES = {
  'Crypto': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT'],
  'Forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY'],
  'Indices': ['US30', 'NAS100', 'SPX500'],
  'Commodities': ['XAUUSD'],
  'Stocks': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
};

const TIMEFRAMES = [
  { value: 'M1', label: '1m' },
  { value: 'M5', label: '5m' },
  { value: 'M15', label: '15m' },
  { value: 'M30', label: '30m' },
  { value: 'H1', label: '1H' },
  { value: 'H4', label: '4H' },
  { value: 'D1', label: '1D' },
];

// Smart Money concept descriptions (short names for display)
// These are now dynamically shown based on ML training status
const CONCEPT_INFO = {
  'fvg': { name: 'Fair Value Gap', short: 'FVG', color: '#ffd700', description: 'Price imbalance zone' },
  'order_block': { name: 'Order Block', short: 'OB', color: '#26a69a', description: 'Institutional entry zone' },
  'breaker_block': { name: 'Breaker Block', short: 'BB', color: '#ff6b6b', description: 'Failed order block' },
  'market_structure': { name: 'Market Structure', short: 'BOS/CHoCH', color: '#4fc3f7', description: 'Trend continuation/reversal' },
  'support_resistance': { name: 'Support/Resistance', short: 'S/R', color: '#9c27b0', description: 'Key price levels' },
  'liquidity': { name: 'Liquidity', short: 'LIQ', color: '#e91e63', description: 'Stop hunt zones' },
  'mitigation_block': { name: 'Mitigation Block', short: 'MB', color: '#00bcd4', description: 'Mitigation zone' },
  'rejection_block': { name: 'Rejection Block', short: 'RB', color: '#ff9800', description: 'Rejection zone' },
};

// Map backend pattern types to display names
const PATTERN_TYPE_MAP = {
  'bullish_order_block': { short: 'OB', color: '#26a69a', direction: 'bullish' },
  'bearish_order_block': { short: 'OB', color: '#ff9800', direction: 'bearish' },  // Orange for bearish OB
  'bullish_fvg': { short: 'FVG', color: '#4caf50', direction: 'bullish' },  // Green for bullish
  'bearish_fvg': { short: 'FVG', color: '#ef5350', direction: 'bearish' },  // Red for bearish FVG
  'bos_bullish': { short: 'BOS', color: '#4fc3f7', direction: 'bullish' },
  'bos_bearish': { short: 'BOS', color: '#29b6f6', direction: 'bearish' },
  'choch_bullish': { short: 'CHoCH', color: '#66bb6a', direction: 'bullish' },
  'choch_bearish': { short: 'CHoCH', color: '#ff9800', direction: 'bearish' },
  'equal_highs': { short: 'EQH', color: '#ef5350', direction: 'neutral' },
  'equal_lows': { short: 'EQL', color: '#66bb6a', direction: 'neutral' },
  'liquidity_sweep_high': { short: 'LIQ↑', color: '#e91e63', direction: 'bearish' },
  'liquidity_sweep_low': { short: 'LIQ↓', color: '#9c27b0', direction: 'bullish' },
  'optimal_trade_entry': { short: 'OTE', color: '#9c27b0', direction: 'neutral' },
  'accumulation': { short: 'ACC', color: '#4caf50', direction: 'bullish' },
  'manipulation': { short: 'MAN', color: '#f44336', direction: 'neutral' },
  'distribution': { short: 'DIST', color: '#ff5722', direction: 'bearish' },
};

function LiveChart() {
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('M15');
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [showLegend, setShowLegend] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [symbolDropdownOpen, setSymbolDropdownOpen] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [currentPrice, setCurrentPrice] = useState(null);
  const [priceChange, setPriceChange] = useState(0);

  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candlestickSeriesRef = useRef(null);
  const wsRef = useRef(null);
  const lastCandleRef = useRef(null);
  const priceLinesRef = useRef([]);
  const patternOverlaysRef = useRef([]);
  const rayOverlaysRef = useRef([]);
  const candlesDataRef = useRef([]);
  const analysisPatternsRef = useRef([]);

  // Format price with appropriate decimals
  const formatPrice = (price, sym = symbol) => {
    if (!price) return '-';
    // Handle crypto symbols (both BTC and BTCUSDT formats)
    if (['BTC', 'BTCUSDT', 'ETH', 'ETHUSDT'].includes(sym)) {
      return `${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USDT`;
    }
    // Other crypto with USDT
    if (sym.endsWith('USDT')) {
      return `${price.toFixed(4)} USDT`;
    }
    if (sym === 'XAUUSD') {
      return `$${price.toFixed(2)}`;
    }
    if (['US30', 'NAS100', 'SPX500'].includes(sym)) {
      return price.toFixed(2);
    }
    return price.toFixed(5);
  };

  // Get timeframe label for display
  const getTimeframeLabel = (tf) => {
    const labels = { 'M1': '1m', 'M5': '5m', 'M15': '15m', 'M30': '30m', 'H1': '1H', 'H4': '4H', 'D1': '1D' };
    return labels[tf] || tf;
  };

  // Clear pattern overlays
  const clearPatternOverlays = useCallback(() => {
    patternOverlaysRef.current.forEach(el => {
      if (el && el.parentNode) {
        el.parentNode.removeChild(el);
      }
    });
    patternOverlaysRef.current = [];

    // Also clear ray overlays
    rayOverlaysRef.current.forEach(el => {
      if (el && el.parentNode) {
        el.parentNode.removeChild(el);
      }
    });
    rayOverlaysRef.current = [];
  }, []);

  // Draw horizontal rays for BOS/CHoCH, liquidity, and equal highs/lows
  const drawHorizontalRays = useCallback((patterns, candles) => {
    if (!chartRef.current || !candlestickSeriesRef.current || !chartContainerRef.current || !patterns || patterns.length === 0 || candles.length === 0) {
      return;
    }

    const chart = chartRef.current;
    const series = candlestickSeriesRef.current;
    const chartElement = chartContainerRef.current;
    const chartHeight = chartElement.clientHeight || 500;

    // Pattern types that should be drawn as horizontal rays
    const rayPatternTypes = [
      'bos_bullish', 'bos_bearish',
      'choch_bullish', 'choch_bearish',
      'equal_highs', 'equal_lows',
      'liquidity_sweep_high', 'liquidity_sweep_low',
      'buyside_liquidity', 'sellside_liquidity'
    ];

    // Get annotation labels for each pattern type (all solid lines now)
    const getAnnotation = (patternType) => {
      const annotations = {
        'bos_bullish': { text: 'BOS ↑', color: '#4fc3f7' },
        'bos_bearish': { text: 'BOS ↓', color: '#ff9800' },
        'choch_bullish': { text: 'CHoCH ↑', color: '#66bb6a' },
        'choch_bearish': { text: 'CHoCH ↓', color: '#ff5722' },
        'equal_highs': { text: 'EQH', color: '#ef5350' },
        'equal_lows': { text: 'EQL', color: '#66bb6a' },
        'liquidity_sweep_high': { text: 'BSL Sweep', color: '#e91e63' },
        'liquidity_sweep_low': { text: 'SSL Sweep', color: '#9c27b0' },
        'buyside_liquidity': { text: 'BSL', color: '#ef5350' },
        'sellside_liquidity': { text: 'SSL', color: '#66bb6a' },
      };
      return annotations[patternType] || { text: patternType, color: '#9ca3af' };
    };

    // Filter to only ray patterns
    const rayPatterns = patterns.filter(p => rayPatternTypes.includes(p.pattern_type));

    rayPatterns.forEach((pattern, idx) => {
      const patternType = pattern.pattern_type;
      const annotation = getAnnotation(patternType);

      // Get the price level for the ray
      const price = pattern.price || pattern.high || pattern.low || pattern.price_high || pattern.price_low;
      if (!price) return;

      // Get Y coordinate for this price
      const y = series.priceToCoordinate(price);

      // Skip if price level is outside visible chart area
      if (y === null || y < 0 || y > chartHeight) return;

      // Determine if this is a high-type or low-type pattern
      const isHighPattern = patternType.includes('high') || patternType.includes('bullish') ||
                           patternType === 'buyside_liquidity' || patternType === 'equal_highs';
      const isLowPattern = patternType.includes('low') || patternType.includes('bearish') ||
                          patternType === 'sellside_liquidity' || patternType === 'equal_lows';

      // Find the candle that CREATED this level (swing high or swing low)
      let startTime = null;
      let foundValidLevel = false;
      const tolerance = 0.005; // 0.5% tolerance

      // For high patterns (BSL, EQH, BOS bullish): find swing high candle
      // For low patterns (SSL, EQL, BOS bearish): find swing low candle
      for (let i = 2; i < candles.length - 2; i++) {
        const candle = candles[i];
        const prevCandle1 = candles[i - 1];
        const prevCandle2 = candles[i - 2];
        const nextCandle1 = candles[i + 1];
        const nextCandle2 = candles[i + 2];

        if (isHighPattern) {
          // Check if this candle is a swing high (higher than surrounding candles)
          const isSwingHigh = candle.high >= prevCandle1.high &&
                             candle.high >= prevCandle2.high &&
                             candle.high >= nextCandle1.high &&
                             candle.high >= nextCandle2.high;

          if (isSwingHigh && Math.abs(candle.high - price) / price < tolerance) {
            startTime = candle.time;
            foundValidLevel = true;
            break;
          }
        } else if (isLowPattern) {
          // Check if this candle is a swing low (lower than surrounding candles)
          const isSwingLow = candle.low <= prevCandle1.low &&
                            candle.low <= prevCandle2.low &&
                            candle.low <= nextCandle1.low &&
                            candle.low <= nextCandle2.low;

          if (isSwingLow && Math.abs(candle.low - price) / price < tolerance) {
            startTime = candle.time;
            foundValidLevel = true;
            break;
          }
        }
      }

      // If no swing point found, try simpler match (candle that touched this level)
      if (!startTime) {
        for (let i = 0; i < candles.length; i++) {
          const candle = candles[i];
          if (isHighPattern && Math.abs(candle.high - price) / price < tolerance) {
            startTime = candle.time;
            foundValidLevel = true;
            break;
          } else if (isLowPattern && Math.abs(candle.low - price) / price < tolerance) {
            startTime = candle.time;
            foundValidLevel = true;
            break;
          }
        }
      }

      // Skip this ray if we couldn't find a valid candle that created this level
      // This prevents showing rays for price levels that don't exist on the current chart
      if (!foundValidLevel) return;

      if (!startTime) return;

      try {
        const timeScale = chart.timeScale();
        let startX = timeScale.timeToCoordinate(startTime);
        const chartWidth = chartElement.clientWidth;

        // If start is off-screen left, start from left edge (x=0)
        if (startX === null || startX < 0) {
          startX = 0;
        }

        const rayWidth = Math.max(chartWidth - startX - 70, 50);

        // Create the horizontal ray line (solid line)
        const rayLine = document.createElement('div');
        rayLine.className = 'pattern-ray-overlay';
        rayLine.style.cssText = `
          position: absolute;
          left: ${startX}px;
          top: ${y}px;
          width: ${rayWidth}px;
          height: 2px;
          background: linear-gradient(to right, ${annotation.color}, ${annotation.color}80);
          pointer-events: none;
          z-index: 4;
          box-shadow: 0 0 4px ${annotation.color}60;
          display: flex;
          align-items: center;
          justify-content: center;
        `;

        // Create the annotation label centered on the ray line
        const labelEl = document.createElement('div');
        labelEl.className = 'pattern-ray-overlay';
        labelEl.style.cssText = `
          color: ${annotation.color};
          font-size: 10px;
          font-weight: 700;
          text-shadow: 0 1px 2px rgba(0,0,0,0.9);
          white-space: nowrap;
          padding: 2px 6px;
          background: rgba(0,0,0,0.8);
          border: 1px solid ${annotation.color};
          border-radius: 3px;
          pointer-events: none;
        `;
        labelEl.textContent = annotation.text;
        rayLine.appendChild(labelEl);

        // Create small circle at the start point (only if visible)
        if (startX > 0) {
          const startMarker = document.createElement('div');
          startMarker.className = 'pattern-ray-overlay';
          startMarker.style.cssText = `
            position: absolute;
            left: ${startX - 4}px;
            top: ${y - 4}px;
            width: 8px;
            height: 8px;
            background: ${annotation.color};
            border-radius: 50%;
            pointer-events: none;
            z-index: 5;
            box-shadow: 0 0 6px ${annotation.color};
          `;
          chartElement.appendChild(startMarker);
          rayOverlaysRef.current.push(startMarker);
        }

        chartElement.style.position = 'relative';
        chartElement.appendChild(rayLine);
        rayOverlaysRef.current.push(rayLine);
      } catch (e) {
        console.log('Error drawing ray:', e);
      }
    });
  }, []);

  // Draw pattern boxes on chart
  const drawPatternBoxes = useCallback((patterns, candles) => {
    if (!chartRef.current || !candlestickSeriesRef.current || !chartContainerRef.current || !patterns || patterns.length === 0) {
      return;
    }

    clearPatternOverlays();

    const chart = chartRef.current;
    const series = candlestickSeriesRef.current;
    const chartElement = chartContainerRef.current;

    const boxPatternTypes = ['bullish_order_block', 'bearish_order_block', 'bullish_fvg', 'bearish_fvg', 'optimal_trade_entry'];
    // Pattern types that are drawn as horizontal rays (skip them here)
    const rayPatternTypes = [
      'bos_bullish', 'bos_bearish',
      'choch_bullish', 'choch_bearish',
      'equal_highs', 'equal_lows',
      'liquidity_sweep_high', 'liquidity_sweep_low',
      'buyside_liquidity', 'sellside_liquidity'
    ];
    const markers = [];

    patterns.forEach((pattern, idx) => {
      const patternType = pattern.pattern_type;
      const patternInfo = PATTERN_TYPE_MAP[patternType] || { short: patternType, color: '#9ca3af', direction: 'neutral' };
      const patternTimeframe = pattern.timeframe || timeframe;
      const label = `${patternInfo.short} ${getTimeframeLabel(patternTimeframe)}`;

      const highPrice = pattern.high || pattern.price_high || pattern.price;
      const lowPrice = pattern.low || pattern.price_low || pattern.price;

      if (!highPrice && !lowPrice && !pattern.price) return;

      // Find where this FVG/OB pattern was created on the chart
      let patternTime;

      // For FVG: Find the candle that created the gap
      // FVG is formed when candle[i-1].low > candle[i+1].high (bearish) or
      // candle[i-1].high < candle[i+1].low (bullish)
      // The FVG zone is the gap between those levels

      if (patternType.includes('fvg')) {
        // Search for where this FVG gap exists in the candle data
        for (let i = 1; i < candles.length - 1; i++) {
          const prevCandle = candles[i - 1];
          const currCandle = candles[i];
          const nextCandle = candles[i + 1];

          if (patternType === 'bearish_fvg') {
            // Bearish FVG: gap between prev candle low and next candle high
            const gapHigh = prevCandle.low;
            const gapLow = nextCandle.high;
            if (gapHigh > gapLow &&
                Math.abs(gapHigh - highPrice) / highPrice < 0.005 &&
                Math.abs(gapLow - lowPrice) / lowPrice < 0.005) {
              patternTime = currCandle.time;
              break;
            }
          } else if (patternType === 'bullish_fvg') {
            // Bullish FVG: gap between prev candle high and next candle low
            const gapLow = prevCandle.high;
            const gapHigh = nextCandle.low;
            if (gapHigh > gapLow &&
                Math.abs(gapHigh - highPrice) / highPrice < 0.005 &&
                Math.abs(gapLow - lowPrice) / lowPrice < 0.005) {
              patternTime = currCandle.time;
              break;
            }
          }
        }
      }

      // For Order Blocks: Find the candle with matching high/low
      if (!patternTime && patternType.includes('order_block')) {
        for (let i = 0; i < candles.length; i++) {
          const candle = candles[i];
          // OB is defined by a specific candle's range
          if (Math.abs(candle.high - highPrice) / highPrice < 0.003 &&
              Math.abs(candle.low - lowPrice) / lowPrice < 0.003) {
            patternTime = candle.time;
            break;
          }
        }
      }

      // Fallback: Find candle where price touched this zone
      if (!patternTime && candles.length > 0) {
        for (let i = 0; i < candles.length; i++) {
          const candle = candles[i];
          // Check if candle wick/body touched this zone
          if (candle.high >= lowPrice && candle.low <= highPrice) {
            patternTime = candle.time;
            break;
          }
        }
      }

      // Last resort: position based on pattern index from backend
      if (!patternTime && candles.length > 0) {
        if (pattern.start_index !== undefined && pattern.start_index >= 0) {
          // Backend uses 200 candle window, map proportionally
          const ratio = pattern.start_index / 200;
          const chartIdx = Math.floor(candles.length * ratio);
          const startIdx = Math.max(0, Math.min(chartIdx, candles.length - 1));
          patternTime = candles[startIdx]?.time;
        } else {
          // Use a position near the end
          const recentIdx = Math.max(0, candles.length - 20);
          patternTime = candles[recentIdx]?.time;
        }
      }

      if (!patternTime) return;

      // Skip ray patterns - they're drawn separately as horizontal rays
      if (rayPatternTypes.includes(patternType)) {
        return;
      }

      if (!boxPatternTypes.includes(patternType)) {
        const price = pattern.price || highPrice || lowPrice;
        const isHigh = patternType.includes('high') || patternType.includes('bullish') || patternType === 'equal_highs';
        markers.push({
          time: patternTime,
          position: isHigh ? 'aboveBar' : 'belowBar',
          color: patternInfo.color,
          shape: 'circle',
          text: label,
          size: 1,
        });
        return;
      }

      let startTime = patternTime;
      // Extend FVG/OB zones to the right edge of the chart
      // This is correct ICT methodology - zones remain valid until filled/mitigated

      if (!startTime || !highPrice || !lowPrice) return;

      try {
        const timeScale = chart.timeScale();
        const chartWidth = chartElement.clientWidth;
        let x1 = timeScale.timeToCoordinate(startTime);
        const y1 = series.priceToCoordinate(highPrice);
        const y2 = series.priceToCoordinate(lowPrice);

        // If start is off-screen left, start from left edge
        if (x1 === null || x1 < 0) {
          x1 = 0;
        }

        if (y1 === null || y2 === null) return;

        // Extend to right edge of chart (minus price scale area ~60px)
        const rightEdge = chartWidth - 60;
        const width = Math.max(rightEdge - x1, 50);
        const height = Math.max(Math.abs(y2 - y1), 15);

        const overlay = document.createElement('div');
        overlay.className = 'pattern-box-overlay';
        overlay.style.cssText = `
          position: absolute;
          left: ${x1}px;
          top: ${Math.min(y1, y2)}px;
          width: ${width}px;
          height: ${height}px;
          background-color: ${patternInfo.color}20;
          border-left: 3px solid ${patternInfo.color};
          border-top: 1px solid ${patternInfo.color}60;
          border-bottom: 1px solid ${patternInfo.color}60;
          pointer-events: none;
          z-index: 3;
          display: flex;
          align-items: center;
          justify-content: center;
        `;

        // Label centered within the box
        const labelEl = document.createElement('div');
        labelEl.style.cssText = `
          color: ${patternInfo.color};
          font-size: 10px;
          font-weight: 700;
          text-shadow: 0 1px 2px rgba(0,0,0,0.9);
          white-space: nowrap;
          padding: 2px 6px;
          background: rgba(0,0,0,0.7);
          border-radius: 3px;
          pointer-events: none;
        `;
        labelEl.textContent = label;

        chartElement.style.position = 'relative';
        overlay.appendChild(labelEl);
        chartElement.appendChild(overlay);
        patternOverlaysRef.current.push(overlay);
      } catch (e) {
        console.log('Error drawing pattern box:', e);
      }
    });

    if (markers.length > 0) {
      try {
        markers.sort((a, b) => a.time - b.time);
        series.setMarkers(markers);
      } catch (e) {
        console.log('Error setting markers:', e);
      }
    }

    // Also draw horizontal rays for BOS/CHoCH, liquidity, equal highs/lows
    drawHorizontalRays(patterns, candles);
  }, [timeframe, clearPatternOverlays, drawHorizontalRays]);

  // Initialize chart and load data
  useEffect(() => {
    if (!chartContainerRef.current) return;

    let chart = null;
    let candlestickSeries = null;

    const initChart = () => {
      // Clean up existing chart
      if (chartRef.current) {
        try {
          chartRef.current.remove();
        } catch (e) {}
        chartRef.current = null;
        candlestickSeriesRef.current = null;
      }

      chart = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: 500,
        layout: {
          background: { type: ColorType.Solid, color: '#ffffff' },
          textColor: '#333333',
        },
        grid: {
          vertLines: { visible: false },
          horzLines: { visible: false },
        },
        crosshair: {
          mode: CrosshairMode.Normal,
          vertLine: { color: '#6366f1', width: 1, style: 2, labelBackgroundColor: '#6366f1' },
          horzLine: { color: '#6366f1', width: 1, style: 2, labelBackgroundColor: '#6366f1' },
        },
        rightPriceScale: {
          borderColor: '#e5e7eb',
          scaleMargins: { top: 0.1, bottom: 0.1 },
        },
        timeScale: {
          borderColor: '#e5e7eb',
          timeVisible: true,
          secondsVisible: false,
        },
      });

      candlestickSeries = chart.addCandlestickSeries({
        upColor: '#22c55e',
        downColor: '#000000',
        borderUpColor: '#22c55e',
        borderDownColor: '#000000',
        wickUpColor: '#22c55e',
        wickDownColor: '#000000',
      });

      chartRef.current = chart;
      candlestickSeriesRef.current = candlestickSeries;

      // Resize handler
      const handleResize = () => {
        if (chartContainerRef.current && chart) {
          chart.applyOptions({ width: chartContainerRef.current.clientWidth });
        }
      };
      window.addEventListener('resize', handleResize);

      // Scroll handler for pattern redraw (both boxes and rays)
      chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
        if (window.patternUpdateTimeout) clearTimeout(window.patternUpdateTimeout);
        window.patternUpdateTimeout = setTimeout(() => {
          if (analysisPatternsRef.current.length > 0 && candlesDataRef.current.length > 0) {
            // Clear and redraw all overlays
            clearPatternOverlays();
            drawPatternBoxes(analysisPatternsRef.current, candlesDataRef.current);
          }
        }, 100);
      });

      return () => {
        window.removeEventListener('resize', handleResize);
        if (window.patternUpdateTimeout) clearTimeout(window.patternUpdateTimeout);
      };
    };

    const loadData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Request more candles for higher timeframes to show more history
        const candleLimit = ['D1', 'W1', 'MN'].includes(timeframe) ? 500 :
                           ['H4', 'H1'].includes(timeframe) ? 300 : 200;
        const ohlcvData = await getLiveOHLCV(symbol, timeframe, candleLimit);

        if (ohlcvData?.candles?.length > 0) {
          const candles = ohlcvData.candles;
          candlesDataRef.current = candles;

          if (candlestickSeriesRef.current) {
            candlestickSeriesRef.current.setData(candles);
            lastCandleRef.current = candles[candles.length - 1];
            setCurrentPrice(lastCandleRef.current.close);

            if (candles.length > 1) {
              const prevClose = candles[candles.length - 2].close;
              setPriceChange(((lastCandleRef.current.close - prevClose) / prevClose) * 100);
            }

            chartRef.current?.timeScale().fitContent();
          }
        } else {
          setError('No data available');
        }

        // Load analysis
        try {
          const analysisData = await getSignalAnalysis(symbol, timeframe);
          setAnalysis(analysisData);

          // Only draw patterns and price lines if ML is trained
          const isMlTrained = analysisData?.ml_status === 'trained';

          if (isMlTrained && analysisData?.patterns && analysisData.patterns.length > 0 && candlesDataRef.current.length > 0) {
            const signalDirection = analysisData?.signal?.direction?.toLowerCase();
            const totalCandles = candlesDataRef.current.length;
            const currentPrice = candlesDataRef.current[candlesDataRef.current.length - 1]?.close || 0;

            // Get visible price range from candles
            const visibleHigh = Math.max(...candlesDataRef.current.map(c => c.high));
            const visibleLow = Math.min(...candlesDataRef.current.map(c => c.low));
            const priceRange = visibleHigh - visibleLow;

            // Map timeframe to priority (current TF gets highest priority)
            const getTfPriority = (patternTf) => {
              if (patternTf === timeframe) return 0;  // Current timeframe - highest priority
              // Higher timeframes are more significant
              const tfOrder = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1'];
              const currentIdx = tfOrder.indexOf(timeframe);
              const patternIdx = tfOrder.indexOf(patternTf);
              if (patternIdx > currentIdx) return 1;  // Higher TF - second priority
              return 2;  // Lower TF - lowest priority
            };

            // Filter patterns to only those within visible price range (with some margin)
            const margin = priceRange * 0.1;  // 10% margin
            const visiblePatterns = analysisData.patterns.filter(p => {
              const price = p.price || ((p.high || 0) + (p.low || 0)) / 2;
              return price >= (visibleLow - margin) && price <= (visibleHigh + margin);
            });

            // Sort patterns: prioritize current timeframe, then by distance from current price
            const sortedPatterns = [...visiblePatterns].sort((a, b) => {
              // First: prioritize current timeframe patterns
              const tfPriorityA = getTfPriority(a.timeframe);
              const tfPriorityB = getTfPriority(b.timeframe);
              if (tfPriorityA !== tfPriorityB) return tfPriorityA - tfPriorityB;

              // Second: prioritize BOS/CHoCH patterns
              const isBosChochA = a.pattern_type?.includes('bos') || a.pattern_type?.includes('choch');
              const isBosChochB = b.pattern_type?.includes('bos') || b.pattern_type?.includes('choch');
              if (isBosChochA && !isBosChochB) return -1;
              if (!isBosChochA && isBosChochB) return 1;

              // Third: sort by distance from current price (nearest first)
              const priceA = a.price || ((a.high || 0) + (a.low || 0)) / 2;
              const priceB = b.price || ((b.high || 0) + (b.low || 0)) / 2;
              const distA = Math.abs(priceA - currentPrice);
              const distB = Math.abs(priceB - currentPrice);
              return distA - distB;
            });

            let relevantPatterns;

            // Separate box patterns (FVG, OB) from ray patterns (BOS, CHoCH, liquidity, etc.)
            const rayPatternTypes = [
              'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
              'equal_highs', 'equal_lows', 'liquidity_sweep_high', 'liquidity_sweep_low',
              'buyside_liquidity', 'sellside_liquidity'
            ];

            const boxPatterns = sortedPatterns.filter(p => !rayPatternTypes.includes(p.pattern_type));
            const rayPatterns = sortedPatterns.filter(p => rayPatternTypes.includes(p.pattern_type));

            // Only filter if we have a clear bullish or bearish signal
            // For WAIT, neutral, or no signal - show all patterns
            if (signalDirection === 'bullish') {
              // Filter to bullish patterns only
              const filteredBoxes = boxPatterns.filter(p => {
                const pt = p.pattern_type;
                return pt.includes('bullish');
              }).slice(0, 5);
              const filteredRays = rayPatterns.filter(p => {
                const pt = p.pattern_type;
                return pt.includes('bullish') || pt === 'liquidity_sweep_low' || pt === 'equal_lows' || pt === 'sellside_liquidity';
              }).slice(0, 8);  // Allow more ray patterns
              relevantPatterns = [...filteredBoxes, ...filteredRays];
            } else if (signalDirection === 'bearish') {
              // Filter to bearish patterns only
              const filteredBoxes = boxPatterns.filter(p => {
                const pt = p.pattern_type;
                return pt.includes('bearish');
              }).slice(0, 5);
              const filteredRays = rayPatterns.filter(p => {
                const pt = p.pattern_type;
                return pt.includes('bearish') || pt === 'liquidity_sweep_high' || pt === 'equal_highs' || pt === 'buyside_liquidity';
              }).slice(0, 8);  // Allow more ray patterns
              relevantPatterns = [...filteredBoxes, ...filteredRays];
            } else {
              // WAIT, neutral, or undefined - show mix of patterns
              // Always include BOS/CHoCH patterns as they're key market structure
              const bosChochPatterns = rayPatterns.filter(p =>
                p.pattern_type?.includes('bos') || p.pattern_type?.includes('choch')
              );
              const otherRays = rayPatterns.filter(p =>
                !p.pattern_type?.includes('bos') && !p.pattern_type?.includes('choch')
              );

              // Deduplicate rays by price level (keep only unique price levels)
              const uniqueRays = [];
              const seenPrices = new Set();
              for (const ray of otherRays) {
                const price = ray.price || ray.high || ray.low;
                const roundedPrice = Math.round(price / 100) * 100; // Round to nearest 100
                if (!seenPrices.has(roundedPrice)) {
                  seenPrices.add(roundedPrice);
                  uniqueRays.push(ray);
                }
              }

              relevantPatterns = [
                ...bosChochPatterns.slice(0, 2),  // BOS/CHoCH (up to 2)
                ...boxPatterns.slice(0, 4),        // Box patterns
                ...uniqueRays.slice(0, 4)          // Other rays (deduplicated)
              ];
            }

            analysisPatternsRef.current = relevantPatterns;
            setTimeout(() => drawPatternBoxes(relevantPatterns, candlesDataRef.current), 200);
          } else {
            // Clear patterns if ML not trained or no patterns
            analysisPatternsRef.current = [];
            clearPatternOverlays();
          }

          // Add price lines only if ML is trained
          if (isMlTrained && analysisData?.signal && candlestickSeriesRef.current) {
            const signal = analysisData.signal;
            priceLinesRef.current.forEach(line => {
              try { candlestickSeriesRef.current.removePriceLine(line); } catch (e) {}
            });
            priceLinesRef.current = [];

            if (signal.entry_zone?.length >= 2) {
              priceLinesRef.current.push(candlestickSeriesRef.current.createPriceLine({
                price: (signal.entry_zone[0] + signal.entry_zone[1]) / 2,
                color: '#3b82f6', lineWidth: 2, lineStyle: 0, axisLabelVisible: true, title: 'Entry',
              }));
            }
            if (signal.stop_loss) {
              priceLinesRef.current.push(candlestickSeriesRef.current.createPriceLine({
                price: signal.stop_loss,
                color: '#ef4444', lineWidth: 2, lineStyle: 2, axisLabelVisible: true, title: 'SL',
              }));
            }
            signal.take_profit?.forEach((tp, idx) => {
              priceLinesRef.current.push(candlestickSeriesRef.current.createPriceLine({
                price: tp,
                color: '#22c55e', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: `TP${idx + 1}`,
              }));
            });
          } else if (!isMlTrained) {
            // Clear price lines if ML not trained
            priceLinesRef.current.forEach(line => {
              try { candlestickSeriesRef.current?.removePriceLine(line); } catch (e) {}
            });
            priceLinesRef.current = [];
          }
        } catch (err) {
          console.log('Analysis not available:', err.message);
        }

        setLastUpdate(new Date());
      } catch (err) {
        console.error('Load error:', err);
        setError(err.message || 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    initChart();
    loadData();

    return () => {
      clearPatternOverlays();
      if (chartRef.current) {
        try { chartRef.current.remove(); } catch (e) {}
        chartRef.current = null;
        candlestickSeriesRef.current = null;
      }
    };
  }, [symbol, timeframe, drawPatternBoxes, clearPatternOverlays]);

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      if (wsRef.current) wsRef.current.close();

      const ws = new WebSocket(getWebSocketUrl(symbol));

      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => setIsConnected(false);
      ws.onerror = () => setIsConnected(false);

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'price_update' && candlestickSeriesRef.current) {
            const now = Math.floor(Date.now() / 1000);
            const newCandle = {
              time: now,
              open: data.open,
              high: data.high,
              low: data.low,
              close: data.close,
            };

            if (lastCandleRef.current) {
              const tfSecs = timeframe === 'M1' ? 60 : timeframe === 'M5' ? 300 : timeframe === 'M15' ? 900 : 3600;
              if (now - lastCandleRef.current.time < tfSecs) {
                newCandle.time = lastCandleRef.current.time;
                newCandle.open = lastCandleRef.current.open;
                newCandle.high = Math.max(lastCandleRef.current.high, data.high);
                newCandle.low = Math.min(lastCandleRef.current.low, data.low);
              }
            }

            candlestickSeriesRef.current.update(newCandle);
            lastCandleRef.current = newCandle;
            setCurrentPrice(data.close);
            setLastUpdate(new Date());
          }
        } catch (err) {}
      };

      wsRef.current = ws;
    };

    connectWebSocket();
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, [symbol, timeframe]);

  const signal = analysis?.signal;
  const isBullish = signal?.direction === 'bullish';
  const isBearish = signal?.direction === 'bearish';

  return (
    <div className={`space-y-4 ${fullscreen ? 'fixed inset-0 z-50 bg-[#0a0a0f] p-4 overflow-auto' : ''}`}>
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="p-2 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-500/20">
            <BarChart2 className="w-6 h-6 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-white">{symbol}</h1>
              {currentPrice && (
                <span className="text-2xl font-bold text-white">{formatPrice(currentPrice)}</span>
              )}
              {priceChange !== 0 && (
                <span className={`text-sm font-semibold px-2 py-1 rounded ${
                  priceChange >= 0 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                }`}>
                  {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-400">
              {isConnected ? (
                <span className="flex items-center gap-1 text-emerald-400"><Wifi className="w-3 h-3" /> Live</span>
              ) : (
                <span className="flex items-center gap-1 text-yellow-400"><WifiOff className="w-3 h-3" /> Connecting...</span>
              )}
              {lastUpdate && <span>Updated: {lastUpdate.toLocaleTimeString()}</span>}
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Symbol Selector */}
          <div className="relative">
            <button
              onClick={() => setSymbolDropdownOpen(!symbolDropdownOpen)}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800/50 border border-slate-700/50 text-white hover:border-indigo-500/50 transition-all text-sm"
            >
              <span className="font-semibold">{symbol}</span>
              <ChevronDown className={`w-4 h-4 transition-transform ${symbolDropdownOpen ? 'rotate-180' : ''}`} />
            </button>

            {symbolDropdownOpen && (
              <div className="absolute top-full mt-2 left-0 w-56 max-h-72 overflow-auto rounded-xl bg-slate-800 border border-slate-700 shadow-2xl z-50">
                {Object.entries(SYMBOL_CATEGORIES).map(([category, symbols]) => (
                  <div key={category}>
                    <div className="px-3 py-1.5 text-xs font-semibold text-slate-400 bg-slate-900/50 sticky top-0">{category}</div>
                    <div className="grid grid-cols-2 gap-1 p-1">
                      {symbols.map((sym) => (
                        <button
                          key={sym}
                          onClick={() => { setSymbol(sym); setSymbolDropdownOpen(false); }}
                          className={`px-2 py-1.5 rounded text-xs font-medium transition-all ${
                            symbol === sym ? 'bg-indigo-500 text-white' : 'text-slate-300 hover:bg-slate-700'
                          }`}
                        >
                          {sym}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Timeframe Selector */}
          <div className="flex items-center bg-slate-800/50 rounded-lg border border-slate-700/50 p-0.5">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf.value}
                onClick={() => setTimeframe(tf.value)}
                className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                  timeframe === tf.value ? 'bg-indigo-500 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                {tf.label}
              </button>
            ))}
          </div>

          <button
            onClick={() => setShowLegend(!showLegend)}
            className={`p-2 rounded-lg border transition-all ${
              showLegend ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-400' : 'bg-slate-800/50 border-slate-700/50 text-slate-400'
            }`}
            title="Toggle Signal Panel"
          >
            <Eye className="w-4 h-4" />
          </button>

          <button
            onClick={() => setFullscreen(!fullscreen)}
            className="p-2 rounded-lg bg-slate-800/50 border border-slate-700/50 text-slate-400 hover:text-white transition-all"
            title="Fullscreen"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
        {/* Chart Area */}
        <div className={`${showLegend ? 'xl:col-span-3' : 'xl:col-span-4'}`}>
          <div className="card-dark rounded-xl overflow-hidden relative">
            <div
              ref={chartContainerRef}
              className="w-full bg-white"
              style={{ height: fullscreen ? 'calc(100vh - 200px)' : '500px' }}
            />

            {loading && (
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-10">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-10 h-10 border-3 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
                  <span className="text-slate-400 text-sm">Loading {symbol} data...</span>
                </div>
              </div>
            )}

            {error && !loading && (
              <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center gap-4 z-10">
                <AlertCircle className="w-12 h-12 text-red-400" />
                <span className="text-red-400">{error}</span>
              </div>
            )}
          </div>

          {/* ML Not Trained Notice */}
          {analysis?.ml_status === 'not_trained' && !loading && (
            <div className="mt-4 p-4 bg-amber-500/10 rounded-xl border border-amber-500/30">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center">
                  <AlertCircle className="w-5 h-5 text-amber-400" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-amber-400">AI Not Trained</h3>
                  <p className="text-xs text-slate-400 mt-1">
                    AI analysis is disabled. Go to Dashboard → ML Training Manager and train from ICT videos to enable pattern detection and signal generation.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* AI Analysis Summary - Based on ML's learned knowledge from ICT videos */}
          {analysis?.signal && analysis?.ml_status === 'trained' && !loading && (
            <div className="mt-4 p-4 bg-slate-900/50 rounded-xl border border-slate-700/50">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-400" />
                AI Analysis Summary
                <span className="text-xs text-slate-500 ml-auto">Based on ML's learned ICT knowledge</span>
              </h3>

              <div className="space-y-4 text-sm">
                {/* ML Reasoning Section - Shows WHY based on learned knowledge */}
                {analysis.ml_reasoning && (
                  <div className="p-3 rounded-lg bg-indigo-500/5 border border-indigo-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Eye className="w-4 h-4 text-indigo-400" />
                      <span className="font-semibold text-indigo-400">ML Analysis Reasoning</span>
                    </div>
                    <div className="text-slate-300 leading-relaxed whitespace-pre-line text-xs">
                      {analysis.ml_reasoning}
                    </div>
                  </div>
                )}

                {/* Entry Zone Explanation - From ML Knowledge */}
                {analysis.signal.entry_zone && (
                  <div className="p-3 rounded-lg bg-blue-500/5 border border-blue-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Crosshair className="w-4 h-4 text-blue-400" />
                      <span className="font-semibold text-blue-400">Entry Zone: {formatPrice(analysis.signal.entry_zone[0])} - {formatPrice(analysis.signal.entry_zone[1])}</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed text-xs">
                      {analysis.entry_exit_reasoning?.entry_reason ||
                        `Entry based on ${analysis.ml_patterns_detected?.join(', ') || 'market structure'} patterns detected by ML.`}
                    </p>
                  </div>
                )}

                {/* Stop Loss Explanation - From ML Knowledge */}
                {analysis.signal.stop_loss && (
                  <div className="p-3 rounded-lg bg-red-500/5 border border-red-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Shield className="w-4 h-4 text-red-400" />
                      <span className="font-semibold text-red-400">Stop Loss: {formatPrice(analysis.signal.stop_loss)}</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed text-xs">
                      {analysis.entry_exit_reasoning?.stop_reason ||
                        'Stop loss placed beyond key structure level based on ML analysis.'}
                    </p>
                  </div>
                )}

                {/* Take Profit Explanation - From ML Knowledge */}
                {analysis.signal.take_profit && analysis.signal.take_profit.length > 0 && (
                  <div className="p-3 rounded-lg bg-emerald-500/5 border border-emerald-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="w-4 h-4 text-emerald-400" />
                      <span className="font-semibold text-emerald-400">Take Profit Targets</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed mb-2 text-xs">
                      {analysis.entry_exit_reasoning?.target_reason ||
                        'Targets at liquidity levels identified by ML.'}
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mt-2">
                      {analysis.signal.take_profit.map((tp, idx) => (
                        <div key={idx} className="p-2 rounded bg-emerald-500/10 border border-emerald-500/30 text-center">
                          <div className="text-xs text-slate-400">TP{idx + 1}</div>
                          <div className="font-mono text-emerald-400 font-semibold">{formatPrice(tp)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Patterns Detected */}
                {analysis.patterns && analysis.patterns.length > 0 && (
                  <div className="p-3 rounded-lg bg-purple-500/5 border border-purple-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Eye className="w-4 h-4 text-purple-400" />
                      <span className="font-semibold text-purple-400">Patterns Detected: {analysis.patterns.length}</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {[...new Set(analysis.patterns.map(p => p.pattern_type))].map(patternType => {
                        const count = analysis.patterns.filter(p => p.pattern_type === patternType).length;
                        const mappedInfo = PATTERN_TYPE_MAP[patternType] || { short: patternType, color: '#9ca3af' };
                        return (
                          <span
                            key={patternType}
                            className="px-2 py-1 rounded-full text-xs font-medium"
                            style={{ backgroundColor: mappedInfo.color + '20', color: mappedInfo.color, border: `1px solid ${mappedInfo.color}40` }}
                          >
                            {count}x {mappedInfo.short}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Risk-Reward Info */}
                {analysis.signal.entry_zone && analysis.signal.stop_loss && analysis.signal.take_profit?.[0] && (
                  <div className="p-3 rounded-lg bg-indigo-500/5 border border-indigo-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <BarChart2 className="w-4 h-4 text-indigo-400" />
                      <span className="font-semibold text-indigo-400">Risk-Reward Analysis</span>
                    </div>
                    <div className="grid grid-cols-3 gap-3 text-center">
                      <div>
                        <div className="text-xs text-slate-400">Risk (pips)</div>
                        <div className="font-mono text-red-400 font-semibold">
                          {Math.abs(((analysis.signal.entry_zone[0] + analysis.signal.entry_zone[1]) / 2 - analysis.signal.stop_loss) * (symbol.includes('JPY') ? 100 : ['BTC', 'ETH'].includes(symbol) ? 1 : 10000)).toFixed(1)}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-400">Reward (TP1)</div>
                        <div className="font-mono text-emerald-400 font-semibold">
                          {Math.abs((analysis.signal.take_profit[0] - (analysis.signal.entry_zone[0] + analysis.signal.entry_zone[1]) / 2) * (symbol.includes('JPY') ? 100 : ['BTC', 'ETH'].includes(symbol) ? 1 : 10000)).toFixed(1)}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-400">R:R Ratio</div>
                        <div className="font-mono text-indigo-400 font-semibold">
                          1:{(Math.abs(analysis.signal.take_profit[0] - (analysis.signal.entry_zone[0] + analysis.signal.entry_zone[1]) / 2) / Math.abs((analysis.signal.entry_zone[0] + analysis.signal.entry_zone[1]) / 2 - analysis.signal.stop_loss)).toFixed(1)}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Signal Panel */}
        {showLegend && (
          <div className="xl:col-span-1 space-y-4">
            {signal && (
              <div className="card-dark rounded-xl p-4">
                <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                  <Target className="w-4 h-4 text-indigo-400" /> Trade Signal
                </h3>

                <div className={`p-3 rounded-lg mb-3 ${
                  isBullish ? 'bg-emerald-500/10 border border-emerald-500/20' :
                  isBearish ? 'bg-red-500/10 border border-red-500/20' :
                  'bg-slate-500/10 border border-slate-500/20'
                }`}>
                  <div className="flex items-center justify-between">
                    <span className={`text-lg font-bold ${
                      isBullish ? 'text-emerald-400' : isBearish ? 'text-red-400' : 'text-slate-400'
                    }`}>
                      {isBullish ? <TrendingUp className="w-5 h-5 inline mr-1" /> :
                       isBearish ? <TrendingDown className="w-5 h-5 inline mr-1" /> : null}
                      {signal.direction?.toUpperCase() || 'WAIT'}
                    </span>
                    <span className={`text-sm font-semibold ${
                      isBullish ? 'text-emerald-400' : isBearish ? 'text-red-400' : 'text-slate-400'
                    }`}>
                      {Math.round((signal.confidence || 0) * 100)}%
                    </span>
                  </div>
                </div>

                <div className="space-y-2 text-xs">
                  {signal.entry_zone && (
                    <div className="flex justify-between p-2 rounded bg-blue-500/10 border border-blue-500/20">
                      <span className="text-slate-400 flex items-center gap-1">
                        <Crosshair className="w-3 h-3 text-blue-400" /> Entry
                      </span>
                      <span className="font-mono text-blue-400">
                        {formatPrice(signal.entry_zone[0])} - {formatPrice(signal.entry_zone[1])}
                      </span>
                    </div>
                  )}

                  {signal.stop_loss && (
                    <div className="flex justify-between p-2 rounded bg-red-500/10 border border-red-500/20">
                      <span className="text-slate-400 flex items-center gap-1">
                        <Shield className="w-3 h-3 text-red-400" /> Stop Loss
                      </span>
                      <span className="font-mono text-red-400">{formatPrice(signal.stop_loss)}</span>
                    </div>
                  )}

                  {signal.take_profit?.map((tp, idx) => (
                    <div key={idx} className="flex justify-between p-2 rounded bg-emerald-500/10 border border-emerald-500/20">
                      <span className="text-slate-400 flex items-center gap-1">
                        <Target className="w-3 h-3 text-emerald-400" /> TP{idx + 1}
                      </span>
                      <span className="font-mono text-emerald-400">{formatPrice(tp)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ML Knowledge Status */}
            <div className="card-dark rounded-xl p-4">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Layers className="w-4 h-4 text-purple-400" /> ML Pattern Knowledge
              </h3>

              {analysis?.ml_status === 'trained' ? (
                <div className="space-y-3">
                  {/* Learned Patterns - Show all patterns from ml_confidence_scores */}
                  {analysis?.ml_confidence_scores && Object.keys(analysis.ml_confidence_scores).length > 0 && (
                    <div>
                      <div className="text-xs text-emerald-400 font-semibold mb-2 flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                        Patterns ML Has Learned ({Object.keys(analysis.ml_confidence_scores).length})
                      </div>
                      <div className="space-y-1.5">
                        {Object.entries(analysis.ml_confidence_scores)
                          .sort(([,a], [,b]) => b - a)  // Sort by confidence descending
                          .map(([pattern, confidence]) => {
                          const info = CONCEPT_INFO[pattern] || { name: pattern, short: pattern.toUpperCase(), color: '#9ca3af', description: 'Pattern' };
                          const isActivelyDetected = analysis.ml_patterns_detected?.includes(pattern);
                          return (
                            <div key={pattern} className={`flex items-center justify-between text-xs p-2 rounded-lg ${
                              isActivelyDetected
                                ? 'bg-emerald-500/10 border border-emerald-500/20'
                                : 'bg-slate-800/50 border border-slate-600/30'
                            }`}>
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded" style={{ backgroundColor: info.color + '40', border: `1px solid ${info.color}` }} />
                                <span className="text-white font-medium">{info.short}</span>
                                {isActivelyDetected && (
                                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400">ACTIVE</span>
                                )}
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                  <div
                                    className={`h-full rounded-full ${isActivelyDetected ? 'bg-emerald-400' : 'bg-slate-500'}`}
                                    style={{ width: `${Math.round(confidence * 100)}%` }}
                                  />
                                </div>
                                <span className={`font-mono w-10 text-right ${isActivelyDetected ? 'text-emerald-400' : 'text-slate-400'}`}>
                                  {Math.round(confidence * 100)}%
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Not Yet Learned Patterns */}
                  {analysis?.ml_patterns_not_learned && analysis.ml_patterns_not_learned.length > 0 && (
                    <div>
                      <div className="text-xs text-amber-400 font-semibold mb-2 flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                        Not Yet Learned
                      </div>
                      <div className="space-y-1.5">
                        {analysis.ml_patterns_not_learned.map((pattern) => {
                          const info = CONCEPT_INFO[pattern] || { name: pattern, short: pattern.toUpperCase(), color: '#6b7280', description: 'Pattern' };
                          return (
                            <div key={pattern} className="flex items-center justify-between text-xs p-2 rounded-lg bg-slate-800/50 border border-slate-700/50 opacity-60">
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded bg-slate-600 border border-slate-500" />
                                <span className="text-slate-400 font-medium">{info.short}</span>
                              </div>
                              <span className="text-slate-500 text-[10px]">Train to enable</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* ML Knowledge Summary */}
                  {analysis?.ml_knowledge_status && (
                    <div className="mt-3 pt-3 border-t border-slate-700/50">
                      <p className="text-[10px] text-slate-500 leading-relaxed">
                        {analysis.ml_knowledge_status}
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-4">
                  <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-amber-500/20 flex items-center justify-center">
                    <AlertCircle className="w-6 h-6 text-amber-400" />
                  </div>
                  <p className="text-xs text-amber-400 font-semibold mb-1">ML Not Trained</p>
                  <p className="text-[10px] text-slate-500">
                    Train the ML from YouTube videos to enable pattern detection.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {fullscreen && (
        <button
          onClick={() => setFullscreen(false)}
          className="fixed top-4 right-4 p-2 rounded-lg bg-slate-800 border border-slate-700 text-white hover:bg-slate-700 transition-colors z-50"
        >
          <X className="w-5 h-5" />
        </button>
      )}
    </div>
  );
}

export default LiveChart;
