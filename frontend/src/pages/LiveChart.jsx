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
  WifiOff,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Home,
  Minus,
  Plus
} from 'lucide-react';
import { getLiveOHLCV, getSignalAnalysis, getWebSocketUrl, predictPrice, getAvailablePlaylists } from '../services/api';

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
  // New ICT patterns from Audio-First Training
  'bullish_displacement': { short: 'DISP ↑', color: '#00e676', direction: 'bullish' },
  'bearish_displacement': { short: 'DISP ↓', color: '#ff1744', direction: 'bearish' },
  'bullish_ote': { short: 'OTE', color: '#aa00ff', direction: 'bullish' },
  'bearish_ote': { short: 'OTE', color: '#d500f9', direction: 'bearish' },
  'bullish_breaker': { short: 'BRK', color: '#00bfa5', direction: 'bullish' },
  'bearish_breaker': { short: 'BRK', color: '#ff6d00', direction: 'bearish' },
  'buy_stops': { short: 'BST', color: '#ff5252', direction: 'neutral' },
  'sell_stops': { short: 'SST', color: '#69f0ae', direction: 'neutral' },
  // Swing point markers
  'swing_high': { short: 'SH', color: '#ff7043', direction: 'neutral' },
  'swing_low': { short: 'SL', color: '#42a5f5', direction: 'neutral' },
  // Market structure labels (HH/HL/LH/LL)
  'higher_high': { short: 'HH', color: '#00e676', direction: 'bullish' },
  'higher_low': { short: 'HL', color: '#69f0ae', direction: 'bullish' },
  'lower_high': { short: 'LH', color: '#ff5252', direction: 'bearish' },
  'lower_low': { short: 'LL', color: '#ff8a80', direction: 'bearish' },
  // Mitigated order blocks
  'bullish_mitigation_block': { short: 'MB', color: '#26a69a', direction: 'bullish' },
  'bearish_mitigation_block': { short: 'MB', color: '#ff9800', direction: 'bearish' },
  // Premium/Discount zones
  'premium_zone': { short: 'PREM', color: '#ff5252', direction: 'bearish' },
  'discount_zone': { short: 'DISC', color: '#69f0ae', direction: 'bullish' },
  'equilibrium': { short: 'EQ', color: '#ffd740', direction: 'neutral' },
};

function LiveChart() {
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('M15');
  const [analysis, setAnalysis] = useState(null);
  const [mlPrediction, setMlPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [showLegend, setShowLegend] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [symbolDropdownOpen, setSymbolDropdownOpen] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [currentPrice, setCurrentPrice] = useState(null);
  const [priceChange, setPriceChange] = useState(0);
  const [playlists, setPlaylists] = useState([]);
  const [selectedPlaylist, setSelectedPlaylist] = useState('all');
  const [loadingHistory, setLoadingHistory] = useState(false);  // Loading state for historical data
  const [atLeftEdge, setAtLeftEdge] = useState(false);  // Track if user is at left edge

  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candlestickSeriesRef = useRef(null);
  const wsRef = useRef(null);
  const lastCandleRef = useRef(null);
  const priceLinesRef = useRef([]);
  const placedLabelsRef = useRef([]); // Shared label position tracking for anti-overlap
  const patternOverlaysRef = useRef([]);
  const rayOverlaysRef = useRef([]);
  const candlesDataRef = useRef([]);
  const analysisPatternsRef = useRef([]);
  const allApiPatternsRef = useRef([]);      // ALL patterns from API (unfiltered)
  const signalDirectionRef = useRef(null);   // Signal direction for pattern filtering
  const isLoadingHistoryRef = useRef(false); // Prevent multiple historical data requests
  const oldestTimestampRef = useRef(null);   // Track oldest candle for scrollback

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

  // Chart control functions (zoom in, zoom out, reset)
  const handleZoomIn = useCallback(() => {
    if (!chartRef.current) return;
    const timeScale = chartRef.current.timeScale();
    const currentRange = timeScale.getVisibleLogicalRange();
    if (currentRange) {
      const center = (currentRange.from + currentRange.to) / 2;
      const newRange = (currentRange.to - currentRange.from) * 0.7; // Zoom in by 30%
      timeScale.setVisibleLogicalRange({
        from: center - newRange / 2,
        to: center + newRange / 2,
      });
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (!chartRef.current) return;
    const timeScale = chartRef.current.timeScale();
    const currentRange = timeScale.getVisibleLogicalRange();
    if (currentRange) {
      const center = (currentRange.from + currentRange.to) / 2;
      const newRange = (currentRange.to - currentRange.from) * 1.4; // Zoom out by 40%
      const candles = candlesDataRef.current;
      // Clamp to available data
      const from = Math.max(0, center - newRange / 2);
      const to = Math.min(candles.length - 1, center + newRange / 2);
      timeScale.setVisibleLogicalRange({ from, to });
    }
  }, []);

  const handleResetChart = useCallback(() => {
    if (!chartRef.current || !candlesDataRef.current.length) return;
    const timeScale = chartRef.current.timeScale();
    const candles = candlesDataRef.current;
    // Show last 200 candles (or all if less)
    const visibleCount = Math.min(200, candles.length);
    timeScale.setVisibleLogicalRange({
      from: candles.length - visibleCount,
      to: candles.length - 1,
    });
  }, []);

  const handleFitAll = useCallback(() => {
    if (!chartRef.current || !candlesDataRef.current.length) return;
    const timeScale = chartRef.current.timeScale();
    const candles = candlesDataRef.current;
    // Fit all loaded candles in view
    timeScale.setVisibleLogicalRange({
      from: 0,
      to: candles.length - 1,
    });
  }, []);

  const handleScrollToLatest = useCallback(() => {
    if (!chartRef.current) return;
    chartRef.current.timeScale().scrollToRealTime();
  }, []);

  // Fetch available playlists for the dropdown
  useEffect(() => {
    getAvailablePlaylists()
      .then(data => {
        if (Array.isArray(data)) {
          setPlaylists(data);
        }
      })
      .catch(err => console.warn('Failed to fetch playlists:', err));
  }, []);

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

  // Reusable: filter API patterns for the current visible price range
  const getRelevantPatterns = useCallback((allPatterns, candles, signalDirection) => {
    if (!allPatterns || allPatterns.length === 0 || !candles || candles.length === 0) return [];

    const visibleHigh = Math.max(...candles.map(c => c.high));
    const visibleLow = Math.min(...candles.map(c => c.low));
    const priceRange = visibleHigh - visibleLow;
    const margin = priceRange * 0.15;  // 15% margin for better coverage
    // Use midpoint of visible range as reference price for sorting
    const currentPrice = (visibleHigh + visibleLow) / 2;

    const getTfPriority = (patternTf) => {
      if (patternTf === timeframe) return 0;
      const tfOrder = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1'];
      const currentIdx = tfOrder.indexOf(timeframe);
      const patternIdx = tfOrder.indexOf(patternTf);
      if (patternIdx > currentIdx) return 1;
      return 2;
    };

    const allVisible = allPatterns.filter(p => {
      if (p.filled) return false;
      const price = p.price || ((p.high || 0) + (p.low || 0)) / 2;
      return price >= (visibleLow - margin) && price <= (visibleHigh + margin);
    });

    allVisible.sort((a, b) => {
      const midA = a.price || ((a.high || 0) + (a.low || 0)) / 2;
      const midB = b.price || ((b.high || 0) + (b.low || 0)) / 2;
      return Math.abs(midA - currentPrice) - Math.abs(midB - currentPrice);
    });

    // Use last candle close as the actual market price for directional filters
    const lastClose = candles[candles.length - 1]?.close || currentPrice;
    const filtered = allVisible.filter(p => {
      const pt = p.pattern_type;
      const pPrice = p.price || ((p.high || 0) + (p.low || 0)) / 2;
      // Only apply directional filters if viewing near current price
      const isNearCurrent = Math.abs(lastClose - currentPrice) / currentPrice < 0.15;
      if (isNearCurrent) {
        if (pt === 'buyside_liquidity' && pPrice < lastClose * 0.98) return false;
        if (pt === 'sellside_liquidity' && pPrice > lastClose * 1.02) return false;
        if (pt === 'equal_highs' && pPrice < lastClose) return false;
        if (pt === 'equal_lows' && pPrice > lastClose) return false;
      }
      return true;
    });

    const typeCounts = {};
    const seenZones = [];
    const visiblePatterns = [];
    for (const p of filtered) {
      const pt = p.pattern_type || '';
      const baseType = pt.replace('bullish_', '').replace('bearish_', '');
      const mid = p.price || ((p.high || 0) + (p.low || 0)) / 2;
      typeCounts[baseType] = (typeCounts[baseType] || 0) + 1;
      const isMarkerType = ['swing_high', 'swing_low', 'higher_high', 'higher_low', 'lower_high', 'lower_low'].includes(pt);
      if (typeCounts[baseType] > (isMarkerType ? 4 : 2)) continue;
      const isBoxType = ['fvg', 'order_block', 'displacement', 'ote', 'breaker'].some(t => baseType.includes(t));
      if (isBoxType) {
        if (seenZones.some(z => Math.abs(z - mid) / mid < 0.03)) continue;
        seenZones.push(mid);
      }
      visiblePatterns.push(p);
    }

    const sortedPatterns = [...visiblePatterns].sort((a, b) => {
      const tpA = getTfPriority(a.timeframe), tpB = getTfPriority(b.timeframe);
      if (tpA !== tpB) return tpA - tpB;
      const bosA = a.pattern_type?.includes('bos') || a.pattern_type?.includes('choch');
      const bosB = b.pattern_type?.includes('bos') || b.pattern_type?.includes('choch');
      if (bosA && !bosB) return -1;
      if (!bosA && bosB) return 1;
      const pA = a.price || ((a.high || 0) + (a.low || 0)) / 2;
      const pB = b.price || ((b.high || 0) + (b.low || 0)) / 2;
      return Math.abs(pA - currentPrice) - Math.abs(pB - currentPrice);
    });

    const rayTypes = [
      'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
      'equal_highs', 'equal_lows', 'liquidity_sweep_high', 'liquidity_sweep_low',
      'buyside_liquidity', 'sellside_liquidity', 'buy_stops', 'sell_stops',
      'equilibrium',
    ];
    const boxPats = sortedPatterns.filter(p => !rayTypes.includes(p.pattern_type));
    const rayPats = sortedPatterns.filter(p => rayTypes.includes(p.pattern_type));

    if (signalDirection === 'bullish') {
      return [
        ...boxPats.filter(p => p.pattern_type.includes('bullish')).slice(0, 5),
        ...rayPats.filter(p => {
          const pt = p.pattern_type;
          return pt.includes('bullish') || pt === 'liquidity_sweep_low' || pt === 'equal_lows' || pt === 'sellside_liquidity' || pt === 'sell_stops';
        }).slice(0, 6),
      ];
    } else if (signalDirection === 'bearish') {
      return [
        ...boxPats.filter(p => p.pattern_type.includes('bearish')).slice(0, 5),
        ...rayPats.filter(p => {
          const pt = p.pattern_type;
          return pt.includes('bearish') || pt === 'liquidity_sweep_high' || pt === 'equal_highs' || pt === 'buyside_liquidity' || pt === 'buy_stops';
        }).slice(0, 6),
      ];
    } else {
      const bosCh = rayPats.filter(p => p.pattern_type?.includes('bos') || p.pattern_type?.includes('choch'));
      const other = rayPats.filter(p => !p.pattern_type?.includes('bos') && !p.pattern_type?.includes('choch'));
      const uniqRays = [];
      const seen = new Set();
      for (const r of other) {
        const rp = Math.round((r.price || r.high || r.low) / 100) * 100;
        if (!seen.has(rp)) { seen.add(rp); uniqRays.push(r); }
      }
      // Separate lightweight marker patterns from box patterns
      const markerTypes = ['swing_high', 'swing_low', 'higher_high', 'higher_low', 'lower_high', 'lower_low'];
      const markerPats = sortedPatterns.filter(p => markerTypes.includes(p.pattern_type));
      const actualBoxPats = boxPats.filter(p => !markerTypes.includes(p.pattern_type));
      return [...bosCh.slice(0, 2), ...actualBoxPats.slice(0, 6), ...uniqRays.slice(0, 6), ...markerPats.slice(0, 8)];
    }
  }, [timeframe]);

  // Draw horizontal rays for BOS/CHoCH, liquidity, and equal highs/lows
  const drawHorizontalRays = useCallback((patterns, candles) => {
    if (!chartRef.current || !candlestickSeriesRef.current || !chartContainerRef.current || !patterns || patterns.length === 0 || candles.length === 0) {
      return;
    }

    const chart = chartRef.current;
    const series = candlestickSeriesRef.current;
    const chartElement = chartContainerRef.current;
    const chartHeight = chartElement.clientHeight || 500;

    const LABEL_HEIGHT = 20;
    const LABEL_MIN_GAP = 4;

    // Helper: find non-overlapping Y using shared ref (max 60px displacement)
    const findNonOverlappingYRay = (desiredY) => {
      const placed = placedLabelsRef.current;
      let y = desiredY;
      let attempts = 0;
      const maxDisplacement = 60;
      while (attempts < 8) {
        const overlaps = placed.some(p => Math.abs(y - p.y) < (LABEL_HEIGHT + LABEL_MIN_GAP));
        if (!overlaps) break;
        const offset = (Math.floor(attempts / 2) + 1) * (LABEL_HEIGHT + LABEL_MIN_GAP) * (attempts % 2 === 0 ? 1 : -1);
        if (Math.abs(offset) > maxDisplacement) break;
        y = desiredY + offset;
        attempts++;
      }
      placed.push({ y, h: LABEL_HEIGHT });
      return y;
    };

    // Pattern types that should be drawn as horizontal rays
    const rayPatternTypes = [
      'bos_bullish', 'bos_bearish',
      'choch_bullish', 'choch_bearish',
      'equal_highs', 'equal_lows',
      'liquidity_sweep_high', 'liquidity_sweep_low',
      'buyside_liquidity', 'sellside_liquidity',
      'buy_stops', 'sell_stops',
      'equilibrium',
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
        'buy_stops': { text: 'BST 🎯', color: '#ff5252' },
        'sell_stops': { text: 'SST 🎯', color: '#69f0ae' },
        'equilibrium': { text: 'EQ (50%)', color: '#ffd740' },
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

      // Equilibrium: special case — always span from left of chart
      if (patternType === 'equilibrium') {
        try {
          const timeScale = chart.timeScale();
          const chartWidth = chartElement.clientWidth;
          const rightEdge = chartWidth - 70;
          const startX = 0;
          const rayWidth = rightEdge;
          const annotation = getAnnotation(patternType);
          const lineEl = document.createElement('div');
          lineEl.className = 'ray-overlay';
          lineEl.style.cssText = `
            position: absolute; left: ${startX}px; top: ${y - 1}px;
            width: ${rayWidth}px; height: 2px;
            background: linear-gradient(to right, ${annotation.color}40, ${annotation.color}80, ${annotation.color}40);
            border-top: 1px dashed ${annotation.color}60;
            pointer-events: none; z-index: 4;
          `;
          const labelEl = document.createElement('div');
          const labelY = findNonOverlappingYRay(y - LABEL_HEIGHT / 2);
          labelEl.className = 'ray-overlay';
          labelEl.style.cssText = `
            position: absolute; right: 75px; top: ${labelY}px;
            color: ${annotation.color}; font-size: 10px; font-weight: 700;
            text-shadow: 0 1px 2px rgba(0,0,0,0.9);
            padding: 1px 5px; background: rgba(0,0,0,0.8);
            border: 1px solid ${annotation.color}60; border-radius: 3px;
            pointer-events: none; z-index: 6; white-space: nowrap;
          `;
          labelEl.textContent = `${annotation.text} ${pattern.timeframe || ''}`;
          chartElement.appendChild(lineEl);
          chartElement.appendChild(labelEl);
          rayOverlaysRef.current.push(lineEl);
          rayOverlaysRef.current.push(labelEl);
        } catch (e) { /* skip */ }
        return;
      }

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

        const rightEdge = chartWidth - 70;
        // If start is off-screen left, show a shorter ray from the left portion
        if (startX === null || startX < 0) {
          startX = 0;
        }

        // Cap ray width so it doesn't span the entire chart
        const maxRayWidth = Math.min(rightEdge * 0.6, 500);
        const rawRayWidth = rightEdge - startX;
        const rayWidth = Math.max(Math.min(rawRayWidth, maxRayWidth), 50);
        // Shift start so the ray ends at the right edge
        startX = rightEdge - rayWidth;

        // Anti-overlap: adjust Y for ray label
        const adjustedRayY = findNonOverlappingYRay(y);
        const labelOffsetY = adjustedRayY - y; // how much the label shifted

        // Create the horizontal ray line (solid line) - always at true price
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
        `;

        // Create the annotation label - offset to avoid overlap
        const labelEl = document.createElement('div');
        labelEl.className = 'pattern-ray-overlay';
        labelEl.style.cssText = `
          position: absolute;
          left: ${startX + rayWidth / 2 - 30}px;
          top: ${adjustedRayY - 10}px;
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
          z-index: 6;
        `;
        labelEl.textContent = annotation.text;

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
        chartElement.appendChild(labelEl);
        rayOverlaysRef.current.push(rayLine);
        rayOverlaysRef.current.push(labelEl);
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

    const boxPatternTypes = [
      'bullish_order_block', 'bearish_order_block',
      'bullish_fvg', 'bearish_fvg',
      'optimal_trade_entry',
      'bullish_ote', 'bearish_ote',
      'bullish_displacement', 'bearish_displacement',
      'bullish_breaker', 'bearish_breaker',
      'bullish_mitigation_block', 'bearish_mitigation_block',
      'premium_zone', 'discount_zone',
    ];
    // Pattern types that are drawn as horizontal rays (skip them here)
    const rayPatternTypes = [
      'bos_bullish', 'bos_bearish',
      'choch_bullish', 'choch_bearish',
      'equal_highs', 'equal_lows',
      'liquidity_sweep_high', 'liquidity_sweep_low',
      'buyside_liquidity', 'sellside_liquidity',
      'buy_stops', 'sell_stops',
      'equilibrium',
    ];
    const markers = [];
    // Reset shared label position tracking
    placedLabelsRef.current = [];
    const LABEL_HEIGHT = 20;
    const LABEL_MIN_GAP = 4;

    // Helper: find a non-overlapping Y position for a label (max 60px displacement)
    const findNonOverlappingY = (desiredY, labelH = LABEL_HEIGHT) => {
      const placed = placedLabelsRef.current;
      let y = desiredY;
      let attempts = 0;
      const maxDisplacement = 60;
      while (attempts < 8) {
        const overlaps = placed.some(p =>
          Math.abs(y - p.y) < (labelH + LABEL_MIN_GAP)
        );
        if (!overlaps) break;
        const offset = (Math.floor(attempts / 2) + 1) * (LABEL_HEIGHT + LABEL_MIN_GAP) * (attempts % 2 === 0 ? 1 : -1);
        if (Math.abs(offset) > maxDisplacement) break;
        y = desiredY + offset;
        attempts++;
      }
      placed.push({ y, h: labelH });
      return y;
    };

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

        if (y1 === null || y2 === null) return;

        // Extend to right edge of chart (minus price scale area ~60px)
        const rightEdge = chartWidth - 60;

        // If pattern start is off-screen left or null, show compact zone on the right
        if (x1 === null || x1 < 0) {
          x1 = rightEdge - 120;
        }

        // Cap maximum box width — keep boxes compact
        const maxWidth = 150;
        const rawWidth = rightEdge - x1;
        const width = Math.max(Math.min(rawWidth, maxWidth), 40);
        // Shift x1 right so the box ends at the right edge
        x1 = rightEdge - width;

        const height = Math.max(Math.abs(y2 - y1), 15);

        const boxTop = Math.min(y1, y2);
        const isMitigated = patternType.includes('mitigation_block');
        const isZone = patternType === 'premium_zone' || patternType === 'discount_zone';
        const bgAlpha = isZone ? '0a' : isMitigated ? '12' : '20';
        const borderStyle = isMitigated ? 'dashed' : 'solid';
        const boxOpacity = isMitigated ? 'opacity: 0.5;' : isZone ? 'opacity: 0.25;' : '';
        const overlay = document.createElement('div');
        overlay.className = 'pattern-box-overlay';
        overlay.style.cssText = `
          position: absolute;
          left: ${x1}px;
          top: ${boxTop}px;
          width: ${width}px;
          height: ${height}px;
          background-color: ${patternInfo.color}${bgAlpha};
          border-left: 3px ${borderStyle} ${patternInfo.color};
          border-top: 1px ${borderStyle} ${patternInfo.color}60;
          border-bottom: 1px ${borderStyle} ${patternInfo.color}60;
          pointer-events: none;
          z-index: 3;
          ${boxOpacity}
        `;

        // Label positioned at right side of box, with anti-overlap
        const desiredLabelY = boxTop + height / 2 - LABEL_HEIGHT / 2;
        const adjustedLabelY = findNonOverlappingY(desiredLabelY);

        const labelEl = document.createElement('div');
        labelEl.className = 'pattern-box-overlay';
        labelEl.style.cssText = `
          position: absolute;
          right: 70px;
          top: ${adjustedLabelY}px;
          color: ${patternInfo.color};
          font-size: 10px;
          font-weight: 700;
          text-shadow: 0 1px 2px rgba(0,0,0,0.9);
          white-space: nowrap;
          padding: 2px 6px;
          background: rgba(0,0,0,0.75);
          border: 1px solid ${patternInfo.color}80;
          border-radius: 3px;
          pointer-events: none;
          z-index: 6;
        `;
        labelEl.textContent = label;

        chartElement.style.position = 'relative';
        chartElement.appendChild(overlay);
        chartElement.appendChild(labelEl);
        patternOverlaysRef.current.push(overlay);
        patternOverlaysRef.current.push(labelEl);
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
          barSpacing: 3,  // Smaller bar spacing to fit more candles
          minBarSpacing: 1,  // Allow very small bars when zoomed out
          rightOffset: 5,  // Small offset on right side
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

      // Resize handler - update both width and height
      const handleResize = () => {
        if (chartContainerRef.current && chart) {
          chart.applyOptions({
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight
          });
        }
      };
      window.addEventListener('resize', handleResize);

      // Scroll/pan handler: re-filter patterns for visible range and redraw
      // Also detects when user scrolls to left edge to load more historical data
      chart.timeScale().subscribeVisibleLogicalRangeChange(async (logicalRange) => {
        if (window.patternUpdateTimeout) clearTimeout(window.patternUpdateTimeout);
        window.patternUpdateTimeout = setTimeout(async () => {
          const candles = candlesDataRef.current;
          const allPatterns = allApiPatternsRef.current;
          if (allPatterns.length > 0 && candles.length > 0 && logicalRange) {
            // Get only the visible candles based on the logical range
            const from = Math.max(0, Math.floor(logicalRange.from));
            const to = Math.min(candles.length - 1, Math.ceil(logicalRange.to));
            const visibleCandles = candles.slice(from, to + 1);
            if (visibleCandles.length > 0) {
              const relevant = getRelevantPatterns(allPatterns, visibleCandles, signalDirectionRef.current);
              analysisPatternsRef.current = relevant;
              clearPatternOverlays();
              drawPatternBoxes(relevant, candles);
            }
          }

          // Dynamic historical data loading: detect when user scrolls to left edge
          // Load more historical data when user reaches the left edge of loaded candles
          if (logicalRange && candles.length > 0 && !isLoadingHistoryRef.current) {
            const leftEdgeThreshold = 20; // Load more when within 20 candles of left edge
            const isAtLeftEdge = logicalRange.from <= leftEdgeThreshold;
            setAtLeftEdge(isAtLeftEdge);

            if (isAtLeftEdge) {
              console.log(`[LiveChart] User scrolled to left edge (from: ${logicalRange.from}), loading more history...`);
              isLoadingHistoryRef.current = true;
              setLoadingHistory(true);

              try {
                // Get the oldest candle's timestamp to fetch data before it
                const oldestCandle = candles[0];
                const endTime = oldestCandle.time;
                console.log(`[LiveChart] Fetching history before timestamp: ${endTime} (${new Date(endTime * 1000).toISOString()})`);

                const historyData = await getLiveOHLCV(symbol, timeframe, 500, endTime);

                if (historyData?.candles?.length > 0) {
                  // Filter out any candles we already have (by timestamp)
                  const existingTimes = new Set(candles.map(c => c.time));
                  const newCandles = historyData.candles.filter(c => !existingTimes.has(c.time));

                  if (newCandles.length > 0) {
                    console.log(`[LiveChart] Adding ${newCandles.length} historical candles`);
                    // Prepend new historical candles
                    const mergedCandles = [...newCandles, ...candles].sort((a, b) => a.time - b.time);
                    candlesDataRef.current = mergedCandles;

                    // Update chart with merged data
                    if (candlestickSeriesRef.current) {
                      candlestickSeriesRef.current.setData(mergedCandles);

                      // Maintain the user's current view position
                      // Calculate where the user was looking and keep them there
                      const shiftAmount = newCandles.length;
                      chartRef.current.timeScale().setVisibleLogicalRange({
                        from: logicalRange.from + shiftAmount,
                        to: logicalRange.to + shiftAmount,
                      });
                    }
                  } else {
                    console.log('[LiveChart] No new candles available (reached end of history)');
                  }
                }
              } catch (err) {
                console.error('[LiveChart] Error loading historical data:', err);
              } finally {
                setLoadingHistory(false);
                // Add a cooldown to prevent too many requests
                setTimeout(() => {
                  isLoadingHistoryRef.current = false;
                }, 1000);
              }
            }
          }
        }, 150);
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
        // Request maximum candles for all timeframes to show full history
        const candleLimit = 2000;  // Maximum candles for all timeframes
        console.log(`[LiveChart] Requesting ${candleLimit} candles for ${symbol} ${timeframe}`);
        const ohlcvData = await getLiveOHLCV(symbol, timeframe, candleLimit);
        console.log(`[LiveChart] Received ${ohlcvData?.candles?.length || 0} candles`);

        if (ohlcvData?.candles?.length > 0) {
          const candles = ohlcvData.candles;
          candlesDataRef.current = candles;
          // Track oldest timestamp for historical scrollback
          oldestTimestampRef.current = candles[0]?.time;
          // Reset history loading flag for new symbol/timeframe
          isLoadingHistoryRef.current = false;

          if (candlestickSeriesRef.current) {
            candlestickSeriesRef.current.setData(candles);
            lastCandleRef.current = candles[candles.length - 1];
            setCurrentPrice(lastCandleRef.current.close);

            if (candles.length > 1) {
              const prevClose = candles[candles.length - 2].close;
              setPriceChange(((lastCandleRef.current.close - prevClose) / prevClose) * 100);
            }

            // For initial load, show recent candles (not all 2000 at once)
            // User can scroll left to load more history dynamically
            if (chartRef.current && candles.length > 0) {
              const numCandles = candles.length;
              console.log(`[LiveChart] Loaded ${numCandles} candles, showing last 200`);

              // Show the most recent ~200 candles initially
              // User can scroll left to see older data and trigger dynamic loading
              const visibleCount = Math.min(200, numCandles);
              chartRef.current.timeScale().setVisibleLogicalRange({
                from: numCandles - visibleCount,
                to: numCandles - 1,
              });
            }
          }
        } else {
          setError('No data available');
        }

        // Load analysis
        try {
          const analysisData = await getSignalAnalysis(symbol, timeframe, selectedPlaylist);
          setAnalysis(analysisData);

          // Only draw patterns and price lines if ML is trained
          const isMlTrained = analysisData?.ml_status === 'trained';

          if (isMlTrained && analysisData?.patterns && analysisData.patterns.length > 0 && candlesDataRef.current.length > 0) {
            const signalDirection = analysisData?.signal?.direction?.toLowerCase();

            // Store all API patterns and signal direction for scroll-based re-filtering
            allApiPatternsRef.current = analysisData.patterns;
            signalDirectionRef.current = signalDirection;

            // Filter for current visible range
            const relevantPatterns = getRelevantPatterns(analysisData.patterns, candlesDataRef.current, signalDirection);
            analysisPatternsRef.current = relevantPatterns;
            setTimeout(() => drawPatternBoxes(relevantPatterns, candlesDataRef.current), 200);
          } else {
            // Clear patterns if ML not trained or no patterns
            allApiPatternsRef.current = [];
            signalDirectionRef.current = null;
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

        // Load ML prediction in background
        try {
          const pred = await predictPrice(symbol);
          setMlPrediction(pred);
        } catch (err) {
          setMlPrediction(null);
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
  }, [symbol, timeframe, selectedPlaylist, drawPatternBoxes, clearPatternOverlays]);

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

  // Handle chart resize when sidebar is toggled or fullscreen mode changes
  useEffect(() => {
    // Small delay to allow DOM to update before measuring new dimensions
    const resizeTimer = setTimeout(() => {
      if (chartRef.current && chartContainerRef.current) {
        const newWidth = chartContainerRef.current.clientWidth;
        const newHeight = chartContainerRef.current.clientHeight;

        // Update both width AND height - critical for fullscreen mode
        chartRef.current.applyOptions({
          width: newWidth,
          height: newHeight
        });

        // Fit content to show all data after resize
        chartRef.current.timeScale().fitContent();

        // Also redraw pattern overlays as they're positioned absolutely
        if (analysisPatternsRef.current.length > 0 && candlesDataRef.current.length > 0) {
          clearPatternOverlays();
          drawPatternBoxes(analysisPatternsRef.current, candlesDataRef.current);
        }
      }
    }, 100);

    return () => clearTimeout(resizeTimer);
  }, [showLegend, fullscreen, clearPatternOverlays, drawPatternBoxes]);

  const signal = analysis?.signal;
  const isBullish = signal?.direction === 'bullish';
  const isBearish = signal?.direction === 'bearish';

  return (
    <div className={`${fullscreen ? 'fixed inset-0 z-50 bg-[#0a0a0f] p-2 overflow-y-auto' : 'space-y-4'}`}>
      {/* Header - Compact in fullscreen mode */}
      <div className={`flex flex-col lg:flex-row lg:items-center justify-between flex-shrink-0 ${fullscreen ? 'gap-1' : 'gap-4'}`}>
        <div className={`flex items-center ${fullscreen ? 'gap-2' : 'gap-4'}`}>
          {!fullscreen && (
            <div className="p-2 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-500/20">
              <BarChart2 className="w-6 h-6 text-indigo-400" />
            </div>
          )}
          <div>
            <div className="flex items-center gap-3">
              <h1 className={`font-bold text-white ${fullscreen ? 'text-lg' : 'text-2xl'}`}>{symbol}</h1>
              {currentPrice && (
                <span className={`font-bold text-white ${fullscreen ? 'text-lg' : 'text-2xl'}`}>{formatPrice(currentPrice)}</span>
              )}
              {priceChange !== 0 && (
                <span className={`text-sm font-semibold px-2 py-1 rounded ${
                  priceChange >= 0 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                }`}>
                  {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                </span>
              )}
            </div>
            {!fullscreen && (
              <div className="flex items-center gap-2 text-sm text-slate-400">
                {isConnected ? (
                  <span className="flex items-center gap-1 text-emerald-400"><Wifi className="w-3 h-3" /> Live</span>
                ) : (
                  <span className="flex items-center gap-1 text-yellow-400"><WifiOff className="w-3 h-3" /> Connecting...</span>
                )}
                {lastUpdate && <span>Updated: {lastUpdate.toLocaleTimeString()}</span>}
              </div>
            )}
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

          {/* Playlist Knowledge Selector */}
          {playlists.length > 0 && (
            <div className="flex items-center">
              <select
                value={selectedPlaylist}
                onChange={(e) => setSelectedPlaylist(e.target.value)}
                className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500/50 cursor-pointer appearance-none"
                title="Select which playlist's ML knowledge drives pattern detection"
                style={{ maxWidth: '220px' }}
              >
                <option value="all">All Playlists (Combined)</option>
                {playlists.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.title} ({p.trained_video_count}/{p.video_count} trained)
                  </option>
                ))}
              </select>
            </div>
          )}

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
      <div className={`grid grid-cols-1 xl:grid-cols-4 ${fullscreen ? 'gap-2' : 'gap-4'}`}>
        {/* Chart Area */}
        <div className={`${showLegend ? 'xl:col-span-3' : 'xl:col-span-4'}`}>
          <div className="card-dark rounded-xl overflow-hidden relative">
            <div
              ref={chartContainerRef}
              className="w-full"
              style={{ height: fullscreen ? 'calc(100vh - 70px)' : '500px', backgroundColor: '#131722' }}
            />

            {loading && (
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-10">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-10 h-10 border-3 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
                  <span className="text-slate-400 text-sm">Loading {symbol} data...</span>
                </div>
              </div>
            )}

            {/* Historical data loading indicator - shows when scrolling left to load more */}
            {loadingHistory && (
              <div className="absolute top-2 left-2 bg-indigo-600/90 text-white px-3 py-1 rounded-lg text-xs flex items-center gap-2 z-20 animate-pulse">
                <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Loading historical data...
              </div>
            )}

            {/* Chart Control Buttons - TradingView style */}
            <div className="absolute bottom-4 left-4 flex flex-col gap-1 z-20">
              {/* Zoom Controls */}
              <div className="flex flex-col bg-slate-800/90 rounded-lg shadow-lg border border-slate-700/50 overflow-hidden">
                <button
                  onClick={handleZoomIn}
                  className="p-2 hover:bg-slate-700/80 text-slate-300 hover:text-white transition-colors border-b border-slate-700/50"
                  title="Zoom In"
                >
                  <Plus className="w-4 h-4" />
                </button>
                <button
                  onClick={handleZoomOut}
                  className="p-2 hover:bg-slate-700/80 text-slate-300 hover:text-white transition-colors"
                  title="Zoom Out"
                >
                  <Minus className="w-4 h-4" />
                </button>
              </div>

              {/* Reset/Navigation Controls */}
              <div className="flex flex-col bg-slate-800/90 rounded-lg shadow-lg border border-slate-700/50 overflow-hidden mt-1">
                <button
                  onClick={handleResetChart}
                  className="p-2 hover:bg-slate-700/80 text-slate-300 hover:text-white transition-colors border-b border-slate-700/50"
                  title="Reset View (Show Recent)"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
                <button
                  onClick={handleFitAll}
                  className="p-2 hover:bg-slate-700/80 text-slate-300 hover:text-white transition-colors border-b border-slate-700/50"
                  title="Fit All Data"
                >
                  <Maximize2 className="w-4 h-4" />
                </button>
                <button
                  onClick={handleScrollToLatest}
                  className="p-2 hover:bg-slate-700/80 text-slate-300 hover:text-white transition-colors"
                  title="Go to Latest"
                >
                  <Home className="w-4 h-4" />
                </button>
              </div>
            </div>

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
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 ml-auto">Expert-trained</span>
                    </div>
                    <div className="text-slate-300 leading-relaxed text-xs max-h-64 overflow-y-auto pr-1 custom-scrollbar">
                      {analysis.ml_reasoning.split('\n').map((line, idx) => {
                        // Format bold text (**text**)
                        const parts = line.split(/(\*\*[^*]+\*\*)/g);
                        const formatted = parts.map((part, pidx) => {
                          if (part.startsWith('**') && part.endsWith('**')) {
                            return <span key={pidx} className="font-semibold text-white">{part.slice(2, -2)}</span>;
                          }
                          return part;
                        });
                        // Indent lines starting with arrow or bullet
                        const isIndented = line.trim().startsWith('↳') || line.trim().startsWith('-') || line.trim().startsWith('•');
                        return (
                          <div key={idx} className={isIndented ? 'pl-4 text-slate-400' : ''}>
                            {formatted}
                          </div>
                        );
                      })}
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

                {/* Quant Engine Insights */}
                {analysis.quant && (
                  <div className="p-3 rounded-lg bg-cyan-500/5 border border-cyan-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <BarChart2 className="w-4 h-4 text-cyan-400" />
                      <span className="font-semibold text-cyan-400">Quant Engine</span>
                    </div>
                    <div className="space-y-2">
                      {/* Market Regime */}
                      {analysis.quant.regime && (
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-slate-400">Market Regime</span>
                          <span className={`px-2 py-0.5 rounded-full font-medium ${
                            analysis.quant.regime.regime === 'trending_up' ? 'bg-emerald-500/20 text-emerald-400' :
                            analysis.quant.regime.regime === 'trending_down' ? 'bg-red-500/20 text-red-400' :
                            analysis.quant.regime.regime === 'high_volatility' ? 'bg-amber-500/20 text-amber-400' :
                            'bg-blue-500/20 text-blue-400'
                          }`}>
                            {(analysis.quant.regime.regime || 'unknown').replace('_', ' ').toUpperCase()}
                          </span>
                        </div>
                      )}
                      {analysis.quant.regime?.confidence && (
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-slate-400">Regime Confidence</span>
                          <span className="font-mono text-white">{Math.round(analysis.quant.regime.confidence * 100)}%</span>
                        </div>
                      )}
                      {/* ML Classifier */}
                      {analysis.quant.ml_classifier && (
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-slate-400">ML Classifiers</span>
                          <span className="text-cyan-400 font-medium">
                            {analysis.quant.ml_classifier.models_trained} trained
                          </span>
                        </div>
                      )}
                      {/* Backtest Stats */}
                      {analysis.quant.backtest && Object.keys(analysis.quant.backtest).length > 0 && (
                        <div className="mt-2 pt-2 border-t border-cyan-500/10">
                          <div className="text-[10px] text-cyan-400 font-semibold mb-1">Backtest Win Rates</div>
                          <div className="space-y-1">
                            {Object.entries(analysis.quant.backtest).slice(0, 4).map(([pattern, stats]) => {
                              const wr = stats.avg_win_rate || stats.win_rate || 0;
                              const wrPct = wr > 1 ? wr : wr * 100;
                              return (
                              <div key={pattern} className="flex items-center justify-between text-xs">
                                <span className="text-slate-400 capitalize">{pattern.replace('_', ' ')}</span>
                                <div className="flex items-center gap-2">
                                  <div className="w-12 h-1 bg-slate-700 rounded-full overflow-hidden">
                                    <div
                                      className={`h-full rounded-full ${
                                        wrPct >= 60 ? 'bg-emerald-400' :
                                        wrPct >= 50 ? 'bg-amber-400' : 'bg-red-400'
                                      }`}
                                      style={{ width: `${Math.round(wrPct)}%` }}
                                    />
                                  </div>
                                  <span className="font-mono text-white w-8 text-right">
                                    {Math.round(wrPct)}%
                                  </span>
                                </div>
                              </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
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
                                {confidence >= 0.85 && (
                                  <span className="text-[8px] px-1 py-0.5 rounded bg-purple-500/20 text-purple-300">Expert</span>
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

            {/* Quant Engine Status Card */}
            {analysis?.quant && (
              <div className="card-dark rounded-xl p-4">
                <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                  <BarChart2 className="w-4 h-4 text-cyan-400" /> Quant Engine
                </h3>
                <div className="space-y-2">
                  {analysis.quant.regime && (
                    <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Regime</span>
                        <span className={`px-2 py-0.5 rounded-full font-semibold ${
                          analysis.quant.regime.regime === 'trending_up' ? 'bg-emerald-500/20 text-emerald-400' :
                          analysis.quant.regime.regime === 'trending_down' ? 'bg-red-500/20 text-red-400' :
                          analysis.quant.regime.regime === 'high_volatility' ? 'bg-amber-500/20 text-amber-400' :
                          'bg-blue-500/20 text-blue-400'
                        }`}>
                          {(analysis.quant.regime.regime || '').replace('_', ' ').toUpperCase()}
                        </span>
                      </div>
                    </div>
                  )}
                  {analysis.quant.ml_classifier && (
                    <div className="flex items-center justify-between text-xs p-2 rounded-lg bg-slate-800/50">
                      <span className="text-slate-400">ML Models</span>
                      <span className="text-cyan-400">{analysis.quant.ml_classifier.models_trained} trained</span>
                    </div>
                  )}
                  {analysis.quant.backtest && Object.keys(analysis.quant.backtest).length > 0 && (
                    <div className="flex items-center justify-between text-xs p-2 rounded-lg bg-slate-800/50">
                      <span className="text-slate-400">Backtested</span>
                      <span className="text-emerald-400">{Object.keys(analysis.quant.backtest).length} patterns</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Video Knowledge → ML Features */}
            {analysis?.video_knowledge?.active && (
              <div className="card-dark rounded-xl p-4">
                <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-purple-400" /> Video Knowledge
                  <span className="text-[9px] bg-purple-500/20 text-purple-400 px-1.5 py-0.5 rounded-full">
                    {analysis.video_knowledge.concepts_loaded} concepts
                  </span>
                </h3>
                <div className="space-y-2">
                  {/* Teaching depth for detected patterns */}
                  {analysis.video_knowledge.pattern_teaching && Object.entries(analysis.video_knowledge.pattern_teaching).length > 0 && (
                    <div>
                      <div className="text-[10px] text-slate-400 mb-1.5">Teaching Depth (detected patterns)</div>
                      {Object.entries(analysis.video_knowledge.pattern_teaching)
                        .sort(([,a], [,b]) => b.depth_score - a.depth_score)
                        .map(([pattern, info]) => (
                        <div key={pattern} className="flex items-center gap-1.5 mb-1">
                          <span className="text-[10px] text-slate-300 w-20 truncate">{pattern.replace(/_/g, ' ')}</span>
                          <div className="flex-1 bg-slate-700 rounded-full h-1.5">
                            <div
                              className="bg-gradient-to-r from-purple-500 to-cyan-400 h-1.5 rounded-full"
                              style={{ width: `${Math.round(info.depth_score * 100)}%` }}
                            />
                          </div>
                          <span className="text-[9px] text-slate-500 w-8 text-right">{info.teaching_depth}u</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {/* Co-occurrences */}
                  {analysis.video_knowledge.co_occurrences?.length > 0 && (
                    <div className="pt-2 border-t border-slate-700/50">
                      <div className="text-[10px] text-slate-400 mb-1">Learned Co-occurrences</div>
                      {analysis.video_knowledge.co_occurrences.slice(0, 3).map((pair, i) => (
                        <div key={i} className="flex items-center gap-1 text-[10px] mb-0.5">
                          <span className="text-cyan-400">{pair.a.replace(/_/g, ' ')}</span>
                          <span className="text-slate-600">+</span>
                          <span className="text-purple-400">{pair.b.replace(/_/g, ' ')}</span>
                          <span className="ml-auto text-yellow-400 font-mono">{Math.round(pair.score * 100)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="text-[9px] text-slate-600 mt-1">
                    {analysis.video_knowledge.features_active} video features active
                  </div>
                </div>
              </div>
            )}

            {/* ML Price Prediction Card */}
            {mlPrediction?.predictions && (
              <div className="card-dark rounded-xl p-4">
                <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-violet-400" /> ML Prediction
                </h3>
                <div className="space-y-2">
                  {['h5', 'h10', 'h20'].map(h => {
                    const pred = mlPrediction.predictions[h];
                    if (!pred || pred.error) return null;
                    return (
                      <div key={h} className={`p-2 rounded-lg border text-xs ${
                        pred.direction === 'bullish' ? 'bg-emerald-500/10 border-emerald-500/20' :
                        pred.direction === 'bearish' ? 'bg-red-500/10 border-red-500/20' :
                        'bg-slate-500/10 border-slate-500/20'
                      }`}>
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400">{h.replace('h', '')}d</span>
                          <div className="flex items-center gap-1">
                            <span className={`font-bold uppercase ${
                              pred.direction === 'bullish' ? 'text-emerald-400' :
                              pred.direction === 'bearish' ? 'text-red-400' : 'text-slate-400'
                            }`}>
                              {pred.direction}
                            </span>
                            <span className="text-slate-500 font-mono">
                              {(pred.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                  {Object.values(mlPrediction.predictions).every(p => p?.error) && (
                    <p className="text-xs text-slate-500 text-center py-2">Not trained yet</p>
                  )}
                </div>
              </div>
            )}
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
