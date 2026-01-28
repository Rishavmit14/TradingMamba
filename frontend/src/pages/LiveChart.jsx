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
const CONCEPT_INFO = {
  'FVG': { name: 'Fair Value Gap', color: '#ffd700', description: 'Price imbalance zone' },
  'OB': { name: 'Order Block', color: '#26a69a', description: 'Institutional entry zone' },
  'BOS': { name: 'Break of Structure', color: '#4fc3f7', description: 'Trend continuation' },
  'CHoCH': { name: 'Change of Character', color: '#ff9800', description: 'Trend reversal signal' },
  'EQH': { name: 'Equal Highs', color: '#ef5350', description: 'Buy-side liquidity' },
  'EQL': { name: 'Equal Lows', color: '#66bb6a', description: 'Sell-side liquidity' },
  'OTE': { name: 'Optimal Trade Entry', color: '#9c27b0', description: 'Fib retracement zone' },
  'LIQ': { name: 'Liquidity Sweep', color: '#e91e63', description: 'Stop hunt pattern' },
};

// Map backend pattern types to display names
const PATTERN_TYPE_MAP = {
  'bullish_order_block': { short: 'OB', color: '#26a69a', direction: 'bullish' },
  'bearish_order_block': { short: 'OB', color: '#ef5350', direction: 'bearish' },
  'bullish_fvg': { short: 'FVG', color: '#ffd700', direction: 'bullish' },
  'bearish_fvg': { short: 'FVG', color: '#ff9800', direction: 'bearish' },
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
    const markers = [];

    patterns.forEach((pattern, idx) => {
      const patternType = pattern.pattern_type;
      const patternInfo = PATTERN_TYPE_MAP[patternType] || { short: patternType, color: '#9ca3af', direction: 'neutral' };
      const patternTimeframe = pattern.timeframe || timeframe;
      const label = `${patternInfo.short} ${getTimeframeLabel(patternTimeframe)}`;

      const highPrice = pattern.high || pattern.price_high || pattern.price;
      const lowPrice = pattern.low || pattern.price_low || pattern.price;

      if (!highPrice && !lowPrice && !pattern.price) return;

      let patternTime;
      if (pattern.start_index !== undefined && candles.length > 0) {
        const startIdx = Math.max(0, Math.min(pattern.start_index, candles.length - 1));
        patternTime = candles[startIdx]?.time;
      } else if (candles.length > 0) {
        const defaultIdx = Math.max(0, candles.length - 20 - idx * 3);
        patternTime = candles[defaultIdx]?.time;
      }

      if (!patternTime) return;

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
      // Always extend patterns to the right edge of the chart (current time)
      // This ensures higher timeframe patterns are visible in lower timeframes
      let endTime = candles.length > 0 ? candles[candles.length - 1]?.time : null;

      if (!startTime || !endTime || !highPrice || !lowPrice) return;

      try {
        const timeScale = chart.timeScale();
        const x1 = timeScale.timeToCoordinate(startTime);
        const x2 = timeScale.timeToCoordinate(endTime);
        const y1 = series.priceToCoordinate(highPrice);
        const y2 = series.priceToCoordinate(lowPrice);

        if (x1 === null || x2 === null || y1 === null || y2 === null) return;

        const width = Math.max(Math.abs(x2 - x1), 30);
        const height = Math.max(Math.abs(y2 - y1), 20);

        const overlay = document.createElement('div');
        overlay.className = 'pattern-box-overlay';
        overlay.style.cssText = `
          position: absolute;
          left: ${Math.min(x1, x2)}px;
          top: ${Math.min(y1, y2)}px;
          width: ${width}px;
          height: ${height}px;
          background-color: ${patternInfo.color}25;
          border: 2px solid ${patternInfo.color}80;
          border-radius: 4px;
          pointer-events: none;
          z-index: 5;
          display: flex;
          align-items: center;
          justify-content: center;
        `;

        const labelEl = document.createElement('span');
        labelEl.style.cssText = `
          color: ${patternInfo.color};
          font-size: 11px;
          font-weight: 700;
          text-shadow: 0 1px 2px rgba(0,0,0,0.9), 0 0 4px rgba(0,0,0,0.7);
          white-space: nowrap;
          padding: 3px 6px;
          background: rgba(0,0,0,0.6);
          border-radius: 4px;
        `;
        labelEl.textContent = label;
        overlay.appendChild(labelEl);

        chartElement.style.position = 'relative';
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
  }, [timeframe, clearPatternOverlays]);

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

      // Scroll handler for pattern redraw
      chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
        if (window.patternUpdateTimeout) clearTimeout(window.patternUpdateTimeout);
        window.patternUpdateTimeout = setTimeout(() => {
          if (analysisPatternsRef.current.length > 0 && candlesDataRef.current.length > 0) {
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
        const ohlcvData = await getLiveOHLCV(symbol, timeframe, 200);

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

          if (analysisData?.patterns && candlesDataRef.current.length > 0 && analysisData?.signal?.direction) {
            const signalDirection = analysisData.signal.direction;
            const relevantPatterns = analysisData.patterns.filter(p => {
              const pt = p.pattern_type;
              if (signalDirection === 'bullish') {
                return pt.includes('bullish') || pt === 'liquidity_sweep_low' || pt === 'equal_lows';
              } else if (signalDirection === 'bearish') {
                return pt.includes('bearish') || pt === 'liquidity_sweep_high' || pt === 'equal_highs';
              }
              return false;
            }).slice(-10);

            analysisPatternsRef.current = relevantPatterns;
            setTimeout(() => drawPatternBoxes(relevantPatterns, candlesDataRef.current), 200);
          }

          // Add price lines
          if (analysisData?.signal && candlestickSeriesRef.current) {
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

          {/* Smart Money Analysis Summary */}
          {analysis?.signal && !loading && (
            <div className="mt-4 p-4 bg-slate-900/50 rounded-xl border border-slate-700/50">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-400" />
                Smart Money Analysis Summary
              </h3>

              <div className="space-y-4 text-sm">
                {/* Entry Zone Explanation */}
                {analysis.signal.entry_zone && (
                  <div className="p-3 rounded-lg bg-blue-500/5 border border-blue-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Crosshair className="w-4 h-4 text-blue-400" />
                      <span className="font-semibold text-blue-400">Entry Zone: {formatPrice(analysis.signal.entry_zone[0])} - {formatPrice(analysis.signal.entry_zone[1])}</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed">
                      {analysis.signal.direction === 'bullish' ? (
                        <>
                          The ML identified this as a <span className="text-yellow-400 font-medium">bullish Order Block (OB)</span> zone where institutional buying previously occurred.
                          Price has swept below an <span className="text-green-400 font-medium">Equal Low (EQL)</span> liquidity pool, grabbing stop losses from retail traders.
                          Combined with a <span className="text-yellow-400 font-medium">Fair Value Gap (FVG)</span> in this zone, the ML expects a reversal to the upside as smart money accumulates positions.
                        </>
                      ) : analysis.signal.direction === 'bearish' ? (
                        <>
                          The ML identified this as a <span className="text-yellow-400 font-medium">bearish Order Block (OB)</span> zone where institutional selling previously occurred.
                          Price has swept above an <span className="text-red-400 font-medium">Equal High (EQH)</span> liquidity pool, triggering stop losses from retail shorts.
                          Combined with a <span className="text-yellow-400 font-medium">Fair Value Gap (FVG)</span> imbalance, the ML expects distribution and a move to the downside.
                        </>
                      ) : (
                        <>The ML is analyzing price action for potential Smart Money accumulation or distribution zones.</>
                      )}
                    </p>
                  </div>
                )}

                {/* Stop Loss Explanation */}
                {analysis.signal.stop_loss && (
                  <div className="p-3 rounded-lg bg-red-500/5 border border-red-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Shield className="w-4 h-4 text-red-400" />
                      <span className="font-semibold text-red-400">Stop Loss: {formatPrice(analysis.signal.stop_loss)}</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed">
                      {analysis.signal.direction === 'bullish' ? (
                        <>
                          Stop loss is placed <span className="text-red-400 font-medium">below the Order Block</span> and recent swing low structure.
                          This level is positioned beyond the <span className="text-green-400 font-medium">liquidity pool</span> that smart money targeted.
                          A <span className="text-cyan-400 font-medium">Break of Structure (BOS)</span> below this level would invalidate the bullish setup.
                        </>
                      ) : analysis.signal.direction === 'bearish' ? (
                        <>
                          Stop loss is placed <span className="text-red-400 font-medium">above the Order Block</span> and recent swing high structure.
                          This protects against a false <span className="text-cyan-400 font-medium">Change of Character (CHoCH)</span>.
                          A break above this level would invalidate the bearish thesis.
                        </>
                      ) : (
                        <>Stop loss is strategically placed beyond key structure levels.</>
                      )}
                    </p>
                  </div>
                )}

                {/* Take Profit Explanation */}
                {analysis.signal.take_profit && analysis.signal.take_profit.length > 0 && (
                  <div className="p-3 rounded-lg bg-emerald-500/5 border border-emerald-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="w-4 h-4 text-emerald-400" />
                      <span className="font-semibold text-emerald-400">Take Profit Targets</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed mb-2">
                      {analysis.signal.direction === 'bullish' ? (
                        <>
                          Take profit levels are set at <span className="text-red-400 font-medium">buy-side liquidity pools (Equal Highs)</span> where retail traders have their stop losses clustered.
                          Smart money will target these levels to distribute positions.
                        </>
                      ) : analysis.signal.direction === 'bearish' ? (
                        <>
                          Take profit levels are set at <span className="text-green-400 font-medium">sell-side liquidity pools (Equal Lows)</span> where retail traders have their stop losses.
                          Smart money will drive price to these levels to cover short positions.
                        </>
                      ) : (
                        <>Take profit levels are identified at key liquidity zones.</>
                      )}
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mt-2">
                      {analysis.signal.take_profit.map((tp, idx) => (
                        <div key={idx} className="p-2 rounded bg-emerald-500/10 border border-emerald-500/30 text-center">
                          <div className="text-xs text-slate-400">TP{idx + 1}</div>
                          <div className="font-mono text-emerald-400 font-semibold">{formatPrice(tp)}</div>
                          <div className="text-xs text-slate-500">
                            {idx === 0 ? 'First Liquidity' : idx === 1 ? 'Key Structure' : 'Extended Target'}
                          </div>
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

            {/* Concepts Legend */}
            <div className="card-dark rounded-xl p-4">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Layers className="w-4 h-4 text-purple-400" /> Smart Money Concepts
              </h3>
              <div className="space-y-2">
                {Object.entries(CONCEPT_INFO).map(([key, info]) => (
                  <div key={key} className="flex items-center gap-2 text-xs">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: info.color + '40', border: `1px solid ${info.color}` }} />
                    <span className="text-white font-medium">{key}</span>
                    <span className="text-slate-500">- {info.description}</span>
                  </div>
                ))}
              </div>
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
