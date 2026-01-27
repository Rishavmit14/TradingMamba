import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Search,
  RefreshCw,
  ChevronRight,
  Target,
  Shield,
  Award
} from 'lucide-react';
import { analyzeSymbol, getSymbols } from '../services/api';

// Full Signal Analysis Card
function SignalAnalysis({ symbol, analysis, loading }) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 animate-pulse">
        <div className="h-8 bg-gray-700 rounded w-1/3 mb-6"></div>
        <div className="space-y-4">
          <div className="h-4 bg-gray-700 rounded w-full"></div>
          <div className="h-4 bg-gray-700 rounded w-2/3"></div>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return null;
  }

  const signal = analysis.signal;
  const hasSignal = signal !== null;
  const isBullish = signal?.direction === 'bullish';

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
      {/* Header */}
      <div className={`p-6 ${
        hasSignal ? (isBullish ? 'bg-green-500/10' : 'bg-red-500/10') : 'bg-gray-700/30'
      }`}>
        <div className="flex justify-between items-start">
          <div>
            <h2 className="text-2xl font-bold">{analysis.symbol}</h2>
            <p className="text-gray-400 text-sm mt-1">
              Analyzed: {analysis.timeframes?.join(', ')}
            </p>
          </div>
          {hasSignal && (
            <div className={`px-4 py-2 rounded-lg ${
              isBullish ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            } flex items-center space-x-2`}>
              {isBullish ? <TrendingUp size={20} /> : <TrendingDown size={20} />}
              <span className="font-bold text-lg">{signal.direction.toUpperCase()}</span>
            </div>
          )}
        </div>
      </div>

      {/* Signal Details */}
      {hasSignal ? (
        <div className="p-6 space-y-6">
          {/* Confidence & Strength */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-gray-700/50 rounded-lg p-4 text-center">
              <Award className="mx-auto mb-2 text-yellow-400" size={24} />
              <p className="text-sm text-gray-400">Confidence</p>
              <p className="text-xl font-bold">{(signal.confidence * 100).toFixed(0)}%</p>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-4 text-center">
              <Target className="mx-auto mb-2 text-blue-400" size={24} />
              <p className="text-sm text-gray-400">Strength</p>
              <p className="text-xl font-bold">{(signal.strength * 100).toFixed(0)}%</p>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-4 text-center">
              <Shield className="mx-auto mb-2 text-purple-400" size={24} />
              <p className="text-sm text-gray-400">Confluence</p>
              <p className="text-xl font-bold">{signal.confluence_score}</p>
            </div>
          </div>

          {/* Entry/SL/TP */}
          <div className="bg-gray-700/30 rounded-lg p-4">
            <h3 className="font-medium mb-3">Trade Levels</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-gray-400">Entry Zone</p>
                <p className="font-mono">
                  {signal.entry_zone[0]?.toFixed(5)} - {signal.entry_zone[1]?.toFixed(5)}
                </p>
              </div>
              <div>
                <p className="text-gray-400">Stop Loss</p>
                <p className="font-mono text-red-400">{signal.stop_loss?.toFixed(5)}</p>
              </div>
              <div>
                <p className="text-gray-400">Take Profit 1</p>
                <p className="font-mono text-green-400">{signal.take_profit?.[0]?.toFixed(5)}</p>
              </div>
              <div>
                <p className="text-gray-400">R:R</p>
                <p className="font-bold text-blue-400">{signal.risk_reward?.toFixed(2)}</p>
              </div>
            </div>
          </div>

          {/* Concepts */}
          <div>
            <h3 className="font-medium mb-3">ICT Concepts Detected</h3>
            <div className="flex flex-wrap gap-2">
              {signal.concepts?.map((concept, i) => (
                <span
                  key={i}
                  className="px-3 py-1 bg-blue-500/20 text-blue-300 rounded-lg text-sm"
                >
                  {concept.replace('_', ' ')}
                </span>
              ))}
            </div>
          </div>

          {/* Reasoning */}
          <div>
            <h3 className="font-medium mb-3">Analysis Reasoning</h3>
            <ul className="space-y-1 text-sm text-gray-400">
              {signal.reasoning?.map((reason, i) => (
                <li key={i} className="flex items-start">
                  <ChevronRight size={14} className="mr-1 mt-1 flex-shrink-0" />
                  {reason}
                </li>
              ))}
            </ul>
          </div>
        </div>
      ) : (
        <div className="p-6 text-center">
          <Activity className="mx-auto mb-4 text-gray-500" size={48} />
          <h3 className="text-lg font-medium text-gray-400">No Signal</h3>
          <p className="text-gray-500 text-sm mt-2">
            Insufficient confluence for a trading signal.
            Multiple ICT concepts must align for signal generation.
          </p>
        </div>
      )}

      {/* Timeframe Analysis */}
      <div className="border-t border-gray-700 p-6">
        <h3 className="font-medium mb-4">Timeframe Breakdown</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(analysis.analyses || {}).map(([tf, data]) => (
            <div key={tf} className="bg-gray-700/30 rounded-lg p-4">
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium">{tf}</span>
                <span className={`text-sm ${
                  data.bias === 'bullish' ? 'text-green-400' :
                  data.bias === 'bearish' ? 'text-red-400' : 'text-gray-400'
                }`}>
                  {data.bias?.toUpperCase()}
                </span>
              </div>
              <div className="text-xs text-gray-400 space-y-1">
                <p>Zone: {data.zone}</p>
                <p>Order Blocks: {data.order_blocks}</p>
                <p>FVGs: {data.fvgs}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Main Signals Page
export default function Signals() {
  const [selectedSymbol, setSelectedSymbol] = useState('EURUSD');
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [symbols, setSymbols] = useState([]);

  const defaultSymbols = ['EURUSD', 'GBPUSD', 'XAUUSD', 'US30', 'NAS100', 'USDJPY'];

  useEffect(() => {
    loadSymbols();
    analyzeSelectedSymbol();
  }, []);

  const loadSymbols = async () => {
    try {
      const data = await getSymbols();
      setSymbols(data.symbols || defaultSymbols);
    } catch (err) {
      setSymbols(defaultSymbols);
    }
  };

  const analyzeSelectedSymbol = async () => {
    setLoading(true);
    try {
      const data = await analyzeSymbol(selectedSymbol);
      setAnalysis(data);
    } catch (err) {
      console.error('Analysis failed:', err);
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  };

  const handleSymbolChange = (symbol) => {
    setSelectedSymbol(symbol);
    setAnalysis(null);
  };

  useEffect(() => {
    if (selectedSymbol) {
      analyzeSelectedSymbol();
    }
  }, [selectedSymbol]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Signal Analysis</h1>
          <p className="text-gray-400">Full ICT methodology analysis with ML insights</p>
        </div>
      </div>

      {/* Symbol Selector */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <div className="flex flex-wrap gap-2">
          {(symbols.length > 0 ? symbols : defaultSymbols).map(symbol => (
            <button
              key={symbol}
              onClick={() => handleSymbolChange(symbol)}
              className={`px-4 py-2 rounded-lg transition-colors ${
                selectedSymbol === symbol
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {symbol}
            </button>
          ))}
        </div>
      </div>

      {/* Analysis Result */}
      <SignalAnalysis
        symbol={selectedSymbol}
        analysis={analysis}
        loading={loading}
      />

      {/* Disclaimer */}
      <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4 text-sm text-yellow-200/80">
        <strong>Disclaimer:</strong> This is an educational tool. Trading involves risk.
        Signals are generated by AI based on ICT methodology and should not be considered financial advice.
        Always do your own analysis and manage risk appropriately.
      </div>
    </div>
  );
}
