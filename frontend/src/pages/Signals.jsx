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
  Award,
  Zap,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  AlertTriangle,
  Sparkles,
  BarChart3,
  Layers,
  Trophy,
  Star,
  CheckCircle,
  XCircle
} from 'lucide-react';
import { analyzeSymbol, getSymbols } from '../services/api';

// Background Orb Component
function BackgroundOrb({ className }) {
  return (
    <div className={`absolute rounded-full blur-3xl opacity-20 animate-pulse ${className}`} />
  );
}

// Grade Badge Component (Hedge Fund Level)
function GradeBadge({ grade, size = 'md' }) {
  const gradeConfig = {
    'A+': { bg: 'bg-gradient-to-br from-amber-400 to-yellow-500', text: 'text-black', icon: Trophy },
    'A': { bg: 'bg-gradient-to-br from-emerald-400 to-green-500', text: 'text-white', icon: Star },
    'B': { bg: 'bg-gradient-to-br from-blue-400 to-indigo-500', text: 'text-white', icon: CheckCircle },
    'C': { bg: 'bg-gradient-to-br from-yellow-400 to-orange-500', text: 'text-black', icon: AlertTriangle },
    'D': { bg: 'bg-gradient-to-br from-orange-400 to-red-500', text: 'text-white', icon: XCircle },
    'F': { bg: 'bg-gradient-to-br from-red-500 to-red-700', text: 'text-white', icon: XCircle },
  };

  const config = gradeConfig[grade] || gradeConfig['C'];
  const sizeClasses = size === 'lg' ? 'w-14 h-14 text-xl' : size === 'sm' ? 'w-8 h-8 text-xs' : 'w-10 h-10 text-sm';

  return (
    <div className={`${sizeClasses} ${config.bg} ${config.text} rounded-xl flex items-center justify-center font-bold shadow-lg`}>
      {grade}
    </div>
  );
}

// Pattern Grade Card Component
function PatternGradeCard({ patternType, gradeInfo }) {
  if (!gradeInfo) return null;

  return (
    <div className="p-4 rounded-xl bg-white/5 border border-white/10">
      <div className="flex items-center justify-between mb-3">
        <span className="text-white font-medium capitalize">{patternType.replace('_', ' ')}</span>
        <GradeBadge grade={gradeInfo.grade} size="sm" />
      </div>
      <div className="text-xs text-slate-400 mb-2">
        Score: {(gradeInfo.score * 100).toFixed(0)}%
      </div>
      {gradeInfo.strengths?.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-2">
          {gradeInfo.strengths.slice(0, 2).map((s, i) => (
            <span key={i} className="px-2 py-0.5 text-xs rounded-full bg-emerald-500/10 text-emerald-400">
              {s}
            </span>
          ))}
        </div>
      )}
      {gradeInfo.weaknesses?.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {gradeInfo.weaknesses.slice(0, 2).map((w, i) => (
            <span key={i} className="px-2 py-0.5 text-xs rounded-full bg-red-500/10 text-red-400">
              {w}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// Animated Confidence Ring
function ConfidenceRing({ value, size = 80, strokeWidth = 6, color }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          className="text-white/10"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-lg font-bold text-white">{value}%</span>
      </div>
    </div>
  );
}

// Metric Card Component
function MetricCard({ icon: Icon, label, value, subLabel, gradient, delay = 0 }) {
  return (
    <div
      className="glass-card p-5 group animate-slide-up"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-center gap-4">
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center shadow-lg transform group-hover:scale-110 transition-transform duration-300`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
        <div className="flex-1">
          <p className="text-sm text-slate-400">{label}</p>
          <p className="text-xl font-bold text-white">{value}</p>
          {subLabel && (
            <p className="text-xs text-slate-500 mt-0.5">{subLabel}</p>
          )}
        </div>
      </div>
    </div>
  );
}

// Trade Level Row Component
function TradeLevelRow({ label, value, type, icon: Icon }) {
  const colorClass = type === 'entry' ? 'text-indigo-400' :
                     type === 'stop' ? 'text-red-400' :
                     type === 'profit' ? 'text-emerald-400' : 'text-blue-400';
  const bgClass = type === 'entry' ? 'bg-indigo-500/10 border-indigo-500/20' :
                  type === 'stop' ? 'bg-red-500/10 border-red-500/20' :
                  type === 'profit' ? 'bg-emerald-500/10 border-emerald-500/20' : 'bg-blue-500/10 border-blue-500/20';

  return (
    <div className={`flex items-center justify-between p-3 rounded-xl border ${bgClass}`}>
      <div className="flex items-center gap-2">
        <Icon className={`w-4 h-4 ${colorClass}`} />
        <span className="text-sm text-slate-400">{label}</span>
      </div>
      <span className={`font-mono font-medium ${colorClass}`}>{value}</span>
    </div>
  );
}

// Full Signal Analysis Card
function SignalAnalysis({ symbol, analysis, loading }) {
  if (loading) {
    return (
      <div className="glass-card p-8 animate-pulse">
        <div className="flex items-center justify-between mb-8">
          <div className="h-10 w-40 skeleton rounded-lg" />
          <div className="h-12 w-32 skeleton rounded-xl" />
        </div>
        <div className="grid grid-cols-3 gap-6 mb-8">
          <div className="h-24 skeleton rounded-xl" />
          <div className="h-24 skeleton rounded-xl" />
          <div className="h-24 skeleton rounded-xl" />
        </div>
        <div className="space-y-4">
          <div className="h-6 skeleton rounded w-1/4" />
          <div className="h-32 skeleton rounded-xl" />
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
  const isBearish = signal?.direction === 'bearish';

  return (
    <div className="glass-card overflow-hidden relative">
      {/* Gradient accent line */}
      <div className={`absolute top-0 left-0 right-0 h-1 ${
        hasSignal
          ? (isBullish ? 'bg-gradient-to-r from-emerald-500 to-teal-500' : 'bg-gradient-to-r from-red-500 to-orange-500')
          : 'bg-gradient-to-r from-slate-500 to-slate-600'
      }`} />

      {/* Header */}
      <div className={`p-8 relative ${
        hasSignal ? (isBullish ? 'bg-emerald-500/5' : 'bg-red-500/5') : ''
      }`}>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-3xl font-bold text-white">{analysis.symbol}</h2>
              <span className="px-3 py-1 text-xs rounded-full bg-white/5 text-slate-400 border border-white/10">
                Multi-Timeframe
              </span>
            </div>
            <p className="text-slate-400 text-sm mt-2 flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Analyzed: {analysis.timeframes?.join(', ')}
            </p>
          </div>
          {hasSignal && (
            <div className="flex items-center gap-4">
              {/* Pattern Grade Badge (Hedge Fund Level) */}
              {signal.best_pattern_grade && (
                <div className="flex flex-col items-center">
                  <GradeBadge grade={signal.best_pattern_grade} size="lg" />
                  <span className="text-xs text-slate-400 mt-1 capitalize">
                    {signal.best_pattern_type?.replace('_', ' ') || 'Pattern'}
                  </span>
                </div>
              )}
              <div className={`flex items-center gap-3 px-6 py-4 rounded-2xl ${
                isBullish
                  ? 'bg-emerald-500/10 border border-emerald-500/20'
                  : 'bg-red-500/10 border border-red-500/20'
              }`}>
                <div className={`w-14 h-14 rounded-xl flex items-center justify-center ${
                  isBullish ? 'bg-emerald-500/20' : 'bg-red-500/20'
                }`}>
                  {isBullish
                    ? <ArrowUpRight className="w-7 h-7 text-emerald-400" />
                    : <ArrowDownRight className="w-7 h-7 text-red-400" />
                  }
                </div>
                <div>
                  <p className={`text-2xl font-bold ${isBullish ? 'text-emerald-400' : 'text-red-400'}`}>
                    {signal.direction.toUpperCase()}
                  </p>
                  <p className="text-xs text-slate-400">
                    {isBullish ? 'Long positions favored' : 'Short positions favored'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Signal Details */}
      {hasSignal ? (
        <div className="p-8 space-y-8">
          {/* Confidence Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="glass-card-static p-6 flex flex-col items-center text-center">
              <ConfidenceRing
                value={(signal.confidence * 100).toFixed(0)}
                color="#facc15"
              />
              <p className="text-sm text-slate-400 mt-3">Confidence</p>
              <p className="text-xs text-slate-500 flex items-center gap-1 mt-1">
                <Award className="w-3 h-3 text-yellow-400" />
                ML Model Score
              </p>
            </div>
            <div className="glass-card-static p-6 flex flex-col items-center text-center">
              <ConfidenceRing
                value={(signal.strength * 100).toFixed(0)}
                color="#3b82f6"
              />
              <p className="text-sm text-slate-400 mt-3">Signal Strength</p>
              <p className="text-xs text-slate-500 flex items-center gap-1 mt-1">
                <Target className="w-3 h-3 text-blue-400" />
                Technical Score
              </p>
            </div>
            <div className="glass-card-static p-6 flex flex-col items-center text-center">
              <ConfidenceRing
                value={Math.min(signal.confluence_score * 10, 100)}
                color="#a855f7"
              />
              <p className="text-sm text-slate-400 mt-3">Confluence</p>
              <p className="text-xs text-slate-500 flex items-center gap-1 mt-1">
                <Shield className="w-3 h-3 text-purple-400" />
                {signal.confluence_score} factors aligned
              </p>
            </div>
          </div>

          {/* Hedge Fund Level: Pattern Grades */}
          {signal.pattern_grades && Object.keys(signal.pattern_grades).length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5 text-indigo-400" />
                Pattern Grades (Hedge Fund Level)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(signal.pattern_grades).map(([patternType, gradeInfo]) => (
                  <PatternGradeCard key={patternType} patternType={patternType} gradeInfo={gradeInfo} />
                ))}
              </div>
              {signal.grade_recommendation && (
                <div className="mt-4 p-4 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
                  <p className="text-sm text-indigo-300 flex items-center gap-2">
                    <Award className="w-4 h-4" />
                    <strong>Recommendation:</strong> {signal.grade_recommendation}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Edge Statistics (Hedge Fund Level) */}
          {signal.edge_statistics && Object.keys(signal.edge_statistics).length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-purple-400" />
                Edge Statistics
              </h3>
              <div className="glass-card-static p-5">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-xs text-slate-400">Pattern</p>
                    <p className="text-white font-medium capitalize">{signal.edge_statistics.pattern_type?.replace('_', ' ')}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Win Rate</p>
                    <p className="text-emerald-400 font-medium">{signal.edge_statistics.win_rate}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Expectancy</p>
                    <p className="text-purple-400 font-medium">{signal.edge_statistics.expectancy}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Has Edge</p>
                    <p className={`font-medium flex items-center gap-1 ${signal.edge_statistics.has_edge ? 'text-emerald-400' : 'text-red-400'}`}>
                      {signal.edge_statistics.has_edge ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
                      {signal.edge_statistics.has_edge ? 'Yes' : 'No'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Historical Validation (Hedge Fund Level) */}
          {signal.historical_validation && signal.historical_validation.validated && (
            <div>
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Clock className="w-5 h-5 text-amber-400" />
                Historical Validation
              </h3>
              <div className="glass-card-static p-5">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div>
                    <p className="text-xs text-slate-400">Patterns Tested</p>
                    <p className="text-white font-medium">{signal.historical_validation.total_tested}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Fill Rate</p>
                    <p className="text-blue-400 font-medium">{(signal.historical_validation.fill_rate * 100).toFixed(0)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Win Rate</p>
                    <p className="text-emerald-400 font-medium">{(signal.historical_validation.win_rate * 100).toFixed(0)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Avg R:R</p>
                    <p className="text-purple-400 font-medium">{signal.historical_validation.avg_rr_achieved?.toFixed(2)}</p>
                  </div>
                </div>
                {signal.historical_validation.recommendation && (
                  <p className="text-sm text-amber-300 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
                    {signal.historical_validation.recommendation}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Trade Levels */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-400" />
              Trade Levels
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              <TradeLevelRow
                label="Entry Zone"
                value={`${signal.entry_zone[0]?.toFixed(5)} - ${signal.entry_zone[1]?.toFixed(5)}`}
                type="entry"
                icon={Target}
              />
              <TradeLevelRow
                label="Stop Loss"
                value={signal.stop_loss?.toFixed(5)}
                type="stop"
                icon={Shield}
              />
              <TradeLevelRow
                label="Take Profit 1"
                value={signal.take_profit?.[0]?.toFixed(5)}
                type="profit"
                icon={TrendingUp}
              />
              <TradeLevelRow
                label="Risk:Reward"
                value={signal.risk_reward?.toFixed(2)}
                type="rr"
                icon={BarChart3}
              />
            </div>
          </div>

          {/* Concepts */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-indigo-400" />
              Smart Money Concepts Detected
            </h3>
            <div className="flex flex-wrap gap-2">
              {signal.concepts?.map((concept, i) => (
                <span
                  key={i}
                  className="px-4 py-2 text-sm rounded-xl bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 hover:bg-indigo-500/20 transition-colors cursor-default"
                >
                  {concept.replace('_', ' ')}
                </span>
              ))}
            </div>
          </div>

          {/* Reasoning */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-indigo-400" />
              Analysis Reasoning
            </h3>
            <div className="glass-card-static p-5 space-y-3">
              {signal.reasoning?.map((reason, i) => (
                <div key={i} className="flex items-start gap-3 text-sm">
                  <div className="w-6 h-6 rounded-lg bg-indigo-500/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <ChevronRight className="w-4 h-4 text-indigo-400" />
                  </div>
                  <span className="text-slate-300">{reason}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="p-12 text-center">
          <div className="w-20 h-20 rounded-2xl bg-slate-500/10 flex items-center justify-center mx-auto mb-6">
            <Activity className="w-10 h-10 text-slate-500" />
          </div>
          <h3 className="text-xl font-semibold text-slate-400 mb-2">No Signal Generated</h3>
          <p className="text-slate-500 text-sm max-w-md mx-auto">
            Insufficient confluence for a trading signal.
            Multiple Smart Money concepts must align across timeframes for signal generation.
          </p>
        </div>
      )}

      {/* Timeframe Analysis */}
      <div className="border-t border-white/5 p-8">
        <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
          <Clock className="w-5 h-5 text-indigo-400" />
          Timeframe Breakdown
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(analysis.analyses || {}).map(([tf, data], i) => (
            <div
              key={tf}
              className="glass-card-static p-5 animate-slide-up"
              style={{ animationDelay: `${i * 100}ms` }}
            >
              <div className="flex justify-between items-center mb-4">
                <span className="text-lg font-bold text-white">{tf}</span>
                <span className={`px-3 py-1 text-xs font-medium rounded-full ${
                  data.bias === 'bullish' ? 'bg-emerald-500/20 text-emerald-400' :
                  data.bias === 'bearish' ? 'bg-red-500/20 text-red-400' :
                  'bg-slate-500/20 text-slate-400'
                }`}>
                  {data.bias?.toUpperCase() || 'NEUTRAL'}
                </span>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">Zone</span>
                  <span className="text-slate-300 capitalize">{data.zone}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Order Blocks</span>
                  <span className="text-slate-300">{data.order_blocks}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">FVGs</span>
                  <span className="text-slate-300">{data.fvgs}</span>
                </div>
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
    <div className="space-y-8 relative">
      {/* Background decorations */}
      <BackgroundOrb className="w-96 h-96 bg-purple-500 -top-48 -right-48" />
      <BackgroundOrb className="w-72 h-72 bg-indigo-500 bottom-0 -left-36" />

      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3">
            <Zap className="w-8 h-8 text-indigo-400" />
            Signal Analysis
          </h1>
          <p className="text-slate-400 mt-1">
            Full Smart Money methodology analysis with ML insights
          </p>
        </div>
        <button
          onClick={analyzeSelectedSymbol}
          disabled={loading}
          className="btn btn-primary"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Analysis</span>
        </button>
      </div>

      {/* Symbol Selector */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-3 mb-4">
          <Search className="w-5 h-5 text-slate-400" />
          <span className="text-sm font-medium text-slate-300">Select Symbol</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {(symbols.length > 0 ? symbols : defaultSymbols).map((symbol, i) => (
            <button
              key={symbol}
              onClick={() => handleSymbolChange(symbol)}
              className={`px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                selectedSymbol === symbol
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg shadow-indigo-500/25'
                  : 'bg-white/5 text-slate-300 hover:bg-white/10 border border-white/5'
              }`}
              style={{ animationDelay: `${i * 50}ms` }}
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
      <div className="glass-card-static p-5 border-amber-500/20 bg-amber-500/5">
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center flex-shrink-0">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
          </div>
          <div>
            <h4 className="font-medium text-amber-300 mb-1">Risk Disclaimer</h4>
            <p className="text-sm text-amber-200/60 leading-relaxed">
              This is an educational tool. Trading involves significant risk of loss.
              Signals are generated by AI based on Smart Money methodology and should not be considered financial advice.
              Always conduct your own analysis and implement proper risk management.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
