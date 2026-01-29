import React, { useState, useEffect, useCallback } from 'react';
import {
  Shield,
  TrendingUp,
  TrendingDown,
  Target,
  Award,
  BarChart3,
  Activity,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Zap,
  Brain,
  Clock,
  DollarSign,
  Percent,
  ArrowUpRight,
  ArrowDownRight,
  ChevronDown,
  Layers,
  Trophy,
  Star,
  PieChart
} from 'lucide-react';
import {
  getHedgeFundStatus,
  getEdgeStatistics,
  getBestPatterns,
  recordTrade
} from '../services/api';

// Background Orb Component
function BackgroundOrb({ className }) {
  return (
    <div className={`absolute rounded-full blur-3xl opacity-20 animate-pulse ${className}`} />
  );
}

// Grade Badge Component
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
  const Icon = config.icon;
  const sizeClasses = size === 'lg' ? 'w-12 h-12 text-xl' : 'w-8 h-8 text-sm';

  return (
    <div className={`${sizeClasses} ${config.bg} ${config.text} rounded-xl flex items-center justify-center font-bold shadow-lg`}>
      {grade}
    </div>
  );
}

// Stat Card Component
function StatCard({ icon: Icon, label, value, subValue, gradient, trend, delay = 0 }) {
  return (
    <div
      className="glass-card p-5 group animate-slide-up"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center shadow-lg transform group-hover:scale-110 transition-transform duration-300`}>
          <Icon className="w-4 h-4 text-white" />
        </div>
        {trend !== undefined && trend !== null && (
          <span className={`text-xs px-2 py-1 rounded-lg ${
            trend > 0 ? 'bg-emerald-500/10 text-emerald-400' : trend < 0 ? 'bg-red-500/10 text-red-400' : 'bg-slate-500/10 text-slate-400'
          }`}>
            {trend > 0 ? '+' : ''}{trend}%
          </span>
        )}
      </div>
      <p className="text-2xl font-bold text-white">{value}</p>
      <p className="text-sm text-slate-400">{label}</p>
      {subValue && <p className="text-xs text-slate-500 mt-1">{subValue}</p>}
    </div>
  );
}

// Component Status Card
function ComponentStatus({ name, available, description }) {
  return (
    <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10">
      <div className="flex items-center gap-3">
        <div className={`w-3 h-3 rounded-full ${available ? 'bg-emerald-500' : 'bg-red-500'}`} />
        <div>
          <p className="text-white font-medium">{name}</p>
          <p className="text-xs text-slate-400">{description}</p>
        </div>
      </div>
      {available ? (
        <CheckCircle className="w-5 h-5 text-emerald-400" />
      ) : (
        <XCircle className="w-5 h-5 text-red-400" />
      )}
    </div>
  );
}

// Pattern Edge Card
function PatternEdgeCard({ pattern, index }) {
  const hasEdge = pattern.expectancy > 0;

  return (
    <div className={`glass-card p-5 animate-slide-up`} style={{ animationDelay: `${index * 100}ms` }}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
            hasEdge ? 'bg-gradient-to-br from-emerald-500 to-green-600' : 'bg-gradient-to-br from-red-500 to-red-700'
          }`}>
            <span className="text-white font-bold text-sm">#{index + 1}</span>
          </div>
          <div>
            <p className="text-white font-semibold capitalize">{pattern.pattern_type.replace('_', ' ')}</p>
            <p className="text-xs text-slate-400">{pattern.total_signals} signals tracked</p>
          </div>
        </div>
        {hasEdge ? (
          <div className="flex items-center gap-1 text-emerald-400">
            <TrendingUp className="w-4 h-4" />
            <span className="text-xs font-medium">Edge</span>
          </div>
        ) : (
          <div className="flex items-center gap-1 text-red-400">
            <TrendingDown className="w-4 h-4" />
            <span className="text-xs font-medium">No Edge</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <p className="text-xs text-slate-400">Win Rate</p>
          <p className={`text-lg font-bold ${pattern.win_rate >= 0.5 ? 'text-emerald-400' : 'text-red-400'}`}>
            {(pattern.win_rate * 100).toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Expectancy</p>
          <p className={`text-lg font-bold ${pattern.expectancy > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {pattern.expectancy > 0 ? '+' : ''}{pattern.expectancy.toFixed(2)}R
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Signals</p>
          <p className="text-lg font-bold text-white">{pattern.total_signals}</p>
        </div>
      </div>
    </div>
  );
}

// Paper Trading Form Component
function PaperTradingForm({ onRecordTrade, isRecording }) {
  const [formData, setFormData] = useState({
    patternType: 'fvg',
    outcome: 'win',
    rrAchieved: '',
    session: '',
    dayOfWeek: ''
  });

  const patternTypes = [
    { value: 'fvg', label: 'Fair Value Gap' },
    { value: 'order_block', label: 'Order Block' },
    { value: 'breaker', label: 'Breaker Block' },
    { value: 'liquidity', label: 'Liquidity Level' },
    { value: 'structure', label: 'Market Structure' },
  ];

  const sessions = [
    { value: '', label: 'Select Session' },
    { value: 'asian', label: 'Asian Session' },
    { value: 'london', label: 'London Session' },
    { value: 'new_york', label: 'New York Session' },
    { value: 'off_hours', label: 'Off Hours' },
  ];

  const days = [
    { value: '', label: 'Select Day' },
    { value: 'Monday', label: 'Monday' },
    { value: 'Tuesday', label: 'Tuesday' },
    { value: 'Wednesday', label: 'Wednesday' },
    { value: 'Thursday', label: 'Thursday' },
    { value: 'Friday', label: 'Friday' },
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    onRecordTrade({
      ...formData,
      rrAchieved: parseFloat(formData.rrAchieved) || 0
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Pattern Type */}
      <div>
        <label className="block text-sm font-medium text-slate-400 mb-2">Pattern Type</label>
        <select
          value={formData.patternType}
          onChange={(e) => setFormData({ ...formData, patternType: e.target.value })}
          className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white focus:border-indigo-500 focus:outline-none"
        >
          {patternTypes.map((pt) => (
            <option key={pt.value} value={pt.value} className="bg-slate-900">{pt.label}</option>
          ))}
        </select>
      </div>

      {/* Outcome Buttons */}
      <div>
        <label className="block text-sm font-medium text-slate-400 mb-2">Trade Outcome</label>
        <div className="grid grid-cols-3 gap-2">
          {['win', 'loss', 'breakeven'].map((outcome) => (
            <button
              key={outcome}
              type="button"
              onClick={() => setFormData({ ...formData, outcome })}
              className={`px-4 py-3 rounded-xl font-medium transition-all ${
                formData.outcome === outcome
                  ? outcome === 'win'
                    ? 'bg-emerald-500 text-white'
                    : outcome === 'loss'
                    ? 'bg-red-500 text-white'
                    : 'bg-yellow-500 text-black'
                  : 'bg-white/5 text-slate-400 hover:bg-white/10'
              }`}
            >
              {outcome === 'win' && <TrendingUp className="w-4 h-4 inline mr-1" />}
              {outcome === 'loss' && <TrendingDown className="w-4 h-4 inline mr-1" />}
              {outcome === 'breakeven' && <Target className="w-4 h-4 inline mr-1" />}
              {outcome.charAt(0).toUpperCase() + outcome.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* R:R Achieved */}
      <div>
        <label className="block text-sm font-medium text-slate-400 mb-2">R:R Achieved (optional)</label>
        <input
          type="number"
          step="0.1"
          placeholder="e.g., 2.5 for 2.5:1"
          value={formData.rrAchieved}
          onChange={(e) => setFormData({ ...formData, rrAchieved: e.target.value })}
          className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
        />
      </div>

      {/* Session & Day Row */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-slate-400 mb-2">Session</label>
          <select
            value={formData.session}
            onChange={(e) => setFormData({ ...formData, session: e.target.value })}
            className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white focus:border-indigo-500 focus:outline-none"
          >
            {sessions.map((s) => (
              <option key={s.value} value={s.value} className="bg-slate-900">{s.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-400 mb-2">Day</label>
          <select
            value={formData.dayOfWeek}
            onChange={(e) => setFormData({ ...formData, dayOfWeek: e.target.value })}
            className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white focus:border-indigo-500 focus:outline-none"
          >
            {days.map((d) => (
              <option key={d.value} value={d.value} className="bg-slate-900">{d.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isRecording}
        className="w-full py-4 rounded-xl bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 text-white font-semibold hover:shadow-lg hover:shadow-indigo-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isRecording ? (
          <span className="flex items-center justify-center gap-2">
            <RefreshCw className="w-4 h-4 animate-spin" />
            Recording...
          </span>
        ) : (
          <span className="flex items-center justify-center gap-2">
            <Activity className="w-4 h-4" />
            Record Trade Outcome
          </span>
        )}
      </button>
    </form>
  );
}

// Main HedgeFund Component
export default function HedgeFund() {
  const [status, setStatus] = useState(null);
  const [edgeStats, setEdgeStats] = useState(null);
  const [bestPatterns, setBestPatterns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordSuccess, setRecordSuccess] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const [statusRes, edgeRes, bestRes] = await Promise.all([
        getHedgeFundStatus(),
        getEdgeStatistics(),
        getBestPatterns(5)
      ]);

      setStatus(statusRes);
      setEdgeStats(edgeRes);
      setBestPatterns(bestRes.best_patterns || []);
    } catch (err) {
      console.error('Failed to fetch hedge fund data:', err);
      setError(err.message || 'Failed to connect to hedge fund API');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleRecordTrade = async (tradeData) => {
    try {
      setIsRecording(true);
      setRecordSuccess(null);

      const result = await recordTrade(
        tradeData.patternType,
        tradeData.outcome,
        tradeData.rrAchieved,
        tradeData.session,
        tradeData.dayOfWeek
      );

      setRecordSuccess(`Trade recorded! ${tradeData.outcome.toUpperCase()} for ${tradeData.patternType}`);

      // Refresh data after recording
      await fetchData();

      // Clear success message after 3 seconds
      setTimeout(() => setRecordSuccess(null), 3000);
    } catch (err) {
      console.error('Failed to record trade:', err);
      setError('Failed to record trade: ' + err.message);
    } finally {
      setIsRecording(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <RefreshCw className="w-8 h-8 text-indigo-500 animate-spin" />
          <p className="text-slate-400">Loading hedge fund features...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="glass-card p-8 text-center max-w-md">
          <AlertTriangle className="w-12 h-12 text-amber-400 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Connection Error</h2>
          <p className="text-slate-400 mb-4">{error}</p>
          <button
            onClick={fetchData}
            className="px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-500 text-white font-medium hover:shadow-lg transition-all"
          >
            <RefreshCw className="w-4 h-4 inline mr-2" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  const allPatternStats = edgeStats?.all_patterns || {};
  const totalPatterns = Object.keys(allPatternStats).length;
  const patternsWithEdge = Object.values(allPatternStats).filter(p => p.has_edge).length;

  return (
    <div className="relative space-y-8">
      {/* Background Effects */}
      <BackgroundOrb className="w-96 h-96 bg-indigo-600 -top-48 -left-48" />
      <BackgroundOrb className="w-96 h-96 bg-purple-600 top-1/2 -right-48" />
      <BackgroundOrb className="w-96 h-96 bg-pink-600 bottom-0 left-1/4" />

      {/* Header */}
      <div className="relative">
        <div className="flex items-center gap-4 mb-2">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/25">
            <Shield className="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">Hedge Fund Level</h1>
            <p className="text-slate-400">Institutional-grade pattern analysis & edge tracking</p>
          </div>
        </div>
      </div>

      {/* Success Message */}
      {recordSuccess && (
        <div className="glass-card p-4 bg-emerald-500/10 border-emerald-500/20 animate-fade-in">
          <div className="flex items-center gap-3">
            <CheckCircle className="w-5 h-5 text-emerald-400" />
            <p className="text-emerald-400 font-medium">{recordSuccess}</p>
          </div>
        </div>
      )}

      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={Shield}
          label="Components Active"
          value={status?.available ? '4/4' : '0/4'}
          gradient="from-indigo-500 to-purple-600"
          delay={0}
        />
        <StatCard
          icon={BarChart3}
          label="Patterns Tracked"
          value={totalPatterns}
          gradient="from-purple-500 to-pink-600"
          delay={100}
        />
        <StatCard
          icon={TrendingUp}
          label="With Edge"
          value={patternsWithEdge}
          subValue={`${totalPatterns > 0 ? ((patternsWithEdge / totalPatterns) * 100).toFixed(0) : 0}% of patterns`}
          gradient="from-emerald-500 to-green-600"
          delay={200}
        />
        <StatCard
          icon={Award}
          label="Best Patterns"
          value={bestPatterns.length}
          subValue="Above threshold"
          gradient="from-amber-500 to-orange-600"
          delay={300}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Components & Paper Trading */}
        <div className="space-y-6">
          {/* Component Status */}
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Layers className="w-5 h-5 text-indigo-400" />
                Hedge Fund Components
              </h2>
              {status?.available && (
                <span className="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-medium">
                  All Active
                </span>
              )}
            </div>
            <div className="space-y-3">
              <ComponentStatus
                name="Pattern Grader"
                available={status?.components?.pattern_grader}
                description="Grades patterns A+ to F"
              />
              <ComponentStatus
                name="Edge Tracker"
                available={status?.components?.edge_tracker}
                description="Tracks statistical edge"
              />
              <ComponentStatus
                name="Historical Validator"
                available={status?.components?.historical_validator}
                description="Validates against history"
              />
              <ComponentStatus
                name="MTF Analyzer"
                available={status?.components?.mtf_analyzer}
                description="Multi-timeframe confluence"
              />
            </div>
          </div>

          {/* Paper Trading Form */}
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Activity className="w-5 h-5 text-purple-400" />
                Record Trade Outcome
              </h2>
            </div>
            <p className="text-sm text-slate-400 mb-4">
              Record your trade outcomes to improve edge tracking. The ML learns from your results!
            </p>
            <PaperTradingForm onRecordTrade={handleRecordTrade} isRecording={isRecording} />
          </div>
        </div>

        {/* Middle Column - Best Patterns */}
        <div className="lg:col-span-2 space-y-6">
          {/* Best Patterns Ranking */}
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Trophy className="w-5 h-5 text-amber-400" />
                Best Performing Patterns
              </h2>
              <button
                onClick={fetchData}
                className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>

            {bestPatterns.length > 0 ? (
              <div className="space-y-4">
                {bestPatterns.map((pattern, index) => (
                  <PatternEdgeCard key={pattern.pattern_type} pattern={pattern} index={index} />
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <BarChart3 className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-400">No pattern statistics yet</p>
                <p className="text-sm text-slate-500 mt-1">Record trades to build edge statistics</p>
              </div>
            )}
          </div>

          {/* All Pattern Stats */}
          {Object.keys(allPatternStats).length > 0 && (
            <div className="glass-card p-6">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
                <PieChart className="w-5 h-5 text-indigo-400" />
                All Pattern Statistics
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-sm text-slate-400 border-b border-white/10">
                      <th className="pb-3">Pattern</th>
                      <th className="pb-3">Win Rate</th>
                      <th className="pb-3">Expectancy</th>
                      <th className="pb-3">Edge</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {Object.entries(allPatternStats).map(([patternType, stats]) => (
                      <tr key={patternType} className="text-sm">
                        <td className="py-3 text-white font-medium capitalize">
                          {patternType.replace('_', ' ')}
                        </td>
                        <td className="py-3 text-slate-300">{stats.win_rate}</td>
                        <td className="py-3 text-slate-300">{stats.expectancy}</td>
                        <td className="py-3">
                          {stats.has_edge ? (
                            <span className="flex items-center gap-1 text-emerald-400">
                              <CheckCircle className="w-4 h-4" /> Yes
                            </span>
                          ) : (
                            <span className="flex items-center gap-1 text-red-400">
                              <XCircle className="w-4 h-4" /> No
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Grade Legend */}
          <div className="glass-card p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Pattern Grade Legend</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {[
                { grade: 'A+', desc: 'Institutional quality - Full position' },
                { grade: 'A', desc: 'Excellent setup - Standard position' },
                { grade: 'B', desc: 'Good setup - Reduced position' },
                { grade: 'C', desc: 'Average - Wait for better' },
                { grade: 'D', desc: 'Poor quality - Avoid' },
                { grade: 'F', desc: 'Invalid pattern - Do not trade' },
              ].map(({ grade, desc }) => (
                <div key={grade} className="flex items-center gap-3 p-3 rounded-xl bg-white/5">
                  <GradeBadge grade={grade} />
                  <div>
                    <p className="text-white font-medium">{grade}</p>
                    <p className="text-xs text-slate-400">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
