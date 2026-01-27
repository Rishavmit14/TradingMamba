import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Target,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  RefreshCw,
  BarChart3,
  PieChart,
  Activity,
  Zap,
  Sparkles,
  ChevronDown,
  Layers,
  Award
} from 'lucide-react';

const API_URL = 'http://localhost:8000/api';

// Background Orb Component
function BackgroundOrb({ className }) {
  return (
    <div className={`absolute rounded-full blur-3xl opacity-20 animate-pulse ${className}`} />
  );
}

// Animated Progress Ring
function ProgressRing({ value, size = 80, strokeWidth = 6, color, label }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
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
          <span className="text-lg font-bold text-white">{value.toFixed(1)}%</span>
        </div>
      </div>
      {label && <p className="text-xs text-slate-400 mt-2">{label}</p>}
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
            trend > 0 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
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

// Progress Bar Component
function ProgressBar({ value, max = 100, label, color = 'indigo', total }) {
  const percentage = Math.min((value / max) * 100, 100);
  const colors = {
    indigo: 'from-indigo-500 to-purple-500',
    emerald: 'from-emerald-500 to-teal-500',
    red: 'from-red-500 to-orange-500',
    amber: 'from-amber-500 to-yellow-500',
  };

  return (
    <div className="mb-4 last:mb-0">
      <div className="flex justify-between text-sm mb-2">
        <span className="text-slate-400">{label}</span>
        <div className="flex items-center gap-2">
          {total && <span className="text-slate-500 text-xs">{total} signals</span>}
          <span className="text-white font-medium">{value.toFixed(1)}%</span>
        </div>
      </div>
      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
        <div
          className={`h-full bg-gradient-to-r ${colors[color]} transition-all duration-1000 ease-out rounded-full`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

// Recommendation Card
function RecommendationCard({ rec, delay = 0 }) {
  const config = {
    warning: { icon: AlertTriangle, gradient: 'from-amber-500 to-yellow-500', bg: 'bg-amber-500/10', border: 'border-amber-500/20' },
    info: { icon: Activity, gradient: 'from-blue-500 to-cyan-500', bg: 'bg-blue-500/10', border: 'border-blue-500/20' },
    increase_weight: { icon: TrendingUp, gradient: 'from-emerald-500 to-teal-500', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20' },
    reduce_weight: { icon: TrendingDown, gradient: 'from-red-500 to-orange-500', bg: 'bg-red-500/10', border: 'border-red-500/20' },
  };

  const { icon: Icon, gradient, bg, border } = config[rec.type] || config.info;

  return (
    <div
      className={`glass-card-static p-4 ${bg} border ${border} animate-slide-up`}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-start gap-4">
        <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center flex-shrink-0`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
        <div className="flex-1">
          <p className="text-sm text-slate-300">{rec.message}</p>
          {rec.concept && (
            <span className="inline-block mt-2 text-xs px-2 py-1 bg-white/5 text-slate-400 rounded-lg border border-white/10">
              Concept: {rec.concept}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// Main Performance Page
export default function Performance() {
  const [summary, setSummary] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [recentSignals, setRecentSignals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [days, setDays] = useState(30);

  useEffect(() => {
    loadData();
  }, [days]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [summaryRes, feedbackRes, signalsRes] = await Promise.all([
        fetch(`${API_URL}/performance/summary?days=${days}`).then(r => r.json()).catch(() => null),
        fetch(`${API_URL}/performance/feedback`).then(r => r.json()).catch(() => null),
        fetch(`${API_URL}/performance/signals?limit=10`).then(r => r.json()).catch(() => ({ signals: [] })),
      ]);

      setSummary(summaryRes);
      setFeedback(feedbackRes);
      setRecentSignals(signalsRes?.signals || []);
    } catch (err) {
      console.error('Failed to load performance data:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatPnl = (pnl) => {
    if (pnl === null || pnl === undefined) return 'N/A';
    return pnl >= 0 ? `+${pnl.toFixed(1)}` : pnl.toFixed(1);
  };

  return (
    <div className="space-y-8 relative">
      {/* Background decorations */}
      <BackgroundOrb className="w-96 h-96 bg-emerald-500 -top-48 -right-48" />
      <BackgroundOrb className="w-72 h-72 bg-purple-500 bottom-0 -left-36" />

      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3">
            <BarChart3 className="w-8 h-8 text-indigo-400" />
            Performance Analytics
          </h1>
          <p className="text-slate-400 mt-1">
            Track and analyze signal performance
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="glass-card p-1 relative">
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="bg-transparent text-white px-4 py-2 pr-8 appearance-none cursor-pointer focus:outline-none"
            >
              <option value={7} className="bg-[#0a0a0f]">Last 7 days</option>
              <option value={14} className="bg-[#0a0a0f]">Last 14 days</option>
              <option value={30} className="bg-[#0a0a0f]">Last 30 days</option>
              <option value={90} className="bg-[#0a0a0f]">Last 90 days</option>
            </select>
            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
          </div>
          <button
            onClick={loadData}
            disabled={loading}
            className="btn btn-secondary"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Key Metrics */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <StatCard
            icon={BarChart3}
            label="Total Signals"
            value={summary.total_signals}
            subValue={`${summary.closed_signals} closed`}
            gradient="from-blue-500 to-cyan-500"
            delay={0}
          />
          <StatCard
            icon={Target}
            label="Win Rate"
            value={`${(summary.win_rate * 100).toFixed(1)}%`}
            gradient={summary.win_rate >= 0.5 ? 'from-emerald-500 to-teal-500' : 'from-red-500 to-orange-500'}
            delay={50}
          />
          <StatCard
            icon={CheckCircle}
            label="Wins"
            value={summary.wins}
            gradient="from-emerald-500 to-green-500"
            delay={100}
          />
          <StatCard
            icon={XCircle}
            label="Losses"
            value={summary.losses}
            gradient="from-red-500 to-rose-500"
            delay={150}
          />
          <StatCard
            icon={TrendingUp}
            label="Total PnL"
            value={`${formatPnl(summary.total_pnl_pips)} pips`}
            gradient={summary.total_pnl_pips >= 0 ? 'from-emerald-500 to-teal-500' : 'from-red-500 to-orange-500'}
            delay={200}
          />
          <StatCard
            icon={Award}
            label="Profit Factor"
            value={summary.profit_factor === 'N/A' ? 'N/A' : summary.profit_factor?.toFixed(2)}
            gradient={summary.profit_factor !== 'N/A' && summary.profit_factor >= 1.5 ? 'from-emerald-500 to-teal-500' : 'from-amber-500 to-orange-500'}
            delay={250}
          />
        </div>
      )}

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* By Direction */}
        {summary?.by_direction && (
          <div className="glass-card p-6">
            <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
              <PieChart className="w-5 h-5 text-indigo-400" />
              Performance by Direction
            </h2>
            <div className="space-y-4">
              <div className="glass-card-static p-4 bg-emerald-500/5 border-emerald-500/20">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                      <TrendingUp className="w-6 h-6 text-emerald-400" />
                    </div>
                    <div>
                      <p className="font-semibold text-white">BUY Signals</p>
                      <p className="text-sm text-slate-400">
                        {summary.by_direction.BUY.wins}/{summary.by_direction.BUY.total} won
                      </p>
                    </div>
                  </div>
                  <ProgressRing
                    value={summary.by_direction.BUY.win_rate * 100}
                    color={summary.by_direction.BUY.win_rate >= 0.5 ? '#10b981' : '#ef4444'}
                    size={70}
                  />
                </div>
              </div>

              <div className="glass-card-static p-4 bg-red-500/5 border-red-500/20">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-red-500/20 flex items-center justify-center">
                      <TrendingDown className="w-6 h-6 text-red-400" />
                    </div>
                    <div>
                      <p className="font-semibold text-white">SELL Signals</p>
                      <p className="text-sm text-slate-400">
                        {summary.by_direction.SELL.wins}/{summary.by_direction.SELL.total} won
                      </p>
                    </div>
                  </div>
                  <ProgressRing
                    value={summary.by_direction.SELL.win_rate * 100}
                    color={summary.by_direction.SELL.win_rate >= 0.5 ? '#10b981' : '#ef4444'}
                    size={70}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* By Confidence */}
        {summary?.by_confidence && (
          <div className="glass-card p-6">
            <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-400" />
              Performance by Confidence
            </h2>
            <div className="space-y-2">
              <ProgressBar
                value={summary.by_confidence['high_75+'].win_rate * 100}
                label="High Confidence (75%+)"
                color="emerald"
                total={summary.by_confidence['high_75+'].total}
              />
              <ProgressBar
                value={summary.by_confidence['medium_50-75'].win_rate * 100}
                label="Medium Confidence (50-75%)"
                color="indigo"
                total={summary.by_confidence['medium_50-75'].total}
              />
              <ProgressBar
                value={summary.by_confidence['low_under_50'].win_rate * 100}
                label="Low Confidence (<50%)"
                color="red"
                total={summary.by_confidence['low_under_50'].total}
              />
            </div>
          </div>
        )}
      </div>

      {/* Concept Performance */}
      {summary?.by_concept && Object.keys(summary.by_concept).length > 0 && (
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-indigo-400" />
            Performance by Smart Money Concept
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {Object.entries(summary.by_concept).slice(0, 8).map(([concept, data], i) => (
              <div
                key={concept}
                className="glass-card-static p-4 animate-slide-up"
                style={{ animationDelay: `${i * 50}ms` }}
              >
                <p className="font-medium text-white text-sm capitalize mb-3">
                  {concept.replace(/_/g, ' ')}
                </p>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs text-slate-400">{data.total} signals</span>
                  <span className={`text-lg font-bold ${
                    data.win_rate >= 0.6 ? 'text-emerald-400' :
                    data.win_rate >= 0.4 ? 'text-amber-400' : 'text-red-400'
                  }`}>
                    {(data.win_rate * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      data.win_rate >= 0.6 ? 'bg-gradient-to-r from-emerald-500 to-teal-500' :
                      data.win_rate >= 0.4 ? 'bg-gradient-to-r from-amber-500 to-yellow-500' :
                      'bg-gradient-to-r from-red-500 to-orange-500'
                    }`}
                    style={{ width: `${data.win_rate * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* AI Recommendations */}
      {feedback?.recommendations && feedback.recommendations.length > 0 && (
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-400" />
            AI Recommendations
          </h2>
          <div className="space-y-3">
            {feedback.recommendations.map((rec, i) => (
              <RecommendationCard key={i} rec={rec} delay={i * 100} />
            ))}
          </div>
        </div>
      )}

      {/* Recent Signals */}
      <div className="glass-card p-6">
        <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
          <Clock className="w-5 h-5 text-indigo-400" />
          Recent Signals
        </h2>

        {recentSignals.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-400 border-b border-white/5">
                  <th className="text-left py-3 px-4 font-medium">Symbol</th>
                  <th className="text-left py-3 px-4 font-medium">Direction</th>
                  <th className="text-left py-3 px-4 font-medium">Confidence</th>
                  <th className="text-left py-3 px-4 font-medium">Status</th>
                  <th className="text-left py-3 px-4 font-medium">PnL</th>
                  <th className="text-left py-3 px-4 font-medium">Kill Zone</th>
                  <th className="text-left py-3 px-4 font-medium">Date</th>
                </tr>
              </thead>
              <tbody>
                {recentSignals.map((signal, i) => (
                  <tr
                    key={i}
                    className="border-b border-white/5 hover:bg-white/[0.02] transition-colors animate-slide-up"
                    style={{ animationDelay: `${i * 50}ms` }}
                  >
                    <td className="py-3 px-4 font-medium text-white">{signal.symbol}</td>
                    <td className="py-3 px-4">
                      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium ${
                        signal.direction === 'BUY'
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : 'bg-red-500/10 text-red-400 border border-red-500/20'
                      }`}>
                        {signal.direction === 'BUY' ? (
                          <TrendingUp className="w-3 h-3" />
                        ) : (
                          <TrendingDown className="w-3 h-3" />
                        )}
                        {signal.direction}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-indigo-500 rounded-full"
                            style={{ width: `${signal.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-slate-300">{(signal.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2.5 py-1 rounded-lg text-xs font-medium ${
                        signal.status === 'tp_hit' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                        signal.status === 'sl_hit' ? 'bg-red-500/10 text-red-400 border border-red-500/20' :
                        signal.status === 'pending' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' :
                        'bg-slate-500/10 text-slate-400 border border-slate-500/20'
                      }`}>
                        {signal.status}
                      </span>
                    </td>
                    <td className={`py-3 px-4 font-medium font-mono ${
                      signal.pnl_pips > 0 ? 'text-emerald-400' :
                      signal.pnl_pips < 0 ? 'text-red-400' : 'text-slate-400'
                    }`}>
                      {signal.pnl_pips ? `${formatPnl(signal.pnl_pips)} pips` : '-'}
                    </td>
                    <td className="py-3 px-4 text-slate-400 text-xs">
                      {signal.kill_zone || '-'}
                    </td>
                    <td className="py-3 px-4 text-slate-500 text-xs">
                      {new Date(signal.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="w-20 h-20 rounded-2xl bg-slate-500/10 flex items-center justify-center mx-auto mb-6">
              <BarChart3 className="w-10 h-10 text-slate-500" />
            </div>
            <h3 className="text-lg font-semibold text-slate-400 mb-2">No signals recorded yet</h3>
            <p className="text-sm text-slate-500">
              Generate some signals to see performance data
            </p>
          </div>
        )}
      </div>

      {/* Info Box */}
      <div className="glass-card-static p-6 border-indigo-500/20 bg-indigo-500/5">
        <h2 className="text-xl font-bold text-indigo-300 mb-4 flex items-center gap-2">
          <Sparkles className="w-5 h-5" />
          About Performance Tracking
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
          <div>
            <p className="font-medium text-white mb-2">How it works:</p>
            <ul className="space-y-2 text-slate-400">
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-2 flex-shrink-0" />
                Signals are automatically recorded when generated
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-2 flex-shrink-0" />
                Update outcomes manually or via API
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-2 flex-shrink-0" />
                AI analyzes patterns in winning/losing trades
              </li>
            </ul>
          </div>
          <div>
            <p className="font-medium text-white mb-2">Improvement loop:</p>
            <ul className="space-y-2 text-slate-400">
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-2 flex-shrink-0" />
                Concepts with high win rates get increased weight
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-2 flex-shrink-0" />
                Low-performing patterns are flagged
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-2 flex-shrink-0" />
                Kill zone performance is tracked
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
