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
  Activity
} from 'lucide-react';

const API_URL = 'http://localhost:8000';

// Stat Card Component
function StatCard({ icon: Icon, label, value, subValue, color = 'gray', trend }) {
  const colors = {
    green: 'text-green-400 bg-green-500/10 border-green-500/30',
    red: 'text-red-400 bg-red-500/10 border-red-500/30',
    blue: 'text-blue-400 bg-blue-500/10 border-blue-500/30',
    purple: 'text-purple-400 bg-purple-500/10 border-purple-500/30',
    orange: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
    gray: 'text-gray-400 bg-gray-500/10 border-gray-500/30',
  };

  return (
    <div className={`rounded-xl p-4 border ${colors[color]}`}>
      <div className="flex items-center justify-between mb-2">
        <Icon size={20} />
        {trend && (
          <span className={`text-xs ${trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {trend > 0 ? '+' : ''}{trend}%
          </span>
        )}
      </div>
      <p className="text-2xl font-bold">{value}</p>
      <p className="text-sm text-gray-400">{label}</p>
      {subValue && <p className="text-xs text-gray-500 mt-1">{subValue}</p>}
    </div>
  );
}

// Progress Bar
function ProgressBar({ value, max = 100, label, color = 'blue' }) {
  const percentage = Math.min((value / max) * 100, 100);
  const colors = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    red: 'bg-red-500',
    purple: 'bg-purple-500',
  };

  return (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-white">{value.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${colors[color]} transition-all duration-500`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

// Recommendation Card
function RecommendationCard({ rec }) {
  const icons = {
    warning: <AlertTriangle className="text-yellow-400" size={18} />,
    info: <Activity className="text-blue-400" size={18} />,
    increase_weight: <TrendingUp className="text-green-400" size={18} />,
    reduce_weight: <TrendingDown className="text-red-400" size={18} />,
  };

  const bgColors = {
    warning: 'bg-yellow-500/10 border-yellow-500/30',
    info: 'bg-blue-500/10 border-blue-500/30',
    increase_weight: 'bg-green-500/10 border-green-500/30',
    reduce_weight: 'bg-red-500/10 border-red-500/30',
  };

  return (
    <div className={`p-3 rounded-lg border ${bgColors[rec.type] || bgColors.info}`}>
      <div className="flex items-start space-x-3">
        {icons[rec.type] || icons.info}
        <div>
          <p className="text-sm">{rec.message}</p>
          {rec.concept && (
            <span className="text-xs text-gray-500">Concept: {rec.concept}</span>
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
        fetch(`${API_URL}/api/performance/summary?days=${days}`).then(r => r.json()).catch(() => null),
        fetch(`${API_URL}/api/performance/feedback`).then(r => r.json()).catch(() => null),
        fetch(`${API_URL}/api/performance/signals?limit=10`).then(r => r.json()).catch(() => ({ signals: [] })),
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
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Performance Analytics</h1>
          <p className="text-gray-400">Track and analyze signal performance</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2"
          >
            <option value={7}>Last 7 days</option>
            <option value={14}>Last 14 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
          <button
            onClick={loadData}
            className="flex items-center space-x-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Key Metrics */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <StatCard
            icon={BarChart3}
            label="Total Signals"
            value={summary.total_signals}
            subValue={`${summary.closed_signals} closed`}
            color="blue"
          />
          <StatCard
            icon={Target}
            label="Win Rate"
            value={`${(summary.win_rate * 100).toFixed(1)}%`}
            color={summary.win_rate >= 0.5 ? 'green' : 'red'}
          />
          <StatCard
            icon={CheckCircle}
            label="Wins"
            value={summary.wins}
            color="green"
          />
          <StatCard
            icon={XCircle}
            label="Losses"
            value={summary.losses}
            color="red"
          />
          <StatCard
            icon={TrendingUp}
            label="Total PnL"
            value={`${formatPnl(summary.total_pnl_pips)} pips`}
            color={summary.total_pnl_pips >= 0 ? 'green' : 'red'}
          />
          <StatCard
            icon={Activity}
            label="Profit Factor"
            value={summary.profit_factor === 'N/A' ? 'N/A' : summary.profit_factor?.toFixed(2)}
            color={summary.profit_factor !== 'N/A' && summary.profit_factor >= 1.5 ? 'green' : 'orange'}
          />
        </div>
      )}

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* By Direction */}
        {summary?.by_direction && (
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 className="text-lg font-bold mb-4 flex items-center">
              <PieChart className="mr-2" size={20} />
              Performance by Direction
            </h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-green-500/10 rounded-lg">
                <div className="flex items-center">
                  <TrendingUp className="text-green-400 mr-3" size={24} />
                  <div>
                    <p className="font-medium">BUY Signals</p>
                    <p className="text-sm text-gray-400">
                      {summary.by_direction.BUY.wins}/{summary.by_direction.BUY.total} won
                    </p>
                  </div>
                </div>
                <span className={`text-xl font-bold ${
                  summary.by_direction.BUY.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {(summary.by_direction.BUY.win_rate * 100).toFixed(1)}%
                </span>
              </div>

              <div className="flex items-center justify-between p-3 bg-red-500/10 rounded-lg">
                <div className="flex items-center">
                  <TrendingDown className="text-red-400 mr-3" size={24} />
                  <div>
                    <p className="font-medium">SELL Signals</p>
                    <p className="text-sm text-gray-400">
                      {summary.by_direction.SELL.wins}/{summary.by_direction.SELL.total} won
                    </p>
                  </div>
                </div>
                <span className={`text-xl font-bold ${
                  summary.by_direction.SELL.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {(summary.by_direction.SELL.win_rate * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}

        {/* By Confidence */}
        {summary?.by_confidence && (
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 className="text-lg font-bold mb-4">Performance by Confidence</h2>
            <div className="space-y-4">
              <ProgressBar
                value={summary.by_confidence['high_75+'].win_rate * 100}
                label={`High Confidence (75%+) - ${summary.by_confidence['high_75+'].total} signals`}
                color="green"
              />
              <ProgressBar
                value={summary.by_confidence['medium_50-75'].win_rate * 100}
                label={`Medium Confidence (50-75%) - ${summary.by_confidence['medium_50-75'].total} signals`}
                color="blue"
              />
              <ProgressBar
                value={summary.by_confidence['low_under_50'].win_rate * 100}
                label={`Low Confidence (<50%) - ${summary.by_confidence['low_under_50'].total} signals`}
                color="red"
              />
            </div>
          </div>
        )}
      </div>

      {/* Concept Performance */}
      {summary?.by_concept && Object.keys(summary.by_concept).length > 0 && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h2 className="text-lg font-bold mb-4">Performance by Smart Money Concept</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {Object.entries(summary.by_concept).slice(0, 8).map(([concept, data]) => (
              <div key={concept} className="p-3 bg-gray-700/50 rounded-lg">
                <p className="font-medium text-sm capitalize mb-2">
                  {concept.replace(/_/g, ' ')}
                </p>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">{data.total} signals</span>
                  <span className={`font-bold ${
                    data.win_rate >= 0.6 ? 'text-green-400' :
                    data.win_rate >= 0.4 ? 'text-yellow-400' : 'text-red-400'
                  }`}>
                    {(data.win_rate * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="mt-2 h-1 bg-gray-600 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${
                      data.win_rate >= 0.6 ? 'bg-green-500' :
                      data.win_rate >= 0.4 ? 'bg-yellow-500' : 'bg-red-500'
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
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h2 className="text-lg font-bold mb-4 flex items-center">
            <AlertTriangle className="mr-2 text-yellow-400" size={20} />
            AI Recommendations
          </h2>
          <div className="space-y-3">
            {feedback.recommendations.map((rec, i) => (
              <RecommendationCard key={i} rec={rec} />
            ))}
          </div>
        </div>
      )}

      {/* Recent Signals */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h2 className="text-lg font-bold mb-4 flex items-center">
          <Clock className="mr-2" size={20} />
          Recent Signals
        </h2>

        {recentSignals.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 border-b border-gray-700">
                  <th className="text-left py-2 px-3">Symbol</th>
                  <th className="text-left py-2 px-3">Direction</th>
                  <th className="text-left py-2 px-3">Confidence</th>
                  <th className="text-left py-2 px-3">Status</th>
                  <th className="text-left py-2 px-3">PnL</th>
                  <th className="text-left py-2 px-3">Kill Zone</th>
                  <th className="text-left py-2 px-3">Date</th>
                </tr>
              </thead>
              <tbody>
                {recentSignals.map((signal, i) => (
                  <tr key={i} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                    <td className="py-2 px-3 font-medium">{signal.symbol}</td>
                    <td className="py-2 px-3">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        signal.direction === 'BUY'
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {signal.direction}
                      </span>
                    </td>
                    <td className="py-2 px-3">{(signal.confidence * 100).toFixed(0)}%</td>
                    <td className="py-2 px-3">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        signal.status === 'tp_hit' ? 'bg-green-500/20 text-green-400' :
                        signal.status === 'sl_hit' ? 'bg-red-500/20 text-red-400' :
                        signal.status === 'pending' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {signal.status}
                      </span>
                    </td>
                    <td className={`py-2 px-3 font-medium ${
                      signal.pnl_pips > 0 ? 'text-green-400' :
                      signal.pnl_pips < 0 ? 'text-red-400' : ''
                    }`}>
                      {signal.pnl_pips ? `${formatPnl(signal.pnl_pips)} pips` : '-'}
                    </td>
                    <td className="py-2 px-3 text-gray-400 text-xs">
                      {signal.kill_zone || '-'}
                    </td>
                    <td className="py-2 px-3 text-gray-400 text-xs">
                      {new Date(signal.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <BarChart3 size={48} className="mx-auto mb-4 opacity-50" />
            <p>No signals recorded yet</p>
            <p className="text-sm mt-2">Generate some signals to see performance data</p>
          </div>
        )}
      </div>

      {/* Info Box */}
      <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-6">
        <h2 className="text-lg font-bold text-blue-400 mb-3">About Performance Tracking</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <p className="font-medium text-blue-300 mb-1">How it works:</p>
            <ul className="space-y-1 text-gray-400">
              <li>- Signals are automatically recorded when generated</li>
              <li>- Update outcomes manually or via API</li>
              <li>- AI analyzes patterns in winning/losing trades</li>
            </ul>
          </div>
          <div>
            <p className="font-medium text-blue-300 mb-1">Improvement loop:</p>
            <ul className="space-y-1 text-gray-400">
              <li>- Concepts with high win rates get increased weight</li>
              <li>- Low-performing patterns are flagged</li>
              <li>- Kill zone performance is tracked</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
