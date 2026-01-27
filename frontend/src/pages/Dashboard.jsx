import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Brain,
  FileText,
  BarChart3,
  RefreshCw,
  AlertCircle
} from 'lucide-react';
import { getAnalysisStatus, quickSignal, getSymbols } from '../services/api';

// Signal Card Component
function SignalCard({ symbol, data, loading, onRefresh }) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 animate-pulse">
        <div className="h-6 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="h-10 bg-gray-700 rounded w-1/2"></div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-xl font-bold">{symbol}</h3>
          <button onClick={onRefresh} className="text-gray-400 hover:text-white">
            <RefreshCw size={16} />
          </button>
        </div>
        <p className="text-gray-500">Click refresh to load</p>
      </div>
    );
  }

  const isBullish = data.bias === 'bullish';
  const isBearish = data.bias === 'bearish';

  return (
    <div className={`bg-gray-800 rounded-xl p-6 border ${
      isBullish ? 'border-green-500/50' : isBearish ? 'border-red-500/50' : 'border-gray-700'
    }`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-bold">{symbol}</h3>
        <button onClick={onRefresh} className="text-gray-400 hover:text-white">
          <RefreshCw size={16} />
        </button>
      </div>

      <div className="flex items-center space-x-3 mb-4">
        {isBullish ? (
          <div className="flex items-center text-green-500">
            <TrendingUp size={24} className="mr-2" />
            <span className="text-2xl font-bold">BULLISH</span>
          </div>
        ) : isBearish ? (
          <div className="flex items-center text-red-500">
            <TrendingDown size={24} className="mr-2" />
            <span className="text-2xl font-bold">BEARISH</span>
          </div>
        ) : (
          <div className="flex items-center text-gray-400">
            <Activity size={24} className="mr-2" />
            <span className="text-2xl font-bold">NEUTRAL</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-400">Zone</span>
          <p className={`font-medium ${
            data.zone === 'discount' ? 'text-green-400' :
            data.zone === 'premium' ? 'text-red-400' : 'text-gray-300'
          }`}>
            {data.zone?.toUpperCase() || 'N/A'}
          </p>
        </div>
        <div>
          <span className="text-gray-400">Price</span>
          <p className="font-medium text-white">
            {data.current_price?.toFixed(5) || 'N/A'}
          </p>
        </div>
      </div>

      {data.concepts?.length > 0 && (
        <div className="mt-4">
          <span className="text-gray-400 text-sm">ICT Concepts</span>
          <div className="flex flex-wrap gap-1 mt-1">
            {data.concepts.map((concept, i) => (
              <span
                key={i}
                className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-300"
              >
                {concept.replace('_', ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      <p className="mt-4 text-xs text-gray-500">{data.summary}</p>
    </div>
  );
}

// Stats Card
function StatCard({ icon: Icon, label, value, subValue, color = 'blue' }) {
  const colors = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    purple: 'from-purple-500 to-purple-600',
    orange: 'from-orange-500 to-orange-600',
  };

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-sm">{label}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
          {subValue && <p className="text-sm text-gray-500 mt-1">{subValue}</p>}
        </div>
        <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${colors[color]} flex items-center justify-center`}>
          <Icon size={24} />
        </div>
      </div>
    </div>
  );
}

// Main Dashboard
export default function Dashboard() {
  const [status, setStatus] = useState(null);
  const [signals, setSignals] = useState({});
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);

  const symbols = ['EURUSD', 'XAUUSD', 'US30', 'GBPUSD'];

  useEffect(() => {
    loadStatus();
    symbols.forEach(symbol => loadSignal(symbol));
  }, []);

  const loadStatus = async () => {
    try {
      const data = await getAnalysisStatus();
      setStatus(data);
    } catch (err) {
      console.error('Failed to load status:', err);
    }
  };

  const loadSignal = async (symbol) => {
    setLoading(prev => ({ ...prev, [symbol]: true }));
    try {
      const data = await quickSignal(symbol);
      setSignals(prev => ({ ...prev, [symbol]: data }));
    } catch (err) {
      console.error(`Failed to load ${symbol}:`, err);
    } finally {
      setLoading(prev => ({ ...prev, [symbol]: false }));
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-gray-400">ICT AI Trading Signal Overview</p>
        </div>
        <button
          onClick={() => {
            loadStatus();
            symbols.forEach(s => loadSignal(s));
          }}
          className="flex items-center space-x-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <RefreshCw size={16} />
          <span>Refresh All</span>
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={FileText}
          label="Transcripts"
          value={status?.transcripts_ready || 0}
          subValue={`of ${status?.total_videos || 0} videos`}
          color="blue"
        />
        <StatCard
          icon={Brain}
          label="ML Status"
          value={status?.ml_trained ? 'Trained' : 'Not Trained'}
          subValue="Self-improving"
          color="purple"
        />
        <StatCard
          icon={BarChart3}
          label="Signals Generated"
          value={status?.signals_generated || 0}
          subValue="Lifetime"
          color="green"
        />
        <StatCard
          icon={Activity}
          label="System Status"
          value={status?.status === 'operational' ? 'Online' : 'Offline'}
          subValue="All systems"
          color="orange"
        />
      </div>

      {/* Signal Cards */}
      <div>
        <h2 className="text-xl font-bold mb-4">Quick Signals</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {symbols.map(symbol => (
            <SignalCard
              key={symbol}
              symbol={symbol}
              data={signals[symbol]}
              loading={loading[symbol]}
              onRefresh={() => loadSignal(symbol)}
            />
          ))}
        </div>
      </div>

      {/* Info Banner */}
      <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4 flex items-start space-x-3">
        <AlertCircle className="text-blue-400 flex-shrink-0 mt-0.5" size={20} />
        <div>
          <h3 className="font-medium text-blue-400">Learning in Progress</h3>
          <p className="text-sm text-gray-400 mt-1">
            The AI is continuously learning from ICT video transcripts.
            As more videos are processed, signal accuracy will improve.
            Currently trained on {status?.transcripts_ready || 0} transcripts.
          </p>
        </div>
      </div>
    </div>
  );
}
