import React, { useState, useEffect, useRef } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Brain,
  FileText,
  BarChart3,
  RefreshCw,
  AlertCircle,
  Zap,
  Target,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  Sparkles,
  ChevronRight,
  Layers,
  Plus,
  Youtube,
  CheckCircle,
  XCircle,
  Loader2,
  Play
} from 'lucide-react';
import { getAnalysisStatus, quickSignal, getSymbols, addPlaylist, getPlaylistStreamUrl } from '../services/api';

// Animated Background Orb
function BackgroundOrb({ className }) {
  return (
    <div className={`absolute rounded-full blur-3xl opacity-20 animate-pulse ${className}`} />
  );
}

// Modern Signal Card Component
function SignalCard({ symbol, data, loading, onRefresh }) {
  if (loading) {
    return (
      <div className="glass-card p-6 animate-pulse">
        <div className="flex justify-between items-center mb-4">
          <div className="h-6 w-20 skeleton rounded" />
          <div className="h-8 w-8 skeleton rounded-lg" />
        </div>
        <div className="h-12 skeleton rounded-lg mb-4" />
        <div className="grid grid-cols-2 gap-3">
          <div className="h-16 skeleton rounded-lg" />
          <div className="h-16 skeleton rounded-lg" />
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="glass-card p-6 group cursor-pointer" onClick={onRefresh}>
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-bold text-white">{symbol}</h3>
          <button className="p-2 rounded-lg bg-white/5 text-slate-400 hover:text-white hover:bg-white/10 transition-all group-hover:rotate-180 duration-500">
            <RefreshCw size={16} />
          </button>
        </div>
        <div className="flex flex-col items-center justify-center py-8 text-slate-500">
          <Zap className="w-8 h-8 mb-2 opacity-50" />
          <p className="text-sm">Click to analyze</p>
        </div>
      </div>
    );
  }

  const isBullish = data.bias === 'bullish';
  const isBearish = data.bias === 'bearish';

  return (
    <div className={`glass-card p-6 relative overflow-hidden group ${
      isBullish ? 'hover:border-emerald-500/30' : isBearish ? 'hover:border-red-500/30' : ''
    }`}>
      {/* Gradient accent */}
      <div className={`absolute top-0 left-0 right-0 h-1 ${
        isBullish ? 'bg-gradient-to-r from-emerald-500 to-teal-500' :
        isBearish ? 'bg-gradient-to-r from-red-500 to-orange-500' :
        'bg-gradient-to-r from-slate-500 to-slate-600'
      }`} />

      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-bold text-white">{symbol}</h3>
          <span className={`text-xs px-2 py-0.5 rounded-full ${
            isBullish ? 'bg-emerald-500/20 text-emerald-400' :
            isBearish ? 'bg-red-500/20 text-red-400' :
            'bg-slate-500/20 text-slate-400'
          }`}>
            {data.zone?.toUpperCase() || 'EQUILIBRIUM'}
          </span>
        </div>
        <button
          onClick={onRefresh}
          className="p-2 rounded-lg bg-white/5 text-slate-400 hover:text-white hover:bg-white/10 transition-all hover:rotate-180 duration-500"
        >
          <RefreshCw size={14} />
        </button>
      </div>

      {/* Signal Direction */}
      <div className={`flex items-center gap-3 p-4 rounded-xl mb-4 ${
        isBullish ? 'bg-emerald-500/10 border border-emerald-500/20' :
        isBearish ? 'bg-red-500/10 border border-red-500/20' :
        'bg-slate-500/10 border border-slate-500/20'
      }`}>
        {isBullish ? (
          <>
            <div className="w-12 h-12 rounded-xl bg-emerald-500/20 flex items-center justify-center">
              <ArrowUpRight className="w-6 h-6 text-emerald-400" />
            </div>
            <div>
              <p className="text-xl font-bold text-emerald-400">BULLISH</p>
              <p className="text-xs text-slate-400">Long positions favored</p>
            </div>
          </>
        ) : isBearish ? (
          <>
            <div className="w-12 h-12 rounded-xl bg-red-500/20 flex items-center justify-center">
              <ArrowDownRight className="w-6 h-6 text-red-400" />
            </div>
            <div>
              <p className="text-xl font-bold text-red-400">BEARISH</p>
              <p className="text-xs text-slate-400">Short positions favored</p>
            </div>
          </>
        ) : (
          <>
            <div className="w-12 h-12 rounded-xl bg-slate-500/20 flex items-center justify-center">
              <Activity className="w-6 h-6 text-slate-400" />
            </div>
            <div>
              <p className="text-xl font-bold text-slate-400">NEUTRAL</p>
              <p className="text-xs text-slate-400">Wait for confirmation</p>
            </div>
          </>
        )}
      </div>

      {/* Price */}
      <div className="flex items-center justify-between px-1 mb-4">
        <span className="text-sm text-slate-400">Current Price</span>
        <span className="text-lg font-mono font-bold text-white">
          {data.current_price?.toFixed(5) || 'N/A'}
        </span>
      </div>

      {/* Concepts */}
      {data.concepts?.length > 0 && (
        <div className="pt-4 border-t border-white/5">
          <div className="flex flex-wrap gap-1.5">
            {data.concepts.slice(0, 4).map((concept, i) => (
              <span
                key={i}
                className="px-2.5 py-1 text-xs rounded-lg bg-indigo-500/10 text-indigo-300 border border-indigo-500/20"
              >
                {concept.replace('_', ' ')}
              </span>
            ))}
            {data.concepts.length > 4 && (
              <span className="px-2.5 py-1 text-xs rounded-lg bg-white/5 text-slate-400">
                +{data.concepts.length - 4}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Modern Stats Card
function StatCard({ icon: Icon, label, value, subValue, gradient, delay = 0 }) {
  return (
    <div
      className="stat-card group animate-slide-up"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm text-slate-400 mb-1">{label}</p>
          <p className="text-2xl font-bold text-white tracking-tight">{value}</p>
          {subValue && (
            <p className="text-xs text-slate-500 mt-1 flex items-center gap-1">
              <Sparkles className="w-3 h-3" />
              {subValue}
            </p>
          )}
        </div>
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center shadow-lg transform group-hover:scale-110 transition-transform duration-300`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
      </div>

      {/* Hover glow effect */}
      <div className={`absolute inset-0 rounded-lg bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-5 transition-opacity duration-300`} />
    </div>
  );
}

// Playlist Processing Card
function PlaylistProcessor({ onProcessingComplete }) {
  const [playlistUrl, setPlaylistUrl] = useState('');
  const [tier, setTier] = useState(3);
  const [trainAfter, setTrainAfter] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const eventSourceRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!playlistUrl.trim()) return;

    setError(null);
    setProcessing(true);
    setProgress({
      status: 'starting',
      message: 'Starting playlist processing...',
      progress: 0,
      total: 0
    });

    try {
      const result = await addPlaylist(playlistUrl, tier, trainAfter);

      if (result.status === 'exists') {
        setError('This playlist has already been added.');
        setProcessing(false);
        setProgress(null);
        return;
      }

      if (result.status === 'started' && result.job_id) {
        // Connect to SSE stream
        const streamUrl = getPlaylistStreamUrl(result.job_id);
        const eventSource = new EventSource(streamUrl);
        eventSourceRef.current = eventSource;

        eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          setProgress(data);

          if (data.status === 'completed' || data.status === 'error') {
            eventSource.close();
            setProcessing(false);
            if (data.status === 'completed') {
              setPlaylistUrl('');
              onProcessingComplete?.();
            }
          }
        };

        eventSource.onerror = () => {
          eventSource.close();
          setProcessing(false);
          setError('Connection lost. Check the status in the Learning page.');
        };
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to start processing');
      setProcessing(false);
      setProgress(null);
    }
  };

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const getProgressPercentage = () => {
    if (!progress || !progress.total) return 0;
    return Math.round((progress.progress / progress.total) * 100);
  };

  return (
    <div className="glass-card p-6 relative overflow-hidden">
      {/* Gradient accent */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-red-500 via-pink-500 to-purple-500" />

      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-red-500 to-pink-500 flex items-center justify-center shadow-lg">
          <Youtube className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="font-bold text-white text-lg">Add YouTube Playlist</h3>
          <p className="text-sm text-slate-400">Train ML with new Smart Money videos</p>
        </div>
      </div>

      {!processing ? (
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <input
              type="text"
              value={playlistUrl}
              onChange={(e) => setPlaylistUrl(e.target.value)}
              placeholder="Paste YouTube playlist URL..."
              className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/50 transition-all"
            />
          </div>

          <div className="flex gap-4">
            <div className="flex-1">
              <label className="text-xs text-slate-400 mb-1 block">Learning Tier</label>
              <select
                value={tier}
                onChange={(e) => setTier(Number(e.target.value))}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-indigo-500/50"
              >
                <option value={1} className="bg-[#0a0a0f]">Tier 1 - Core Concepts</option>
                <option value={2} className="bg-[#0a0a0f]">Tier 2 - Advanced</option>
                <option value={3} className="bg-[#0a0a0f]">Tier 3 - Additional</option>
              </select>
            </div>
            <div className="flex items-end">
              <label className="flex items-center gap-2 px-3 py-2 bg-white/5 border border-white/10 rounded-lg cursor-pointer hover:bg-white/10 transition-colors">
                <input
                  type="checkbox"
                  checked={trainAfter}
                  onChange={(e) => setTrainAfter(e.target.checked)}
                  className="w-4 h-4 rounded border-white/20 bg-white/5 text-indigo-500 focus:ring-indigo-500/50"
                />
                <span className="text-sm text-slate-300">Train ML after</span>
              </label>
            </div>
          </div>

          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2 text-red-400 text-sm">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={!playlistUrl.trim()}
            className="w-full btn btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus className="w-4 h-4" />
            <span>Add Playlist & Process</span>
          </button>
        </form>
      ) : (
        <div className="space-y-4">
          {/* Progress Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {progress?.status === 'completed' ? (
                <CheckCircle className="w-5 h-5 text-emerald-400" />
              ) : progress?.status === 'error' ? (
                <XCircle className="w-5 h-5 text-red-400" />
              ) : (
                <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
              )}
              <span className="text-sm font-medium text-white capitalize">
                {progress?.status?.replace('_', ' ') || 'Processing'}
              </span>
            </div>
            {progress?.total > 0 && (
              <span className="text-sm text-slate-400">
                {progress?.progress || 0} / {progress?.total}
              </span>
            )}
          </div>

          {/* Progress Bar */}
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-500 rounded-full ${
                progress?.status === 'completed' ? 'bg-emerald-500' :
                progress?.status === 'error' ? 'bg-red-500' :
                progress?.status === 'training' ? 'bg-purple-500 animate-pulse' :
                'bg-gradient-to-r from-indigo-500 to-purple-500'
              }`}
              style={{ width: `${progress?.status === 'training' ? 100 : getProgressPercentage()}%` }}
            />
          </div>

          {/* Status Message */}
          <p className="text-sm text-slate-400">
            {progress?.message || 'Processing...'}
          </p>

          {/* Playlist Title */}
          {progress?.playlist_title && (
            <div className="p-3 bg-white/5 rounded-lg">
              <p className="text-xs text-slate-500 mb-1">Playlist</p>
              <p className="text-sm text-white font-medium truncate">{progress.playlist_title}</p>
            </div>
          )}

          {/* Current Video */}
          {progress?.current_video && (
            <div className="p-3 bg-white/5 rounded-lg">
              <div className="flex items-center justify-between mb-1">
                <p className="text-xs text-slate-500">
                  Processing Video {progress.current_video.index}
                </p>
                {progress?.whisper_step && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 flex items-center gap-1">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    Whisper
                  </span>
                )}
              </div>
              <p className="text-sm text-white truncate">{progress.current_video.title}</p>
              {progress?.whisper_step && (
                <p className="text-xs text-amber-400 mt-1">
                  {progress.whisper_step === 'downloading_audio' && 'Downloading audio...'}
                  {progress.whisper_step === 'loading_model' && 'Loading Whisper model...'}
                  {progress.whisper_step === 'transcribing' && 'Transcribing with AI...'}
                  {progress.whisper_step === 'whisper_fallback' && 'No captions found, using Whisper...'}
                </p>
              )}
            </div>
          )}

          {/* Results Summary */}
          {progress?.status === 'completed' && (
            <div className="space-y-3 pt-2">
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-center">
                  <p className="text-lg font-bold text-emerald-400">{progress.completed?.length || 0}</p>
                  <p className="text-xs text-slate-400">Transcribed</p>
                </div>
                <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-center">
                  <p className="text-lg font-bold text-red-400">{progress.failed?.length || 0}</p>
                  <p className="text-xs text-slate-400">Failed</p>
                </div>
              </div>
              {/* Method breakdown */}
              {progress.completed?.length > 0 && (
                <div className="flex gap-2 text-xs">
                  {progress.completed.filter(v => v.method === 'youtube_transcript_api').length > 0 && (
                    <span className="px-2 py-1 bg-blue-500/10 text-blue-400 rounded-full">
                      {progress.completed.filter(v => v.method === 'youtube_transcript_api').length} via YouTube
                    </span>
                  )}
                  {progress.completed.filter(v => v.method === 'whisper').length > 0 && (
                    <span className="px-2 py-1 bg-amber-500/10 text-amber-400 rounded-full">
                      {progress.completed.filter(v => v.method === 'whisper').length} via Whisper
                    </span>
                  )}
                  {progress.completed.filter(v => v.status === 'skipped').length > 0 && (
                    <span className="px-2 py-1 bg-slate-500/10 text-slate-400 rounded-full">
                      {progress.completed.filter(v => v.status === 'skipped').length} skipped
                    </span>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Training Results */}
          {progress?.training_results && (
            <div className="p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg">
              <p className="text-xs text-purple-400 mb-2">ML Training Complete</p>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Transcripts:</span>
                <span className="text-white">{progress.training_results.n_transcripts}</span>
              </div>
              {progress.training_results.classifier_f1 && (
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">F1 Score:</span>
                  <span className="text-white">{(progress.training_results.classifier_f1 * 100).toFixed(1)}%</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Main Dashboard
export default function Dashboard() {
  const [status, setStatus] = useState(null);
  const [signals, setSignals] = useState({});
  const [loading, setLoading] = useState({});
  const [refreshing, setRefreshing] = useState(false);

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

  const refreshAll = async () => {
    setRefreshing(true);
    await loadStatus();
    await Promise.all(symbols.map(s => loadSignal(s)));
    setRefreshing(false);
  };

  return (
    <div className="space-y-8 relative">
      {/* Background decorations */}
      <BackgroundOrb className="w-96 h-96 bg-indigo-500 -top-48 -left-48" />
      <BackgroundOrb className="w-72 h-72 bg-purple-500 top-1/2 -right-36" />

      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">
            Dashboard
          </h1>
          <p className="text-slate-400 mt-1">
            Real-time Smart Money analysis overview
          </p>
        </div>
        <button
          onClick={refreshAll}
          disabled={refreshing}
          className="btn btn-primary"
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          <span>Refresh All</span>
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={FileText}
          label="Transcripts Processed"
          value={status?.transcripts_ready || 0}
          subValue={`of ${status?.total_videos || 0} videos`}
          gradient="from-blue-500 to-cyan-500"
          delay={0}
        />
        <StatCard
          icon={Brain}
          label="ML Model Status"
          value={status?.ml_trained ? 'Trained' : 'Training'}
          subValue="Self-improving AI"
          gradient="from-purple-500 to-pink-500"
          delay={50}
        />
        <StatCard
          icon={Target}
          label="Signals Generated"
          value={status?.signals_generated || 0}
          subValue="Lifetime total"
          gradient="from-emerald-500 to-teal-500"
          delay={100}
        />
        <StatCard
          icon={Activity}
          label="System Status"
          value={status?.status === 'operational' ? 'Online' : 'Offline'}
          subValue="All systems nominal"
          gradient="from-orange-500 to-amber-500"
          delay={150}
        />
      </div>

      {/* Two Column Layout - Playlist Processor & Info */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Playlist Processor */}
        <PlaylistProcessor onProcessingComplete={loadStatus} />

        {/* Info Banner */}
        <div className="glass-card-static p-6 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 rounded-full blur-2xl" />

          <div className="flex items-start gap-4 relative">
            <div className="w-12 h-12 rounded-xl bg-indigo-500/20 flex items-center justify-center flex-shrink-0">
              <Layers className="w-6 h-6 text-indigo-400" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-white mb-1">
                Continuous Learning Active
              </h3>
              <p className="text-sm text-slate-400 leading-relaxed">
                The AI is continuously analyzing Smart Money video transcripts to improve signal accuracy.
                Currently trained on <span className="text-indigo-400 font-medium">{status?.transcripts_ready || 0} transcripts</span> with
                <span className="text-emerald-400 font-medium"> {status?.total_videos || 0} videos</span> in the knowledge base.
              </p>
            </div>
          </div>

          {/* Progress bar */}
          <div className="mt-4 pt-4 border-t border-white/5">
            <div className="flex items-center justify-between text-sm mb-2">
              <span className="text-slate-400">Learning Progress</span>
              <span className="text-white font-medium">
                {status?.transcripts_ready && status?.total_videos
                  ? Math.round((status.transcripts_ready / status.total_videos) * 100)
                  : 0}%
              </span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${status?.transcripts_ready && status?.total_videos
                    ? (status.transcripts_ready / status.total_videos) * 100
                    : 0}%`
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Signal Cards Section */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-indigo-400" />
              Quick Market Analysis
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Click any card to refresh the signal
            </p>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <Clock className="w-4 h-4" />
            <span>Updates every 5 minutes</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {symbols.map((symbol, i) => (
            <div
              key={symbol}
              className="animate-slide-up"
              style={{ animationDelay: `${i * 50}ms` }}
            >
              <SignalCard
                symbol={symbol}
                data={signals[symbol]}
                loading={loading[symbol]}
                onRefresh={() => loadSignal(symbol)}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
