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
  Play,
  Trash2,
  FolderOpen,
  ChevronDown,
  GraduationCap,
  Eye,
  Image,
  Video,
  Scan,
  Shield
} from 'lucide-react';
import { getAnalysisStatus, quickSignal, getSymbols, addPlaylist, getPlaylistStreamUrl, whitewashML, getTranscriptsGrouped, trainFromPlaylist, getSelectiveTrainingStatus, trainWithVision, trainSingleVideoWithVision, getVisionTrainingStatus, getVisionCapabilities, getVisualKnowledge, getHedgeFundStatus, getEdgeStatistics } from '../services/api';

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
      const result = await addPlaylist(playlistUrl, 1, trainAfter);

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

          <div className="flex items-center">
            <label className="flex items-center gap-2 px-3 py-2 bg-white/5 border border-white/10 rounded-lg cursor-pointer hover:bg-white/10 transition-colors">
              <input
                type="checkbox"
                checked={trainAfter}
                onChange={(e) => setTrainAfter(e.target.checked)}
                className="w-4 h-4 rounded border-white/20 bg-white/5 text-indigo-500 focus:ring-indigo-500/50"
              />
              <span className="text-sm text-slate-300">Train ML after processing</span>
            </label>
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

// ML Training Manager Component
function MLTrainingManager({ onTrainingComplete }) {
  const [groupedData, setGroupedData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [whitewashing, setWhitewashing] = useState(false);
  const [selectedPlaylist, setSelectedPlaylist] = useState(null);
  const [expandedPlaylist, setExpandedPlaylist] = useState(null);
  const [training, setTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [error, setError] = useState(null);
  const eventSourceRef = useRef(null);

  useEffect(() => {
    loadGroupedTranscripts();
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const loadGroupedTranscripts = async () => {
    setLoading(true);
    try {
      const data = await getTranscriptsGrouped();
      setGroupedData(data);
    } catch (err) {
      console.error('Failed to load grouped transcripts:', err);
      setError('Failed to load transcripts');
    } finally {
      setLoading(false);
    }
  };

  const handleWhitewash = async () => {
    if (!window.confirm('Are you sure you want to WHITEWASH all ML training? This will delete all trained models but keep your transcripts.')) {
      return;
    }

    setWhitewashing(true);
    setError(null);

    try {
      await whitewashML();
      setTrainingProgress(null);
      setSelectedPlaylist(null);
      onTrainingComplete?.();
    } catch (err) {
      setError(err.response?.data?.detail || 'Whitewash failed');
    } finally {
      setWhitewashing(false);
    }
  };

  const handleTrainPlaylist = async (playlist) => {
    if (!playlist?.id) return;

    setError(null);
    setTraining(true);
    setSelectedPlaylist(playlist);
    setTrainingProgress({
      status: 'starting',
      message: 'Initializing training...',
      playlist_title: playlist.title
    });

    try {
      const result = await trainFromPlaylist(playlist.id);

      if (result.status === 'started' && result.job_id) {
        // Use polling instead of SSE for reliability
        const pollStatus = async () => {
          try {
            const status = await getSelectiveTrainingStatus(result.job_id);
            setTrainingProgress(status);

            if (status.status === 'completed' || status.status === 'error') {
              setTraining(false);
              if (status.status === 'completed') {
                onTrainingComplete?.();
              }
            } else {
              // Continue polling every 500ms
              setTimeout(pollStatus, 500);
            }
          } catch (err) {
            console.error('Polling error:', err);
            setTraining(false);
            setError('Failed to get training status');
          }
        };

        // Start polling
        pollStatus();
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Training failed to start');
      setTraining(false);
      setTrainingProgress(null);
    }
  };

  const togglePlaylistExpand = (playlistId) => {
    setExpandedPlaylist(expandedPlaylist === playlistId ? null : playlistId);
  };

  return (
    <div className="glass-card p-6 relative overflow-hidden">
      {/* Gradient accent */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-purple-500 via-indigo-500 to-blue-500" />

      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-500 flex items-center justify-center shadow-lg">
            <GraduationCap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="font-bold text-white text-lg">ML Training Manager</h3>
            <p className="text-sm text-slate-400">Train ML from specific playlists</p>
          </div>
        </div>

        {/* Whitewash Button */}
        <button
          onClick={handleWhitewash}
          disabled={whitewashing || training}
          className="px-4 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm font-medium hover:bg-red-500/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {whitewashing ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Trash2 className="w-4 h-4" />
          )}
          <span>Whitewash ML</span>
        </button>
      </div>

      {error && (
        <div className="p-3 mb-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4" />
          {error}
        </div>
      )}

      {/* Training Progress */}
      {trainingProgress && (
        <div className="mb-6 p-4 bg-purple-500/10 border border-purple-500/20 rounded-xl">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              {trainingProgress.status === 'completed' ? (
                <CheckCircle className="w-5 h-5 text-emerald-400" />
              ) : trainingProgress.status === 'error' ? (
                <XCircle className="w-5 h-5 text-red-400" />
              ) : (
                <Loader2 className="w-5 h-5 text-purple-400 animate-spin" />
              )}
              <span className="text-sm font-medium text-white capitalize">
                {trainingProgress.status?.replace('_', ' ')}
              </span>
            </div>
            <span className="text-xs text-slate-400">
              {trainingProgress.playlist_title}
            </span>
          </div>
          <p className="text-sm text-slate-300 mb-3">{trainingProgress.message}</p>

          {/* Progress bar with actual progress */}
          {(trainingProgress.status === 'loading' || trainingProgress.status === 'training') && (
            <div className="mb-3">
              <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                <span>
                  {trainingProgress.status === 'loading'
                    ? `Loading transcripts: ${trainingProgress.current_transcript || 0} / ${trainingProgress.total_transcripts}`
                    : trainingProgress.training_phase === 'extracting_concepts'
                      ? 'Extracting Smart Money concepts...'
                      : trainingProgress.training_phase === 'saving'
                        ? 'Saving models...'
                        : 'Training ML models...'}
                </span>
                {trainingProgress.status === 'loading' && (
                  <span>{Math.round((trainingProgress.current_transcript / trainingProgress.total_transcripts) * 100)}%</span>
                )}
              </div>
              <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    trainingProgress.status === 'loading'
                      ? 'bg-gradient-to-r from-blue-500 to-indigo-500'
                      : 'bg-gradient-to-r from-purple-500 to-indigo-500 animate-pulse'
                  }`}
                  style={{
                    width: trainingProgress.status === 'loading'
                      ? `${(trainingProgress.current_transcript / trainingProgress.total_transcripts) * 100}%`
                      : '100%'
                  }}
                />
              </div>
            </div>
          )}

          {/* Transcripts loaded list - scrollable */}
          {trainingProgress.transcripts_loaded && trainingProgress.transcripts_loaded.length > 0 && (
            <div className="mt-3 max-h-32 overflow-y-auto">
              <p className="text-xs text-slate-500 mb-2">Transcripts processed:</p>
              <div className="space-y-1">
                {trainingProgress.transcripts_loaded.map((t, idx) => (
                  <div
                    key={idx}
                    className={`flex items-center gap-2 text-xs py-1 px-2 rounded ${
                      t.status === 'loaded' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
                    }`}
                  >
                    {t.status === 'loaded' ? (
                      <CheckCircle className="w-3 h-3 flex-shrink-0" />
                    ) : (
                      <XCircle className="w-3 h-3 flex-shrink-0" />
                    )}
                    <span className="truncate">{t.title}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Results */}
          {trainingProgress.status === 'completed' && trainingProgress.results && (
            <div className="grid grid-cols-3 gap-3 mt-4">
              <div className="p-2 bg-white/5 rounded-lg text-center">
                <p className="text-lg font-bold text-white">{trainingProgress.results.n_transcripts}</p>
                <p className="text-xs text-slate-400">Transcripts</p>
              </div>
              <div className="p-2 bg-white/5 rounded-lg text-center">
                <p className="text-lg font-bold text-white">
                  {trainingProgress.results.classifier_f1 ? `${(trainingProgress.results.classifier_f1 * 100).toFixed(1)}%` : 'N/A'}
                </p>
                <p className="text-xs text-slate-400">F1 Score</p>
              </div>
              <div className="p-2 bg-white/5 rounded-lg text-center">
                <p className="text-lg font-bold text-white">{trainingProgress.results.concepts_defined || 'N/A'}</p>
                <p className="text-xs text-slate-400">Concepts</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Playlist List */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
        </div>
      ) : groupedData?.playlists?.length > 0 ? (
        <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
          <p className="text-xs text-slate-500 mb-3">
            Select a playlist to train ML on its transcripts ({groupedData.total} total transcripts)
          </p>
          {groupedData.playlists.map((playlist) => (
            <div key={playlist.id} className="glass-card-static rounded-lg overflow-hidden">
              {/* Playlist Header */}
              <div
                className="p-4 flex items-center justify-between cursor-pointer hover:bg-white/[0.05] transition-colors"
                onClick={() => togglePlaylistExpand(playlist.id)}
              >
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <div className="w-10 h-10 rounded-lg bg-indigo-500/20 flex items-center justify-center flex-shrink-0">
                    <FolderOpen className="w-5 h-5 text-indigo-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-white truncate">{playlist.title}</p>
                    <div className="flex items-center gap-2 text-xs text-slate-500">
                      <span>{playlist.transcript_count} transcripts</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleTrainPlaylist(playlist);
                    }}
                    disabled={training}
                    className="px-3 py-1.5 bg-indigo-500/20 border border-indigo-500/30 rounded-lg text-indigo-300 text-xs font-medium hover:bg-indigo-500/30 transition-colors disabled:opacity-50 flex items-center gap-1"
                  >
                    <Play className="w-3 h-3" />
                    Train
                  </button>
                  <ChevronDown className={`w-5 h-5 text-slate-400 transition-transform ${expandedPlaylist === playlist.id ? 'rotate-180' : ''}`} />
                </div>
              </div>

              {/* Expanded Transcripts */}
              {expandedPlaylist === playlist.id && (
                <div className="border-t border-white/5 p-3 bg-black/20 max-h-48 overflow-y-auto">
                  <p className="text-xs text-slate-500 mb-2">Transcripts in this playlist:</p>
                  <div className="space-y-1">
                    {playlist.transcripts.map((t, idx) => (
                      <div key={idx} className="flex items-center justify-between py-1.5 px-2 bg-white/[0.02] rounded text-xs">
                        <span className="text-slate-300 truncate flex-1">{t.title || t.video_id}</span>
                        <span className="text-slate-500 ml-2">{t.word_count?.toLocaleString()} words</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Ungrouped transcripts */}
          {groupedData.ungrouped?.length > 0 && (
            <div className="glass-card-static rounded-lg p-4 mt-4">
              <div className="flex items-center gap-2 mb-2">
                <FileText className="w-4 h-4 text-slate-400" />
                <span className="text-sm text-slate-400">Ungrouped Transcripts ({groupedData.ungrouped.length})</span>
              </div>
              <p className="text-xs text-slate-500">
                These transcripts are not associated with any playlist.
              </p>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="w-16 h-16 rounded-2xl bg-slate-500/10 flex items-center justify-center mx-auto mb-4">
            <FolderOpen className="w-8 h-8 text-slate-500" />
          </div>
          <h3 className="text-lg font-semibold text-slate-400 mb-2">No playlists with transcripts</h3>
          <p className="text-sm text-slate-500">
            Add a YouTube playlist above to get started
          </p>
        </div>
      )}
    </div>
  );
}

// Vision Training Manager Component
function VisionTrainingManager({ onTrainingComplete }) {
  const [groupedData, setGroupedData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [visionCapabilities, setVisionCapabilities] = useState(null);
  const [visualKnowledge, setVisualKnowledge] = useState(null);
  const [selectedPlaylist, setSelectedPlaylist] = useState(null);
  const [expandedPlaylist, setExpandedPlaylist] = useState(null);
  const [training, setTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [visionProvider, setVisionProvider] = useState('local');
  const [extractionMode, setExtractionMode] = useState('comprehensive');
  const [maxFrames, setMaxFrames] = useState(0); // 0 = no limit
  const [error, setError] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [grouped, capabilities, knowledge] = await Promise.all([
        getTranscriptsGrouped(),
        getVisionCapabilities().catch(() => ({ vision_available: false })),
        getVisualKnowledge().catch(() => ({ has_vision_knowledge: false }))
      ]);
      setGroupedData(grouped);
      setVisionCapabilities(capabilities);
      setVisualKnowledge(knowledge);
    } catch (err) {
      console.error('Failed to load data:', err);
      setError('Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const handleTrainWithVision = async (playlist) => {
    if (!playlist?.id) return;

    setError(null);
    setTraining(true);
    setSelectedPlaylist(playlist);
    setTrainingProgress({
      status: 'starting',
      message: 'Initializing vision training...',
      playlist_title: playlist.title,
      frames_analyzed: 0,
      charts_detected: 0,
      patterns_found: 0
    });

    try {
      const result = await trainWithVision(playlist.id, visionProvider, maxFrames, extractionMode);

      if (result.status === 'started' && result.job_id) {
        // Poll for status
        const pollStatus = async () => {
          try {
            const status = await getVisionTrainingStatus(result.job_id);
            setTrainingProgress(status);

            if (status.status === 'completed' || status.status === 'error') {
              setTraining(false);
              if (status.status === 'completed') {
                // Refresh visual knowledge
                const knowledge = await getVisualKnowledge().catch(() => null);
                if (knowledge) setVisualKnowledge(knowledge);
                onTrainingComplete?.();
              }
            } else {
              setTimeout(pollStatus, 1000); // Poll every second for vision training
            }
          } catch (err) {
            console.error('Polling error:', err);
            setTraining(false);
            setError('Failed to get training status');
          }
        };

        pollStatus();
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Vision training failed to start');
      setTraining(false);
      setTrainingProgress(null);
    }
  };

  // Train vision on a SINGLE video
  const handleTrainSingleVideo = async (video, playlistTitle) => {
    if (!video?.video_id) return;

    setError(null);
    setTraining(true);
    setTrainingProgress({
      status: 'starting',
      message: `Initializing vision training for: ${video.title || video.video_id}`,
      playlist_title: playlistTitle,
      video_title: video.title,
      frames_analyzed: 0,
      charts_detected: 0,
      patterns_found: 0
    });

    try {
      const result = await trainSingleVideoWithVision(video.video_id, visionProvider, maxFrames, extractionMode);

      if (result.status === 'started' && result.job_id) {
        // Poll for status
        const pollStatus = async () => {
          try {
            const status = await getVisionTrainingStatus(result.job_id);
            setTrainingProgress({
              ...status,
              video_title: video.title
            });

            if (status.status === 'completed' || status.status === 'error') {
              setTraining(false);
              if (status.status === 'completed') {
                // Refresh visual knowledge
                const knowledge = await getVisualKnowledge().catch(() => null);
                if (knowledge) setVisualKnowledge(knowledge);
                onTrainingComplete?.();
              }
            } else {
              setTimeout(pollStatus, 1000);
            }
          } catch (err) {
            console.error('Polling error:', err);
            setTraining(false);
            setError('Failed to get training status');
          }
        };

        pollStatus();
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Vision training failed to start');
      setTraining(false);
      setTrainingProgress(null);
    }
  };

  const togglePlaylistExpand = (playlistId) => {
    setExpandedPlaylist(expandedPlaylist === playlistId ? null : playlistId);
  };

  if (!visionCapabilities?.vision_available) {
    return (
      <div className="glass-card p-6 relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500" />

        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center shadow-lg">
            <Eye className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="font-bold text-white text-lg">Vision Training</h3>
            <p className="text-sm text-slate-400">Multimodal video analysis</p>
          </div>
        </div>

        <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-xl">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-amber-300 font-medium">Vision Training Not Available</p>
              <p className="text-sm text-slate-400 mt-1">
                Vision training requires additional dependencies and API keys.
                Set <code className="text-cyan-400">ANTHROPIC_API_KEY</code> or <code className="text-cyan-400">OPENAI_API_KEY</code> environment variable.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card p-6 relative overflow-hidden">
      {/* Gradient accent */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500" />

      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center shadow-lg">
            <Eye className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="font-bold text-white text-lg">Vision Training</h3>
            <p className="text-sm text-slate-400">Analyze video frames with AI vision</p>
          </div>
        </div>

        {/* Vision Knowledge Badge */}
        {visualKnowledge?.has_vision_knowledge && (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
            <Scan className="w-4 h-4 text-cyan-400" />
            <span className="text-xs text-cyan-300">
              {visualKnowledge.patterns_learned || 0} patterns learned
            </span>
          </div>
        )}
      </div>

      {/* Vision Knowledge Summary */}
      {visualKnowledge?.has_vision_knowledge && (
        <div className="mb-6 p-4 bg-cyan-500/10 border border-cyan-500/20 rounded-xl">
          <div className="flex items-center gap-2 mb-3">
            <Image className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-medium text-white">Visual Knowledge Summary</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-2 bg-white/5 rounded-lg text-center">
              <p className="text-lg font-bold text-cyan-400">{visualKnowledge.videos_with_vision || 0}</p>
              <p className="text-xs text-slate-400">Videos Analyzed</p>
            </div>
            <div className="p-2 bg-white/5 rounded-lg text-center">
              <p className="text-lg font-bold text-blue-400">{visualKnowledge.visual_concepts || 0}</p>
              <p className="text-xs text-slate-400">Visual Concepts</p>
            </div>
            <div className="p-2 bg-white/5 rounded-lg text-center">
              <p className="text-lg font-bold text-indigo-400">{visualKnowledge.patterns_learned || 0}</p>
              <p className="text-xs text-slate-400">Patterns Learned</p>
            </div>
            <div className="p-2 bg-white/5 rounded-lg text-center">
              <p className="text-lg font-bold text-purple-400">{visualKnowledge.key_teaching_moments_count || 0}</p>
              <p className="text-xs text-slate-400">Teaching Moments</p>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="p-3 mb-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4" />
          {error}
        </div>
      )}

      {/* Training Options */}
      <div className="mb-6 p-4 bg-white/5 border border-white/10 rounded-xl">
        <p className="text-sm text-slate-400 mb-3">Training Configuration</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-slate-500 mb-1">Vision Provider</label>
            <select
              value={visionProvider}
              onChange={(e) => setVisionProvider(e.target.value)}
              disabled={training}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:border-cyan-500/50"
            >
              <option value="local">🍎 Local (FREE - Apple Silicon)</option>
              <option value="anthropic">Claude (Anthropic - Paid)</option>
              <option value="openai">GPT-4V (OpenAI - Paid)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1">Learning Mode</label>
            <select
              value={extractionMode}
              onChange={(e) => setExtractionMode(e.target.value)}
              disabled={training}
              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:border-cyan-500/50"
            >
              <option value="comprehensive">Comprehensive - Learn Everything (3s intervals)</option>
              <option value="thorough">Thorough - Good Coverage (5s intervals)</option>
              <option value="balanced">Balanced - Moderate (10-15s intervals)</option>
              <option value="selective">Selective - Key Moments Only (fastest)</option>
            </select>
          </div>
        </div>

        {/* Provider Description */}
        {visionProvider === 'local' && (
          <div className="mt-3 p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
            <div className="flex items-start gap-2">
              <span className="text-lg">🆓</span>
              <div className="text-xs text-emerald-300">
                <strong>100% FREE!</strong> Uses Ollama + LLaVA locally on your Mac. No API costs, no data sent to cloud.<br/>
                <span className="text-emerald-400/70">Setup: Install Ollama from ollama.ai, then run: <code className="bg-black/30 px-1 rounded">ollama pull llava</code></span>
              </div>
            </div>
          </div>
        )}
        {visionProvider !== 'local' && (
          <div className="mt-3 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
            <div className="flex items-start gap-2">
              <span className="text-lg">💰</span>
              <div className="text-xs text-amber-300">
                <strong>Paid API:</strong> Uses cloud API which costs ~$0.01-0.03 per frame analyzed. Comprehensive mode on a 10-min video = ~$2-6.
              </div>
            </div>
          </div>
        )}

        {/* Learning Mode Description */}
        <div className="mt-3 p-3 bg-white/[0.02] rounded-lg">
          <div className="flex items-start gap-2">
            <GraduationCap className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
            <div className="text-xs text-slate-400">
              {extractionMode === 'comprehensive' && (
                <span><strong className="text-cyan-300">Dedicated Student Mode:</strong> Extracts a frame every 3 seconds to capture EVERYTHING shown in the video. The AI will learn every visual detail just like a student watching attentively.</span>
              )}
              {extractionMode === 'thorough' && (
                <span><strong className="text-blue-300">Thorough Mode:</strong> Extracts frames every 5 seconds with extra focus on key teaching moments. Good balance between coverage and efficiency.</span>
              )}
              {extractionMode === 'balanced' && (
                <span><strong className="text-indigo-300">Balanced Mode:</strong> Extracts frames every 10-15 seconds with keyword-triggered boosts. Efficient but may miss some visual details.</span>
              )}
              {extractionMode === 'selective' && (
                <span><strong className="text-purple-300">Selective Mode:</strong> Only extracts frames when demonstrative language is used ("this", "here", "look at"). Fastest but may miss important visual content shown without verbal cues.</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Training Progress */}
      {trainingProgress && (
        <div className="mb-6 p-4 bg-cyan-500/10 border border-cyan-500/20 rounded-xl">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              {trainingProgress.status === 'completed' ? (
                <CheckCircle className="w-5 h-5 text-emerald-400" />
              ) : trainingProgress.status === 'error' ? (
                <XCircle className="w-5 h-5 text-red-400" />
              ) : (
                <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
              )}
              <span className="text-sm font-medium text-white capitalize">
                {trainingProgress.status?.replace('_', ' ')}
              </span>
            </div>
            <span className="text-xs text-slate-400">
              {trainingProgress.playlist_title}
            </span>
          </div>

          <p className="text-sm text-slate-300 mb-3">{trainingProgress.message}</p>

          {/* Vision-specific progress */}
          {(trainingProgress.status === 'training' || trainingProgress.status === 'loading') && (
            <div className="mb-3">
              <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                <span>Analyzing videos...</span>
                {trainingProgress.total > 0 && (
                  <span>{trainingProgress.progress} / {trainingProgress.total}</span>
                )}
              </div>
              <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-300 bg-gradient-to-r from-cyan-500 to-blue-500"
                  style={{
                    width: trainingProgress.total > 0
                      ? `${(trainingProgress.progress / trainingProgress.total) * 100}%`
                      : '0%'
                  }}
                />
              </div>
            </div>
          )}

          {/* Vision stats during training */}
          <div className="grid grid-cols-3 gap-2 mt-3">
            <div className="p-2 bg-white/5 rounded-lg text-center">
              <div className="flex items-center justify-center gap-1">
                <Video className="w-3 h-3 text-cyan-400" />
                <p className="text-sm font-bold text-white">{trainingProgress.frames_analyzed || 0}</p>
              </div>
              <p className="text-xs text-slate-400">Frames</p>
            </div>
            <div className="p-2 bg-white/5 rounded-lg text-center">
              <div className="flex items-center justify-center gap-1">
                <BarChart3 className="w-3 h-3 text-blue-400" />
                <p className="text-sm font-bold text-white">{trainingProgress.charts_detected || 0}</p>
              </div>
              <p className="text-xs text-slate-400">Charts</p>
            </div>
            <div className="p-2 bg-white/5 rounded-lg text-center">
              <div className="flex items-center justify-center gap-1">
                <Scan className="w-3 h-3 text-indigo-400" />
                <p className="text-sm font-bold text-white">{trainingProgress.patterns_found || 0}</p>
              </div>
              <p className="text-xs text-slate-400">Patterns</p>
            </div>
          </div>

          {/* Current video being analyzed */}
          {trainingProgress.current_video && (
            <div className="mt-3 p-2 bg-white/5 rounded-lg">
              <p className="text-xs text-slate-500 mb-1">Currently analyzing:</p>
              <p className="text-sm text-white truncate">{trainingProgress.current_video}</p>
            </div>
          )}

          {/* Results */}
          {trainingProgress.status === 'completed' && trainingProgress.results && (
            <div className="mt-4 pt-4 border-t border-white/10">
              <p className="text-xs text-slate-500 mb-2">Training Results</p>
              <div className="grid grid-cols-2 gap-2">
                <div className="p-2 bg-emerald-500/10 rounded-lg text-center">
                  <p className="text-lg font-bold text-emerald-400">
                    {trainingProgress.results.vision_analysis?.total_frames_analyzed || 0}
                  </p>
                  <p className="text-xs text-slate-400">Total Frames</p>
                </div>
                <div className="p-2 bg-blue-500/10 rounded-lg text-center">
                  <p className="text-lg font-bold text-blue-400">
                    {trainingProgress.results.vision_analysis?.chart_frames || 0}
                  </p>
                  <p className="text-xs text-slate-400">Chart Frames</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Playlist List */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
        </div>
      ) : groupedData?.playlists?.length > 0 ? (
        <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
          <p className="text-xs text-slate-500 mb-3">
            Select a playlist for vision training (analyzes video frames for visual patterns)
          </p>
          {groupedData.playlists.map((playlist) => (
            <div key={playlist.id} className="glass-card-static rounded-lg overflow-hidden">
              <div
                className="p-4 flex items-center justify-between cursor-pointer hover:bg-white/[0.05] transition-colors"
                onClick={() => togglePlaylistExpand(playlist.id)}
              >
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <div className="w-10 h-10 rounded-lg bg-cyan-500/20 flex items-center justify-center flex-shrink-0">
                    <Video className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-white truncate">{playlist.title}</p>
                    <div className="flex items-center gap-2 text-xs text-slate-500">
                      <span>{playlist.transcript_count} videos</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleTrainWithVision(playlist);
                    }}
                    disabled={training}
                    className="px-3 py-1.5 bg-cyan-500/20 border border-cyan-500/30 rounded-lg text-cyan-300 text-xs font-medium hover:bg-cyan-500/30 transition-colors disabled:opacity-50 flex items-center gap-1"
                  >
                    <Eye className="w-3 h-3" />
                    Vision Train
                  </button>
                  <ChevronDown className={`w-5 h-5 text-slate-400 transition-transform ${expandedPlaylist === playlist.id ? 'rotate-180' : ''}`} />
                </div>
              </div>

              {/* Expanded videos list */}
              {expandedPlaylist === playlist.id && (
                <div className="border-t border-white/5 p-3 bg-black/20 max-h-64 overflow-y-auto">
                  <p className="text-xs text-slate-500 mb-3">
                    Select a video to train vision on, or train entire playlist:
                  </p>
                  <div className="space-y-2">
                    {playlist.transcripts.map((video, idx) => (
                      <div
                        key={video.video_id || idx}
                        className="flex items-center justify-between p-2 bg-white/[0.03] hover:bg-white/[0.06] rounded-lg transition-colors group"
                      >
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          <div className="w-8 h-8 rounded bg-cyan-500/10 flex items-center justify-center flex-shrink-0">
                            <Video className="w-4 h-4 text-cyan-400/70" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm text-white truncate">{video.title || video.video_id}</p>
                            <p className="text-xs text-slate-500">{video.word_count?.toLocaleString()} words in transcript</p>
                          </div>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleTrainSingleVideo(video, playlist.title);
                          }}
                          disabled={training}
                          className="px-2 py-1 bg-cyan-500/10 border border-cyan-500/20 rounded text-cyan-300 text-xs font-medium hover:bg-cyan-500/20 transition-colors disabled:opacity-50 flex items-center gap-1 opacity-70 group-hover:opacity-100"
                        >
                          <Eye className="w-3 h-3" />
                          Train
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="w-16 h-16 rounded-2xl bg-slate-500/10 flex items-center justify-center mx-auto mb-4">
            <Video className="w-8 h-8 text-slate-500" />
          </div>
          <h3 className="text-lg font-semibold text-slate-400 mb-2">No playlists available</h3>
          <p className="text-sm text-slate-500">
            Add a YouTube playlist above to enable vision training
          </p>
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
  const [hedgeFundStatus, setHedgeFundStatus] = useState(null);
  const [edgeStats, setEdgeStats] = useState(null);

  const symbols = ['EURUSD', 'XAUUSD', 'US30', 'GBPUSD'];

  useEffect(() => {
    loadStatus();
    loadHedgeFundStatus();
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

  const loadHedgeFundStatus = async () => {
    try {
      const [hfStatus, edgeData] = await Promise.all([
        getHedgeFundStatus(),
        getEdgeStatistics()
      ]);
      setHedgeFundStatus(hfStatus);
      setEdgeStats(edgeData);
    } catch (err) {
      console.error('Failed to load hedge fund status:', err);
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
    await loadHedgeFundStatus();
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
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
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
          icon={Shield}
          label="Hedge Fund"
          value={hedgeFundStatus?.available ? 'Active' : 'Inactive'}
          subValue={`${Object.keys(edgeStats?.all_patterns || {}).length} patterns tracked`}
          gradient="from-indigo-500 to-violet-500"
          delay={150}
        />
        <StatCard
          icon={Activity}
          label="System Status"
          value={status?.status === 'operational' ? 'Online' : 'Offline'}
          subValue="All systems nominal"
          gradient="from-orange-500 to-amber-500"
          delay={200}
        />
      </div>

      {/* Two Column Layout - Playlist Processor & Info */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Playlist Processor */}
        <PlaylistProcessor onProcessingComplete={loadStatus} />

        {/* Info Banner - Shows different content based on ML training status */}
        <div className={`glass-card-static p-6 relative overflow-hidden ${!status?.ml_trained ? 'border-amber-500/30' : ''}`}>
          <div className={`absolute top-0 right-0 w-32 h-32 ${status?.ml_trained ? 'bg-indigo-500/10' : 'bg-amber-500/10'} rounded-full blur-2xl`} />

          {status?.ml_trained ? (
            <>
              <div className="flex items-start gap-4 relative">
                <div className="w-12 h-12 rounded-xl bg-indigo-500/20 flex items-center justify-center flex-shrink-0">
                  <Layers className="w-6 h-6 text-indigo-400" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-white mb-1">
                    ML Model Trained
                  </h3>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    The AI has been trained on Smart Money video transcripts and is ready for signal generation.
                    Trained on <span className="text-indigo-400 font-medium">{status?.n_transcripts_trained || 0} transcripts</span> out of
                    <span className="text-emerald-400 font-medium"> {status?.transcripts_ready || 0} available</span>.
                  </p>
                </div>
              </div>

              {/* Progress bar - Shows trained vs available */}
              <div className="mt-4 pt-4 border-t border-white/5">
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-slate-400">Training Coverage</span>
                  <span className="text-white font-medium">
                    {status?.n_transcripts_trained || 0} / {status?.transcripts_ready || 0} transcripts
                  </span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${status?.n_transcripts_trained && status?.transcripts_ready
                        ? (status.n_transcripts_trained / status.transcripts_ready) * 100
                        : 0}%`
                    }}
                  />
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-start gap-4 relative">
                <div className="w-12 h-12 rounded-xl bg-amber-500/20 flex items-center justify-center flex-shrink-0">
                  <AlertCircle className="w-6 h-6 text-amber-400" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-amber-400 mb-1">
                    ML Not Trained
                  </h3>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    The AI model needs to be trained before generating signals. Use the ML Training Manager below to train from a playlist.
                    {status?.transcripts_ready > 0 && (
                      <span className="block mt-1">
                        <span className="text-slate-500">{status?.transcripts_ready} transcripts available</span> - ready for training.
                      </span>
                    )}
                  </p>
                </div>
              </div>

              {/* No progress bar when not trained - show prompt instead */}
              <div className="mt-4 pt-4 border-t border-white/5">
                <div className="flex items-center gap-2 text-sm text-amber-400/80">
                  <Brain className="w-4 h-4" />
                  <span>Select a playlist below and click "Train ML" to get started</span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* ML Training Manager - Full Width */}
      <MLTrainingManager onTrainingComplete={loadStatus} />

      {/* Vision Training Manager - Full Width */}
      <VisionTrainingManager onTrainingComplete={loadStatus} />

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
