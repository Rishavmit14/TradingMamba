import React, { useState, useEffect } from 'react';
import {
  Brain,
  Database,
  TrendingUp,
  RefreshCw,
  Play,
  CheckCircle,
  AlertCircle,
  Clock,
  FileText,
  Zap,
  Sparkles,
  Target,
  ArrowRight,
  Layers,
  Eye,
  Image,
  Video,
  BarChart3,
  Box
} from 'lucide-react';
import { getMLStatus, triggerTraining, getTranscripts, getVisualKnowledge, getVisionCapabilities } from '../services/api';

// Background Orb Component
function BackgroundOrb({ className }) {
  return (
    <div className={`absolute rounded-full blur-3xl opacity-20 animate-pulse ${className}`} />
  );
}

// Animated Progress Ring
function ProgressRing({ value, max, size = 120, strokeWidth = 8, color }) {
  const percentage = max > 0 ? (value / max) * 100 : 0;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percentage / 100) * circumference;

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
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-white">{value}</span>
        <span className="text-xs text-slate-400">of {max}</span>
      </div>
    </div>
  );
}

// Stat Card
function StatCard({ icon: Icon, label, value, description, gradient, delay = 0 }) {
  return (
    <div
      className="glass-card p-6 group animate-slide-up"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-start justify-between mb-4">
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center shadow-lg transform group-hover:scale-110 transition-transform duration-300`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
      </div>
      <p className="text-sm text-slate-400">{label}</p>
      <p className="text-2xl font-bold text-white mt-1">{value}</p>
      {description && (
        <p className="text-xs text-slate-500 mt-2 flex items-center gap-1">
          <Sparkles className="w-3 h-3" />
          {description}
        </p>
      )}
    </div>
  );
}

// Pattern info for display
const PATTERN_INFO = {
  'fvg': { name: 'Fair Value Gap', short: 'FVG', color: '#ffd700', icon: Box },
  'order_block': { name: 'Order Block', short: 'OB', color: '#26a69a', icon: Layers },
  'breaker_block': { name: 'Breaker Block', short: 'BB', color: '#ff6b6b', icon: Target },
  'market_structure': { name: 'Market Structure', short: 'BOS/CHoCH', color: '#4fc3f7', icon: TrendingUp },
  'support_resistance': { name: 'Support/Resistance', short: 'S/R', color: '#9c27b0', icon: BarChart3 },
  'liquidity': { name: 'Liquidity', short: 'LIQ', color: '#e91e63', icon: Zap },
};

// Main Learning Page
export default function Learning() {
  const [mlStatus, setMLStatus] = useState(null);
  const [transcripts, setTranscripts] = useState([]);
  const [visualKnowledge, setVisualKnowledge] = useState(null);
  const [visionCapabilities, setVisionCapabilities] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [status, transcriptsData, vision, capabilities] = await Promise.all([
        getMLStatus().catch(() => null),
        getTranscripts().catch(() => ({ transcripts: [] })),
        getVisualKnowledge().catch(() => null),
        getVisionCapabilities().catch(() => null),
      ]);
      setMLStatus(status);
      setTranscripts(transcriptsData.transcripts || []);
      setVisualKnowledge(vision);
      setVisionCapabilities(capabilities);
    } catch (err) {
      console.error('Failed to load data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTraining = async () => {
    setTraining(true);
    setTrainingResult(null);
    try {
      const result = await triggerTraining(true);
      setTrainingResult(result);
      loadData();
    } catch (err) {
      setTrainingResult({ error: err.message || 'Training failed' });
    } finally {
      setTraining(false);
    }
  };

  const totalWords = transcripts.reduce((sum, t) => sum + (t.word_count || 0), 0);

  return (
    <div className="space-y-8 relative">
      {/* Background decorations */}
      <BackgroundOrb className="w-96 h-96 bg-purple-500 -top-48 -left-48" />
      <BackgroundOrb className="w-72 h-72 bg-blue-500 bottom-0 -right-36" />

      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3">
            <Brain className="w-8 h-8 text-indigo-400" />
            ML Learning Status
          </h1>
          <p className="text-slate-400 mt-1">
            Monitor and manage the AI learning process
          </p>
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

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={FileText}
          label="Transcripts Processed"
          value={mlStatus?.n_transcripts_processed || transcripts.length}
          description="Smart Money video transcripts"
          gradient="from-blue-500 to-cyan-500"
          delay={0}
        />
        <StatCard
          icon={Eye}
          label="Vision Training"
          value={visualKnowledge?.has_vision_knowledge ? `${visualKnowledge.patterns_learned || 0} patterns` : 'Not trained'}
          description={visualKnowledge?.has_vision_knowledge ? `${visualKnowledge.videos_with_vision || 0} videos analyzed` : 'Train from Dashboard'}
          gradient={visualKnowledge?.has_vision_knowledge ? 'from-cyan-500 to-blue-500' : 'from-orange-500 to-amber-500'}
          delay={50}
        />
        <StatCard
          icon={Database}
          label="Total Words"
          value={totalWords.toLocaleString()}
          description="Training corpus size"
          gradient="from-purple-500 to-pink-500"
          delay={100}
        />
        <StatCard
          icon={Brain}
          label="Model Status"
          value={mlStatus?.knowledge_base_loaded ? 'Trained' : 'Not Trained'}
          description={mlStatus?.classifier_performance?.trend || 'N/A'}
          gradient={mlStatus?.knowledge_base_loaded ? 'from-emerald-500 to-green-500' : 'from-orange-500 to-amber-500'}
          delay={150}
        />
      </div>

      {/* Vision Knowledge Section */}
      {visualKnowledge?.has_vision_knowledge && (
        <div className="glass-card p-6 relative overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500" />

          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center shadow-lg">
                <Eye className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Visual Pattern Knowledge</h2>
                <p className="text-sm text-slate-400">Patterns learned from video frame analysis</p>
              </div>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
              <Video className="w-4 h-4 text-cyan-400" />
              <span className="text-xs text-cyan-300">{visualKnowledge.videos_with_vision || 0} videos analyzed</span>
            </div>
          </div>

          {/* Patterns Learned Grid */}
          {visualKnowledge.pattern_details && visualKnowledge.pattern_details.length > 0 && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-slate-400 mb-3 flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                Patterns ML Has Learned ({visualKnowledge.pattern_details.length})
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {visualKnowledge.pattern_details.map((pattern) => {
                  const info = PATTERN_INFO[pattern.type] || {
                    name: pattern.type.replace('_', ' '),
                    short: pattern.type.toUpperCase(),
                    color: '#6b7280',
                    icon: Box
                  };
                  const IconComp = info.icon;
                  return (
                    <div
                      key={pattern.type}
                      className="p-4 rounded-xl bg-white/[0.03] border border-white/10 hover:border-white/20 transition-colors"
                    >
                      <div className="flex items-center gap-3 mb-3">
                        <div
                          className="w-10 h-10 rounded-lg flex items-center justify-center"
                          style={{ backgroundColor: info.color + '20' }}
                        >
                          <IconComp className="w-5 h-5" style={{ color: info.color }} />
                        </div>
                        <div>
                          <p className="font-semibold text-white">{info.name}</p>
                          <p className="text-xs text-slate-500">{info.short}</p>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-slate-400">Understanding</span>
                          <span className={`font-medium capitalize ${
                            pattern.understanding_level === 'expert' ? 'text-green-400' :
                            pattern.understanding_level === 'proficient' ? 'text-blue-400' :
                            pattern.understanding_level === 'intermediate' ? 'text-amber-400' :
                            'text-slate-300'
                          }`}>
                            {pattern.understanding_level || 'learning'}
                          </span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-slate-400">Confidence</span>
                          <span className="text-white font-medium">{Math.round(pattern.confidence * 100)}%</span>
                        </div>
                        <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.round(pattern.confidence * 100)}%`,
                              backgroundColor: info.color
                            }}
                          />
                        </div>
                        {pattern.can_detect_universally && (
                          <div className="flex items-center gap-1 text-xs text-green-400">
                            <span>✓</span>
                            <span>Can detect universally</span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Patterns Not Yet Learned */}
          {visualKnowledge.patterns_not_learned && visualKnowledge.patterns_not_learned.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-400 mb-3 flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-amber-400" />
                Patterns Not Yet Learned ({visualKnowledge.patterns_not_learned.length})
              </h3>
              <div className="flex flex-wrap gap-2">
                {visualKnowledge.patterns_not_learned.map((pattern) => {
                  const info = PATTERN_INFO[pattern] || { name: pattern.replace('_', ' '), short: pattern.toUpperCase() };
                  return (
                    <div
                      key={pattern}
                      className="px-3 py-2 rounded-lg bg-slate-800/50 border border-slate-700/50 text-slate-400 text-sm flex items-center gap-2"
                    >
                      <span>{info.name}</span>
                      <span className="text-xs text-slate-500 bg-slate-700/50 px-1.5 py-0.5 rounded">Train more</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Vision Stats Summary */}
          <div className="mt-6 pt-6 border-t border-white/10">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-cyan-400">{visualKnowledge.total_frames_analyzed || 0}</p>
                <p className="text-xs text-slate-400">Frames Analyzed</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-400">{visualKnowledge.chart_frames || 0}</p>
                <p className="text-xs text-slate-400">Chart Frames</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-indigo-400">{visualKnowledge.visual_concepts || 0}</p>
                <p className="text-xs text-slate-400">Visual Concepts</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-purple-400">{visualKnowledge.key_teaching_moments_count || 0}</p>
                <p className="text-xs text-slate-400">Teaching Moments</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* No Vision Knowledge Notice */}
      {!visualKnowledge?.has_vision_knowledge && visionCapabilities?.vision_available && (
        <div className="glass-card p-6 border-amber-500/20 bg-amber-500/5">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl bg-amber-500/20 flex items-center justify-center flex-shrink-0">
              <Eye className="w-6 h-6 text-amber-400" />
            </div>
            <div>
              <h3 className="font-semibold text-amber-400 mb-1">No Visual Patterns Learned Yet</h3>
              <p className="text-sm text-slate-400">
                The ML has not been trained on video frames yet. Go to <span className="text-indigo-400">Dashboard → Vision Training Manager</span> to
                analyze video frames and learn visual patterns like FVGs, Order Blocks, and Market Structure.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Training Controls */}
      <div className="glass-card p-6">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-indigo-400" />
              Training Controls
            </h2>
            <p className="text-sm text-slate-400 mt-1">Train or retrain the ML models</p>
          </div>
          <button
            onClick={handleTraining}
            disabled={training}
            className={`btn ${training ? 'btn-secondary opacity-50 cursor-not-allowed' : 'btn-primary'}`}
          >
            {training ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Training...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Train Models</span>
              </>
            )}
          </button>
        </div>

        {/* Training Result */}
        {trainingResult && (
          <div className={`p-5 rounded-xl animate-slide-up ${
            trainingResult.error
              ? 'bg-red-500/10 border border-red-500/20'
              : 'bg-emerald-500/10 border border-emerald-500/20'
          }`}>
            {trainingResult.error ? (
              <div className="flex items-center gap-3 text-red-400">
                <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center">
                  <AlertCircle className="w-5 h-5" />
                </div>
                <span className="font-medium">{trainingResult.error}</span>
              </div>
            ) : (
              <div>
                <div className="flex items-center gap-3 text-emerald-400 mb-4">
                  <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                    <CheckCircle className="w-5 h-5" />
                  </div>
                  <span className="font-semibold">Training Complete!</span>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { label: 'Transcripts', value: trainingResult.results?.n_transcripts },
                    { label: 'Classifier F1', value: `${(trainingResult.results?.classifier_f1 * 100)?.toFixed(1)}%` },
                    { label: 'Concepts Defined', value: trainingResult.results?.concepts_defined },
                    { label: 'Rules Extracted', value: trainingResult.results?.rules_extracted },
                  ].map((item, i) => (
                    <div key={i} className="glass-card-static p-3 text-center">
                      <p className="text-xs text-slate-400 mb-1">{item.label}</p>
                      <p className="text-lg font-bold text-white">{item.value}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Performance Trends */}
      {mlStatus?.classifier_performance && (
        <div className="glass-card p-6">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <Target className="w-5 h-5 text-indigo-400" />
            Performance Trends
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="glass-card-static p-6 flex flex-col items-center text-center">
              <div className={`w-16 h-16 rounded-xl flex items-center justify-center mb-4 ${
                mlStatus.classifier_performance.trend === 'improving'
                  ? 'bg-emerald-500/20'
                  : mlStatus.classifier_performance.trend === 'declining'
                  ? 'bg-red-500/20'
                  : 'bg-slate-500/20'
              }`}>
                <TrendingUp className={`w-8 h-8 ${
                  mlStatus.classifier_performance.trend === 'improving'
                    ? 'text-emerald-400'
                    : mlStatus.classifier_performance.trend === 'declining'
                    ? 'text-red-400'
                    : 'text-slate-400'
                }`} />
              </div>
              <p className="text-sm text-slate-400 mb-1">Trend</p>
              <p className="text-lg font-bold text-white capitalize">
                {mlStatus.classifier_performance.trend || 'No data'}
              </p>
            </div>
            <div className="glass-card-static p-6 flex flex-col items-center text-center">
              <ProgressRing
                value={((mlStatus.classifier_performance.current_f1 || 0) * 100).toFixed(0)}
                max={100}
                color="#6366f1"
                size={100}
              />
              <p className="text-sm text-slate-400 mt-4">Current F1 Score</p>
            </div>
            <div className="glass-card-static p-6 flex flex-col items-center text-center">
              <ProgressRing
                value={((mlStatus.classifier_performance.best_f1 || 0) * 100).toFixed(0)}
                max={100}
                color="#10b981"
                size={100}
              />
              <p className="text-sm text-slate-400 mt-4">Best F1 Score</p>
            </div>
          </div>
        </div>
      )}

      {/* All Transcripts */}
      <div className="glass-card p-6">
        <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-400" />
          All Transcripts ({transcripts.length})
        </h2>
        {transcripts.length > 0 ? (
          <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
            {[...transcripts]
              .sort((a, b) => new Date(b.transcribed_at) - new Date(a.transcribed_at))
              .map((t, i) => (
              <div
                key={i}
                className="glass-card-static p-4 flex items-center justify-between group hover:bg-white/[0.08] transition-colors animate-slide-up"
                style={{ animationDelay: `${i * 30}ms` }}
              >
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-white truncate group-hover:text-indigo-300 transition-colors">
                    {t.title || t.video_id}
                  </p>
                  <div className="flex items-center gap-3 mt-1 text-sm text-slate-500">
                    <span>{t.word_count?.toLocaleString()} words</span>
                    <span className="px-2 py-0.5 bg-indigo-500/10 text-indigo-300 rounded text-xs border border-indigo-500/20">
                      {t.method}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-500 ml-4">
                  <Clock className="w-3 h-3" />
                  {new Date(t.transcribed_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="w-20 h-20 rounded-2xl bg-slate-500/10 flex items-center justify-center mx-auto mb-6">
              <FileText className="w-10 h-10 text-slate-500" />
            </div>
            <h3 className="text-lg font-semibold text-slate-400 mb-2">No transcripts yet</h3>
            <p className="text-sm text-slate-500">
              Run transcript collection to start learning
            </p>
          </div>
        )}
      </div>

      {/* How It Works */}
      <div className="glass-card-static p-6 border-indigo-500/20 bg-indigo-500/5">
        <h2 className="text-xl font-bold text-indigo-300 mb-6 flex items-center gap-2">
          <Sparkles className="w-5 h-5" />
          How TradingMamba Learns
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            {
              step: '1',
              title: 'Transcript Collection',
              description: 'Smart Money videos are transcribed using free YouTube captions or local Whisper',
              icon: FileText,
              color: 'blue'
            },
            {
              step: '2',
              title: 'Vision Training',
              description: 'Video frames are analyzed to learn visual patterns like FVGs, Order Blocks, and chart structures',
              icon: Eye,
              color: 'cyan'
            },
            {
              step: '3',
              title: 'Concept Extraction',
              description: 'ML combines transcript and visual knowledge to understand Smart Money methodology',
              icon: Brain,
              color: 'purple'
            },
            {
              step: '4',
              title: 'Signal Generation',
              description: 'Learned patterns are detected on live charts to generate trading signals',
              icon: Target,
              color: 'emerald'
            }
          ].map((item, i) => {
            const IconComp = item.icon;
            return (
              <div key={i} className="flex flex-col items-center text-center animate-slide-up" style={{ animationDelay: `${i * 100}ms` }}>
                <div className={`w-14 h-14 rounded-xl bg-${item.color}-500/20 flex items-center justify-center mb-4`}>
                  <IconComp className={`w-7 h-7 text-${item.color}-400`} />
                </div>
                <div className={`w-8 h-8 rounded-full bg-${item.color}-500/20 flex items-center justify-center mb-3`}>
                  <span className={`text-${item.color}-400 font-bold text-sm`}>{item.step}</span>
                </div>
                <p className="font-semibold text-white mb-2">{item.title}</p>
                <p className="text-sm text-slate-400">{item.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
