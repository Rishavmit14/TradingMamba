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
  Layers
} from 'lucide-react';
import { getMLStatus, triggerTraining, getTranscripts } from '../services/api';

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

// Main Learning Page
export default function Learning() {
  const [mlStatus, setMLStatus] = useState(null);
  const [transcripts, setTranscripts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [status, transcriptsData] = await Promise.all([
        getMLStatus().catch(() => null),
        getTranscripts().catch(() => ({ transcripts: [] })),
      ]);
      setMLStatus(status);
      setTranscripts(transcriptsData.transcripts || []);
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
          icon={Database}
          label="Total Words"
          value={totalWords.toLocaleString()}
          description="Training corpus size"
          gradient="from-purple-500 to-pink-500"
          delay={50}
        />
        <StatCard
          icon={TrendingUp}
          label="Training Runs"
          value={mlStatus?.n_training_runs || 0}
          description="Model iterations"
          gradient="from-emerald-500 to-teal-500"
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

      {/* Recent Transcripts */}
      <div className="glass-card p-6">
        <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-400" />
          Recent Transcripts
        </h2>
        {transcripts.length > 0 ? (
          <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
            {transcripts.slice(0, 20).map((t, i) => (
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
          How the Learning Works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            {
              step: '1',
              title: 'Transcript Collection',
              description: 'Smart Money videos are transcribed using free YouTube captions or local Whisper'
            },
            {
              step: '2',
              title: 'Concept Extraction',
              description: 'ML models learn Smart Money concepts, patterns, and trading rules from transcripts'
            },
            {
              step: '3',
              title: 'Signal Generation',
              description: 'Learned concepts are combined with market data to generate trading signals'
            }
          ].map((item, i) => (
            <div key={i} className="flex items-start gap-4 animate-slide-up" style={{ animationDelay: `${i * 100}ms` }}>
              <div className="w-10 h-10 rounded-xl bg-indigo-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-indigo-400 font-bold">{item.step}</span>
              </div>
              <div>
                <p className="font-semibold text-white mb-1">{item.title}</p>
                <p className="text-sm text-slate-400">{item.description}</p>
              </div>
              {i < 2 && (
                <ArrowRight className="w-5 h-5 text-indigo-500/30 hidden md:block absolute right-0 top-1/2 -translate-y-1/2" />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
