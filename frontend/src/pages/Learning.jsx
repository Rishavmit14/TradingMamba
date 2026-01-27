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
  Zap
} from 'lucide-react';
import { getMLStatus, triggerTraining, getTranscripts } from '../services/api';

// Progress Bar
function ProgressBar({ value, max, label, color = 'blue' }) {
  const percentage = max > 0 ? (value / max) * 100 : 0;
  const colors = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500',
  };

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-white">{value} / {max}</span>
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

// Stat Card
function StatCard({ icon: Icon, label, value, description, color = 'gray' }) {
  const colors = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    purple: 'text-purple-400',
    orange: 'text-orange-400',
    gray: 'text-gray-400',
  };

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <Icon className={`${colors[color]} mb-3`} size={24} />
      <p className="text-sm text-gray-400">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
      {description && <p className="text-xs text-gray-500 mt-2">{description}</p>}
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
      loadData(); // Reload status
    } catch (err) {
      setTrainingResult({ error: err.message || 'Training failed' });
    } finally {
      setTraining(false);
    }
  };

  const totalWords = transcripts.reduce((sum, t) => sum + (t.word_count || 0), 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">ML Learning Status</h1>
          <p className="text-gray-400">Monitor and manage the AI learning process</p>
        </div>
        <button
          onClick={loadData}
          className="flex items-center space-x-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg"
        >
          <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
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
          color="blue"
        />
        <StatCard
          icon={Database}
          label="Total Words"
          value={totalWords.toLocaleString()}
          description="Training corpus size"
          color="purple"
        />
        <StatCard
          icon={TrendingUp}
          label="Training Runs"
          value={mlStatus?.n_training_runs || 0}
          description="Model iterations"
          color="green"
        />
        <StatCard
          icon={Brain}
          label="Model Status"
          value={mlStatus?.knowledge_base_loaded ? 'Trained' : 'Not Trained'}
          description={mlStatus?.classifier_performance?.trend || 'N/A'}
          color={mlStatus?.knowledge_base_loaded ? 'green' : 'orange'}
        />
      </div>

      {/* Training Controls */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-lg font-bold">Training Controls</h2>
            <p className="text-sm text-gray-400">Train or retrain the ML models</p>
          </div>
          <button
            onClick={handleTraining}
            disabled={training}
            className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-colors ${
              training
                ? 'bg-gray-600 cursor-not-allowed'
                : 'bg-blue-500 hover:bg-blue-600'
            }`}
          >
            {training ? (
              <>
                <RefreshCw size={18} className="animate-spin" />
                <span>Training...</span>
              </>
            ) : (
              <>
                <Play size={18} />
                <span>Train Models</span>
              </>
            )}
          </button>
        </div>

        {/* Training Result */}
        {trainingResult && (
          <div className={`p-4 rounded-lg ${
            trainingResult.error
              ? 'bg-red-500/10 border border-red-500/30'
              : 'bg-green-500/10 border border-green-500/30'
          }`}>
            {trainingResult.error ? (
              <div className="flex items-center space-x-2 text-red-400">
                <AlertCircle size={20} />
                <span>{trainingResult.error}</span>
              </div>
            ) : (
              <div>
                <div className="flex items-center space-x-2 text-green-400 mb-3">
                  <CheckCircle size={20} />
                  <span className="font-medium">Training Complete!</span>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-gray-400">Transcripts</p>
                    <p className="font-medium">{trainingResult.results?.n_transcripts}</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Classifier F1</p>
                    <p className="font-medium">
                      {(trainingResult.results?.classifier_f1 * 100)?.toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-400">Concepts Defined</p>
                    <p className="font-medium">{trainingResult.results?.concepts_defined}</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Rules Extracted</p>
                    <p className="font-medium">{trainingResult.results?.rules_extracted}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Performance Trends */}
      {mlStatus?.classifier_performance && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h2 className="text-lg font-bold mb-4">Performance Trends</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <p className="text-sm text-gray-400 mb-2">Trend</p>
              <div className={`flex items-center space-x-2 ${
                mlStatus.classifier_performance.trend === 'improving'
                  ? 'text-green-400'
                  : mlStatus.classifier_performance.trend === 'declining'
                  ? 'text-red-400'
                  : 'text-gray-400'
              }`}>
                <TrendingUp size={20} />
                <span className="font-medium capitalize">
                  {mlStatus.classifier_performance.trend || 'No data'}
                </span>
              </div>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-2">Current F1 Score</p>
              <p className="text-2xl font-bold">
                {((mlStatus.classifier_performance.current_f1 || 0) * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-2">Best F1 Score</p>
              <p className="text-2xl font-bold text-green-400">
                {((mlStatus.classifier_performance.best_f1 || 0) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Recent Transcripts */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h2 className="text-lg font-bold mb-4">Recent Transcripts</h2>
        {transcripts.length > 0 ? (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {transcripts.slice(0, 20).map((t, i) => (
              <div
                key={i}
                className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
              >
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{t.title || t.video_id}</p>
                  <p className="text-sm text-gray-400">
                    {t.word_count?.toLocaleString()} words | {t.method}
                  </p>
                </div>
                <div className="text-xs text-gray-500 ml-4">
                  {new Date(t.transcribed_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <FileText size={48} className="mx-auto mb-4 opacity-50" />
            <p>No transcripts yet</p>
            <p className="text-sm mt-2">Run transcript collection to start learning</p>
          </div>
        )}
      </div>

      {/* How It Works */}
      <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-6">
        <h2 className="text-lg font-bold text-blue-400 mb-4">How the Learning Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0">
              <span className="text-blue-400 font-bold">1</span>
            </div>
            <div>
              <p className="font-medium">Transcript Collection</p>
              <p className="text-gray-400">Smart Money videos are transcribed using free YouTube captions or local Whisper</p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0">
              <span className="text-blue-400 font-bold">2</span>
            </div>
            <div>
              <p className="font-medium">Concept Extraction</p>
              <p className="text-gray-400">ML models learn Smart Money concepts, patterns, and trading rules from transcripts</p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0">
              <span className="text-blue-400 font-bold">3</span>
            </div>
            <div>
              <p className="font-medium">Signal Generation</p>
              <p className="text-gray-400">Learned concepts are combined with market data to generate trading signals</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
