import axios from 'axios';

// Use absolute URL to bypass proxy issues
const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 minutes for chart generation
});

// Signals & Analysis
export const analyzeSymbol = async (symbol, timeframes = 'H1,H4,D1') => {
  const response = await api.get(`/signals/analyze/${symbol}`, {
    params: { timeframes },
  });
  return response.data;
};

export const quickSignal = async (symbol) => {
  const response = await api.get(`/signals/quick/${symbol}`);
  return response.data;
};

// Market Data
export const getPrice = async (symbol) => {
  const response = await api.get(`/market/price/${symbol}`);
  return response.data;
};

export const getOHLCV = async (symbol, timeframe = 'H1', limit = 100) => {
  const response = await api.get(`/market/ohlcv/${symbol}`, {
    params: { timeframe, limit },
  });
  return response.data;
};

export const getSymbols = async () => {
  const response = await api.get('/market/symbols');
  return response.data;
};

// ML & Concepts
export const getMLStatus = async () => {
  const response = await api.get('/ml/status');
  return response.data;
};

export const queryConcept = async (conceptName) => {
  const response = await api.get(`/ml/concepts/${conceptName}`);
  return response.data;
};

export const predictConcepts = async (text) => {
  const response = await api.post('/ml/predict', null, {
    params: { text },
  });
  return response.data;
};

export const triggerTraining = async (incremental = true) => {
  const response = await api.post('/ml/train', null, {
    params: { incremental },
  });
  return response.data;
};

// System Status
export const getAnalysisStatus = async () => {
  const response = await api.get('/analysis/status');
  return response.data;
};

// Transcripts
export const getTranscripts = async () => {
  const response = await api.get('/transcripts');
  return response.data;
};

// Concepts
export const getConcepts = async () => {
  const response = await api.get('/concepts');
  return response.data;
};

// Playlist Processing
export const addPlaylist = async (url, tier = 3, trainAfter = true) => {
  const response = await api.post('/playlist/add', null, {
    params: { url, tier, train_after: trainAfter },
  });
  return response.data;
};

export const getPlaylistStatus = async (jobId) => {
  const response = await api.get(`/playlist/status/${jobId}`);
  return response.data;
};

export const getProcessingJobs = async () => {
  const response = await api.get('/playlist/jobs');
  return response.data;
};

// Stream playlist progress (returns EventSource URL)
export const getPlaylistStreamUrl = (jobId) => {
  return `${API_BASE}/playlist/stream/${jobId}`;
};

// Chart Generation with Smart Money annotations
export const getChart = async (symbol, timeframe = 'H1', withSignal = true, withPatterns = true) => {
  const response = await api.get(`/chart/${symbol}`, {
    params: { timeframe, with_signal: withSignal, with_patterns: withPatterns },
  });
  return response.data;
};

// Detailed Signal Analysis (playlist-aware, single timeframe)
export const getSignalAnalysis = async (symbol, timeframe = 'H1', playlistId = 'all') => {
  const response = await api.get(`/signals/analyze/${symbol}`, {
    params: {
      timeframes: timeframe,  // Backend expects 'timeframes' param (can be comma-separated, but we send single TF)
      playlist_id: playlistId
    },
  });
  return response.data;
};

// Get available playlists with training stats (for dropdown)
export const getAvailablePlaylists = async () => {
  const response = await api.get('/playlists/available');
  return response.data;
};

// Live OHLCV data for real-time charting
// Falls back to market/ohlcv if live endpoint fails (e.g., IP banned by Binance)
// end_time parameter enables historical scrollback (fetch candles before this timestamp)
export const getLiveOHLCV = async (symbol, timeframe = 'M1', limit = 100, end_time = null) => {
  try {
    const params = { timeframe, limit };
    if (end_time) {
      params.end_time = end_time;
    }
    const response = await api.get(`/live/ohlcv/${symbol}`, { params });
    return response.data;
  } catch (error) {
    // Fallback to market OHLCV endpoint
    console.log('Live OHLCV failed, falling back to market OHLCV:', error.message);
    const response = await api.get(`/market/ohlcv/${symbol}`, {
      params: { timeframe, limit },
    });
    // Transform market OHLCV format to match live format expected by LiveChart
    const marketData = response.data;
    const candles = (marketData.data || [])
      .map(d => {
        // Handle both "Date" (for D1/W1/MN) and "Datetime" (for intraday) keys
        const dateStr = d.Date || d.Datetime;
        return {
          time: Math.floor(new Date(dateStr).getTime() / 1000),
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
          volume: d.volume
        };
      })
      .filter(c => !isNaN(c.time) && c.time > 0)  // Filter out invalid times
      .sort((a, b) => a.time - b.time);  // Sort ascending by time

    return {
      symbol: marketData.symbol,
      timeframe: marketData.timeframe,
      candles
    };
  }
};

// WebSocket URL for live price updates
export const getWebSocketUrl = (symbol) => {
  return `ws://localhost:8000/ws/live/${symbol}`;
};

// Get transcripts grouped by playlist
export const getTranscriptsGrouped = async () => {
  const response = await api.get('/transcripts/grouped');
  return response.data;
};

// Train ML from specific playlist
export const trainFromPlaylist = async (playlistId) => {
  const response = await api.post(`/ml/train/playlist/${playlistId}`);
  return response.data;
};

// Get selective training status
export const getSelectiveTrainingStatus = async (jobId) => {
  const response = await api.get(`/ml/train/status/${jobId}`);
  return response.data;
};

// Stream training progress URL
export const getTrainingStreamUrl = (jobId) => {
  return `${API_BASE}/ml/train/stream/${jobId}`;
};

// ============================================================================
// Synchronized Learning APIs (RECOMMENDED)
// Audio-Visual verified training - prevents contamination
// ============================================================================

// Get synchronized knowledge (verified audio-visual aligned)
export const getSynchronizedKnowledge = async (concept = null) => {
  const params = concept ? { concept } : {};
  const response = await api.get('/ml/synchronized/knowledge', { params });
  return response.data;
};

// Start synchronized training (state-of-the-art)
// This is the RECOMMENDED training method that verifies audio matches visual
export const startSynchronizedTraining = async (playlistId, options = {}) => {
  const response = await api.post(`/ml/train/synchronized/${playlistId}`, null, {
    params: {
      vision_provider: options.visionProvider || 'local',
      max_frames: options.maxFrames || 0,
      extraction_mode: options.extractionMode || 'sincere_student',
      alignment_threshold: options.alignmentThreshold || 0.6,
      sync_window: options.syncWindow || 2.0,
    },
  });
  return response.data;
};

// Get synchronized training status
export const getSyncTrainingStatus = async (jobId) => {
  const response = await api.get(`/ml/train/synchronized/status/${jobId}`);
  return response.data;
};

// Get teaching moments from synchronized knowledge
export const getTeachingMoments = async (concept = null) => {
  // Now redirects to synchronized knowledge endpoint
  const response = await api.get('/ml/synchronized/teaching-moments', {
    params: concept ? { concept } : {},
  });
  return response.data;
};

// Get visual pattern examples from synchronized knowledge
export const getVisualPatternExamples = async (patternType) => {
  // Now redirects to synchronized knowledge endpoint
  const response = await api.get(`/ml/synchronized/patterns/${patternType}`);
  return response.data;
};

// ============================================================================
// DEPRECATED Vision Training APIs
// These now redirect to Synchronized Learning on the backend
// Kept for backward compatibility only
// ============================================================================

// DEPRECATED: Use startSynchronizedTraining instead
// This now redirects to synchronized training on the backend
export const trainWithVision = async (playlistId, visionProvider = 'local', maxFrames = 0, extractionMode = 'comprehensive') => {
  console.warn('trainWithVision is DEPRECATED. Use startSynchronizedTraining instead.');
  // Redirect to synchronized training
  return startSynchronizedTraining(playlistId, {
    visionProvider,
    maxFrames,
    extractionMode: 'sincere_student', // Map to sync mode
  });
};

// DEPRECATED: Use startSynchronizedTraining instead
export const trainSingleVideoWithVision = async (videoId, visionProvider = 'local', maxFrames = 0, extractionMode = 'comprehensive') => {
  console.warn('trainSingleVideoWithVision is DEPRECATED. Use startSynchronizedTraining instead.');
  // Backend endpoint redirects to synchronized training
  const response = await api.post(`/ml/train/vision/video/${videoId}`, null, {
    params: { vision_provider: visionProvider, max_frames: maxFrames, extraction_mode: extractionMode },
  });
  return response.data;
};

// DEPRECATED: Use getSyncTrainingStatus instead
export const getVisionTrainingStatus = async (jobId) => {
  console.warn('getVisionTrainingStatus is DEPRECATED. Use getSyncTrainingStatus instead.');
  return getSyncTrainingStatus(jobId);
};

// DEPRECATED: Use getSyncTrainingStreamUrl instead
export const getVisionTrainingStreamUrl = (jobId) => {
  console.warn('getVisionTrainingStreamUrl is DEPRECATED.');
  return `${API_BASE}/ml/train/synchronized/stream/${jobId}`;
};

// DEPRECATED: Use getSynchronizedKnowledge instead
export const getVisionCapabilities = async () => {
  console.warn('getVisionCapabilities is DEPRECATED. Use getSynchronizedKnowledge instead.');
  const response = await api.get('/ml/synchronized/status');
  return response.data;
};

// DEPRECATED: Use getSynchronizedKnowledge instead
export const getVisualKnowledge = async () => {
  console.warn('getVisualKnowledge is DEPRECATED. Use getSynchronizedKnowledge instead.');
  return getSynchronizedKnowledge();
};

// ============================================================================
// Hedge Fund APIs - Institutional-grade pattern analysis
// ============================================================================

// Get hedge fund components status
export const getHedgeFundStatus = async () => {
  const response = await api.get('/hedge-fund/status');
  return response.data;
};

// Get edge statistics for patterns
// patternType: optional filter (e.g., 'fvg', 'order_block')
export const getEdgeStatistics = async (patternType = null) => {
  const params = patternType ? { pattern_type: patternType } : {};
  const response = await api.get('/hedge-fund/edge-statistics', { params });
  return response.data;
};

// Grade a pattern (A+ to F)
export const gradePattern = async (patternType, patternData, marketContext) => {
  const response = await api.post('/hedge-fund/grade-pattern', null, {
    params: {
      pattern_type: patternType,
      pattern_data: JSON.stringify(patternData),
      market_context: JSON.stringify(marketContext),
    },
  });
  return response.data;
};

// Analyze multi-timeframe confluence
export const analyzeConfluence = async (primaryPattern, primaryTf, allTfPatterns) => {
  const response = await api.post('/hedge-fund/analyze-confluence', null, {
    params: {
      primary_pattern: JSON.stringify(primaryPattern),
      primary_tf: primaryTf,
      all_tf_patterns: JSON.stringify(allTfPatterns),
    },
  });
  return response.data;
};

// Record trade outcome for edge tracking (feedback loop)
export const recordTrade = async (patternType, outcome, rrAchieved = 0, session = '', dayOfWeek = '') => {
  const response = await api.post('/hedge-fund/record-trade', null, {
    params: {
      pattern_type: patternType,
      outcome,
      rr_achieved: rrAchieved,
      session,
      day_of_week: dayOfWeek,
    },
  });
  return response.data;
};

// Get best performing patterns ranked by expectancy
export const getBestPatterns = async (minSignals = 10) => {
  const response = await api.get('/hedge-fund/best-patterns', {
    params: { min_signals: minSignals },
  });
  return response.data;
};

// ============================================================================
// Quant Engine APIs (Tier 1-5) - Backtesting, ML Classifiers, Regime Detection
// ============================================================================

// Get quant dashboard (all tiers summary)
export const getQuantDashboard = async () => {
  const response = await api.get('/quant/dashboard');
  return response.data;
};

// Get market regime for a symbol
export const getQuantRegime = async (symbol) => {
  const response = await api.get(`/quant/regime/${symbol}`);
  return response.data;
};

// Get multi-asset correlation matrix
export const getQuantCorrelation = async () => {
  const response = await api.get('/quant/correlation');
  return response.data;
};

// Get full quant signal for a symbol
export const getQuantSignal = async (symbol, timeframe = 'D1') => {
  const response = await api.post(`/quant/signal/${symbol}`, null, {
    params: { timeframe },
  });
  return response.data;
};

// Run backtest for a symbol
export const runBacktest = async (symbol, timeframe = 'D1', lookbackDays = 365) => {
  const response = await api.post(`/backtest/${symbol}`, null, {
    params: { timeframe, lookback_days: lookbackDays },
  });
  return response.data;
};

// Get backtest results
export const getBacktestResults = async (symbol) => {
  const response = await api.get(`/backtest/results/${symbol}`);
  return response.data;
};

// Train ML classifier for a symbol
export const trainMLClassifier = async (symbol, timeframe = 'D1') => {
  const response = await api.post(`/ml/train-classifier/${symbol}`, null, {
    params: { timeframe },
  });
  return response.data;
};

// Get ML classifier status
export const getMLClassifierStatus = async () => {
  const response = await api.get('/ml/classifier/status');
  return response.data;
};

// ============================================================================
// Tier 6: Genuine ML Price Predictor
// ============================================================================

// Generate forward price predictions for a symbol
export const predictPrice = async (symbol, timeframe = 'D1') => {
  const response = await api.post(`/ml/predict-price/${symbol}`, null, {
    params: { timeframe },
  });
  return response.data;
};

// Get recent predictions and outcomes for a symbol
export const getPredictions = async (symbol, limit = 20) => {
  const response = await api.get(`/ml/predictions/${symbol}`, {
    params: { limit },
  });
  return response.data;
};

// Train prediction models for a symbol
export const trainPredictor = async (symbol, timeframe = 'D1') => {
  const response = await api.post(`/ml/train-predictor/${symbol}`, null, {
    params: { timeframe },
  });
  return response.data;
};

// Get live accuracy metrics from resolved predictions
export const getPredictorPerformance = async (symbol = null, lookbackDays = 90) => {
  const params = { lookback_days: lookbackDays };
  if (symbol) params.symbol = symbol;
  const response = await api.get('/ml/predictor/performance', { params });
  return response.data;
};

// Get status of all trained prediction models
export const getPredictorStatus = async () => {
  const response = await api.get('/ml/predictor/status');
  return response.data;
};

// Resolve pending predictions against actual prices
export const resolvePredictions = async () => {
  const response = await api.post('/ml/predictor/resolve');
  return response.data;
};

// ============================================================================
// Video Knowledge Integration - Video training → ML features bridge
// ============================================================================

// Get video knowledge index status (concepts, teaching depth, co-occurrence)
export const getVideoKnowledgeStatus = async () => {
  const response = await api.get('/ml/video-knowledge/status');
  return response.data;
};

// Train pattern quality model (video-learned features + OHLCV)
export const trainPatternQuality = async (symbol = 'BTCUSDT', timeframe = 'D1') => {
  const response = await api.post('/ml/train-pattern-quality', null, {
    params: { symbol, timeframe },
  });
  return response.data;
};

// Get pattern quality model status and feature importance
export const getPatternQualityStatus = async () => {
  const response = await api.get('/ml/pattern-quality/status');
  return response.data;
};

export default api;
