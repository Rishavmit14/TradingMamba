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

// Detailed Signal Analysis
export const getSignalAnalysis = async (symbol, timeframe = 'H1') => {
  const response = await api.get(`/signals/analyze/${symbol}`, {
    params: { timeframe },
  });
  return response.data;
};

// Live OHLCV data for real-time charting
export const getLiveOHLCV = async (symbol, timeframe = 'M1', limit = 100) => {
  const response = await api.get(`/live/ohlcv/${symbol}`, {
    params: { timeframe, limit },
  });
  return response.data;
};

// WebSocket URL for live price updates
export const getWebSocketUrl = (symbol) => {
  return `ws://localhost:8000/ws/live/${symbol}`;
};

// ML Whitewash - Delete all trained models
export const whitewashML = async () => {
  const response = await api.post('/ml/whitewash');
  return response.data;
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
// Vision Training APIs - Multimodal video analysis
// ============================================================================

// Start vision training for a playlist
// visionProvider: 'local' (FREE on M1/M2/M3 Mac), 'anthropic' (paid), 'openai' (paid)
// extractionMode: 'comprehensive' (every 3s - learns everything), 'thorough' (5s), 'balanced' (10-15s), 'selective' (key moments)
export const trainWithVision = async (playlistId, visionProvider = 'local', maxFrames = 0, extractionMode = 'comprehensive') => {
  const response = await api.post(`/ml/train/vision/${playlistId}`, null, {
    params: { vision_provider: visionProvider, max_frames: maxFrames, extraction_mode: extractionMode },
  });
  return response.data;
};

// Start vision training for a SINGLE video
export const trainSingleVideoWithVision = async (videoId, visionProvider = 'local', maxFrames = 0, extractionMode = 'comprehensive') => {
  const response = await api.post(`/ml/train/vision/video/${videoId}`, null, {
    params: { vision_provider: visionProvider, max_frames: maxFrames, extraction_mode: extractionMode },
  });
  return response.data;
};

// Get vision training status
export const getVisionTrainingStatus = async (jobId) => {
  const response = await api.get(`/ml/train/vision/status/${jobId}`);
  return response.data;
};

// Stream vision training progress URL
export const getVisionTrainingStreamUrl = (jobId) => {
  return `${API_BASE}/ml/train/vision/stream/${jobId}`;
};

// Get vision capabilities status
export const getVisionCapabilities = async () => {
  const response = await api.get('/ml/vision/status');
  return response.data;
};

// Get visual pattern examples
export const getVisualPatternExamples = async (patternType) => {
  const response = await api.get(`/ml/vision/patterns/${patternType}`);
  return response.data;
};

// Get teaching moments
export const getTeachingMoments = async (concept = null) => {
  const response = await api.get('/ml/vision/teaching-moments', {
    params: concept ? { concept } : {},
  });
  return response.data;
};

// Get visual knowledge summary
export const getVisualKnowledge = async () => {
  const response = await api.get('/ml/vision/knowledge');
  return response.data;
};

export default api;
