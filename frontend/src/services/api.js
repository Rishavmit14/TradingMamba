import axios from 'axios';

// Use absolute URL to bypass proxy issues
const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
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

export default api;
