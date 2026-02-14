/** API client for the TradingMamba backend (localhost:8000). */

import {
  AnalysisResult,
  BacktestResult,
  DetailedSignals,
  DeepAnalysis,
  DemoAccount,
  DemoTrade,
  DemoEquityPoint,
  TelegramStatus,
  ResolvedSignal,
  SignalStoreStats,
  MarketIntelData,
} from "./types";

const API_BASE = "http://localhost:8000/api";

export async function fetchAnalysis(timeframe: string): Promise<AnalysisResult> {
  const res = await fetch(`${API_BASE}/analyze/${timeframe}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchMultiTFAnalysis(): Promise<Record<string, AnalysisResult>> {
  const res = await fetch(`${API_BASE}/analyze`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

/** Phase 3: Backtest API */
export async function runBacktest(params: {
  start_date: string;
  end_date: string;
  symbol?: string;
  step_size?: number;
}): Promise<BacktestResult> {
  const searchParams = new URLSearchParams({
    start_date: params.start_date,
    end_date: params.end_date,
    symbol: params.symbol || "BTCUSDT",
    step_size: String(params.step_size || 96),
  });
  const res = await fetch(`${API_BASE}/backtest?${searchParams}`, { method: "POST" });
  if (!res.ok) throw new Error(`Backtest error: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchLatestBacktest(): Promise<BacktestResult | null> {
  const res = await fetch(`${API_BASE}/backtest/latest`);
  if (!res.ok) return null;
  const data = await res.json();
  if (data.error) return null;
  return data;
}

/** Phase 4: Detailed Signals API */
export async function fetchDetailedSignals(): Promise<DetailedSignals> {
  const res = await fetch(`${API_BASE}/signals/detailed`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

/** Deep Analysis: comprehensive multi-TF analysis for a signal */
export async function fetchDeepAnalysis(signalId: string): Promise<DeepAnalysis> {
  const res = await fetch(`${API_BASE}/signals/${signalId}/deep-analysis`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

/** Signal lifecycle: resolved signals (SL hit, TP hit, expired) */
export async function fetchResolvedSignals(
  limit: number = 50
): Promise<{ resolved: ResolvedSignal[]; stats: SignalStoreStats }> {
  const res = await fetch(`${API_BASE}/signals/resolved?limit=${limit}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

/** Phase 5: Demo Account API */
export async function fetchDemoAccount(): Promise<DemoAccount> {
  const res = await fetch(`${API_BASE}/demo/account`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function resetDemoAccount(): Promise<void> {
  const res = await fetch(`${API_BASE}/demo/account/reset`, { method: "POST" });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
}

export async function updateDemoSettings(settings: {
  risk_per_trade_pct?: number;
}): Promise<void> {
  const res = await fetch(`${API_BASE}/demo/account/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
}

export async function fetchDemoTrades(
  status?: string,
  limit?: number
): Promise<DemoTrade[]> {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  if (limit) params.set("limit", String(limit));
  const res = await fetch(`${API_BASE}/demo/trades?${params}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = await res.json();
  return data.trades;
}

export async function takeDemoTrade(tradeId: number): Promise<DemoTrade> {
  const res = await fetch(`${API_BASE}/demo/trades/${tradeId}/take`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function skipDemoTrade(tradeId: number): Promise<void> {
  const res = await fetch(`${API_BASE}/demo/trades/${tradeId}/skip`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
}

export async function updateTradeSLTP(tradeId: number, params: {
  stop_loss?: number;
  take_profit?: number;
}): Promise<DemoTrade> {
  const res = await fetch(`${API_BASE}/demo/trades/${tradeId}/sl-tp`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function closeDemoTrade(tradeId: number): Promise<DemoTrade> {
  const res = await fetch(`${API_BASE}/demo/trades/${tradeId}/close`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function createManualTrade(params: {
  direction: "bullish" | "bearish";
  stop_loss: number;
  take_profit: number;
}): Promise<DemoTrade> {
  const res = await fetch(`${API_BASE}/demo/trades/manual`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchDemoEquity(): Promise<DemoEquityPoint[]> {
  const res = await fetch(`${API_BASE}/demo/equity`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = await res.json();
  return data.equity_curve;
}

export async function fetchTelegramStatus(): Promise<TelegramStatus> {
  const res = await fetch(`${API_BASE}/telegram/status`);
  if (!res.ok) return { connected: false, chat_id: null, bot_username: null };
  return res.json();
}

export async function sendTelegramTest(): Promise<void> {
  const res = await fetch(`${API_BASE}/telegram/test`, { method: "POST" });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
}

/** Phase 6: Market Intel API */
export async function fetchMarketIntel(symbol: string = "BTCUSDT"): Promise<MarketIntelData> {
  const res = await fetch(`${API_BASE}/market-intel?symbol=${symbol}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

/** Lightweight price ticker from Binance (no backend needed). */
export interface PriceTicker {
  price: number;
  changePercent: number;
  high24h: number;
  low24h: number;
  volume24h: number;
}

export async function fetchLivePrice(): Promise<PriceTicker> {
  const res = await fetch("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT");
  if (!res.ok) throw new Error("Binance API error");
  const data = await res.json();
  return {
    price: parseFloat(data.lastPrice),
    changePercent: parseFloat(data.priceChangePercent),
    high24h: parseFloat(data.highPrice),
    low24h: parseFloat(data.lowPrice),
    volume24h: parseFloat(data.volume),
  };
}
