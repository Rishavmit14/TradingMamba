/** API client for the TradingMamba backend (localhost:8000). */

import {
  AnalysisResult,
  BacktestResult,
  DetailedSignals,
  DeepAnalysis,
  DemoAccount,
  DemoTrade,
  DemoEquityPoint,
  EngineMode,
  TelegramStatus,
  ResolvedSignal,
  SignalStoreStats,
  MarketIntelData,
} from "./types";

const API_BASE = "http://localhost:8000/api";

export async function fetchAnalysis(timeframe: string, mode: EngineMode = "smc"): Promise<AnalysisResult> {
  const res = await fetch(`${API_BASE}/analyze/${timeframe}?mode=${mode}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchMultiTFAnalysis(mode: EngineMode = "smc"): Promise<Record<string, AnalysisResult>> {
  const res = await fetch(`${API_BASE}/analyze?mode=${mode}`);
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
export async function fetchDetailedSignals(mode: EngineMode = "smc"): Promise<DetailedSignals> {
  const res = await fetch(`${API_BASE}/signals/detailed?mode=${mode}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

/** Deep Analysis: comprehensive multi-TF analysis for a signal */
export async function fetchDeepAnalysis(signalId: string, mode: EngineMode = "smc"): Promise<DeepAnalysis> {
  const res = await fetch(`${API_BASE}/signals/${signalId}/deep-analysis?mode=${mode}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

/** Signal lifecycle: resolved signals (SL hit, TP hit, expired) */
export async function fetchResolvedSignals(
  limit: number = 50,
  mode: EngineMode = "smc"
): Promise<{ resolved: ResolvedSignal[]; stats: SignalStoreStats }> {
  const res = await fetch(`${API_BASE}/signals/resolved?limit=${limit}&mode=${mode}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

/** Phase 5: Demo Account API */
export async function fetchDemoAccount(mode: EngineMode = "smc"): Promise<DemoAccount> {
  const res = await fetch(`${API_BASE}/demo/account?mode=${mode}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function resetDemoAccount(mode: EngineMode = "smc"): Promise<void> {
  const res = await fetch(`${API_BASE}/demo/account/reset?mode=${mode}`, { method: "POST" });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
}

export async function updateDemoSettings(settings: {
  risk_per_trade_pct?: number;
}, mode: EngineMode = "smc"): Promise<void> {
  const res = await fetch(`${API_BASE}/demo/account/settings?mode=${mode}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
}

export async function fetchDemoTrades(
  status?: string,
  limit?: number,
  mode: EngineMode = "smc"
): Promise<DemoTrade[]> {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  if (limit) params.set("limit", String(limit));
  params.set("mode", mode);
  const res = await fetch(`${API_BASE}/demo/trades?${params}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = await res.json();
  return data.trades;
}

export async function takeDemoTrade(tradeId: number, mode: EngineMode = "smc"): Promise<DemoTrade> {
  const res = await fetch(`${API_BASE}/demo/trades/${tradeId}/take?mode=${mode}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function skipDemoTrade(tradeId: number, mode: EngineMode = "smc"): Promise<void> {
  const res = await fetch(`${API_BASE}/demo/trades/${tradeId}/skip?mode=${mode}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
}

export async function updateTradeSLTP(tradeId: number, params: {
  stop_loss?: number;
  take_profit?: number;
}, mode: EngineMode = "smc"): Promise<DemoTrade> {
  const res = await fetch(`${API_BASE}/demo/trades/${tradeId}/sl-tp?mode=${mode}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function closeDemoTrade(tradeId: number, mode: EngineMode = "smc"): Promise<DemoTrade> {
  const res = await fetch(`${API_BASE}/demo/trades/${tradeId}/close?mode=${mode}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function createManualTrade(params: {
  direction: "bullish" | "bearish";
  stop_loss: number;
  take_profit: number;
}, mode: EngineMode = "smc"): Promise<DemoTrade> {
  const res = await fetch(`${API_BASE}/demo/trades/manual?mode=${mode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchDemoEquity(mode: EngineMode = "smc"): Promise<DemoEquityPoint[]> {
  const res = await fetch(`${API_BASE}/demo/equity?mode=${mode}`);
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
