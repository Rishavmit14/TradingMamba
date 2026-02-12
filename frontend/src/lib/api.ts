/** API client for the TradingMamba backend (localhost:8000). */

import { AnalysisResult, BacktestResult, DetailedSignals } from "./types";

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
