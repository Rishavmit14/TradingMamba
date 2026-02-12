/** API client for the TradingMamba backend (localhost:8000). */

import { AnalysisResult } from "./types";

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
