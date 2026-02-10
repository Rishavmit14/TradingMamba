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
