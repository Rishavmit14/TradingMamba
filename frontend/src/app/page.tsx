"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import {
  Activity,
  BarChart3,
  Zap,
  TrendingUp,
  TrendingDown,
  Minus,
  Loader2,
  AlertTriangle,
  Radio,
} from "lucide-react";
import Chart from "@/components/Chart";
import DetectionPanel from "@/components/DetectionPanel";
import SignalCard from "@/components/SignalCard";
import AnalysisPanel from "@/components/AnalysisPanel";
import ElementPicker from "@/components/ElementPicker";
import { fetchAnalysis, fetchLivePrice, PriceTicker } from "@/lib/api";
import { AnalysisResult, DetectorVisibility, SelectedElement, ChartClickResult, ClickCandidate } from "@/lib/types";

const TIMEFRAMES = ["1M", "W1", "D1", "H4", "H1", "M15", "M5"] as const;

const DETECTOR_LABELS: { key: keyof DetectorVisibility; label: string; color: string }[] = [
  { key: "swings", label: "Swings", color: "#f59e0b" },
  { key: "idm", label: "IDM", color: "#60a5fa" },
  { key: "bos", label: "BOS", color: "#22d3ee" },
  { key: "choch", label: "CHoCH", color: "#a855f7" },
  { key: "fvg", label: "FVG", color: "#10b981" },
  { key: "ob", label: "OB", color: "#3b82f6" },
  { key: "pd", label: "P/D", color: "#eab308" },
];

const DEFAULT_VISIBILITY: DetectorVisibility = {
  swings: true, idm: true, bos: true, choch: true, fvg: true, ob: true, pd: true,
};

function TrendIcon({ trend }: { trend: string }) {
  if (trend === "bullish") return <TrendingUp className="w-3.5 h-3.5" />;
  if (trend === "bearish") return <TrendingDown className="w-3.5 h-3.5" />;
  return <Minus className="w-3.5 h-3.5" />;
}

export default function Dashboard() {
  const [selectedTF, setSelectedTF] = useState<string>("H4");
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [visibility, setVisibility] = useState<DetectorVisibility>(DEFAULT_VISIBILITY);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [ticker, setTicker] = useState<PriceTicker | null>(null);
  const [selectedElement, setSelectedElement] = useState<SelectedElement | null>(null);
  const [pickerState, setPickerState] = useState<{ candidates: ClickCandidate[]; x: number; y: number } | null>(null);

  const handleChartClick = useCallback((result: ChartClickResult | null) => {
    if (!result || result.candidates.length === 0) {
      setSelectedElement(null);
      setPickerState(null);
      return;
    }

    if (result.candidates.length === 1) {
      // Single candidate — select directly
      const c = result.candidates[0];
      setSelectedElement({ type: c.type, index: c.index, candle_index: c.candle_index });
      setPickerState(null);
    } else {
      // Multiple candidates — show picker
      setPickerState({ candidates: result.candidates, x: result.clickX, y: result.clickY });
    }
  }, []);

  const handlePickerSelect = useCallback((candidate: ClickCandidate) => {
    setSelectedElement({ type: candidate.type, index: candidate.index, candle_index: candidate.candle_index });
    setPickerState(null);
  }, []);

  const toggleDetector = useCallback((key: keyof DetectorVisibility) => {
    setVisibility((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const silentRefresh = useCallback(async (tf: string) => {
    try {
      const result = await fetchAnalysis(tf);
      setAnalysis(result);
      setError(null);
    } catch {
      // Silent fail — keep showing last data
    }
  }, []);

  const runAnalysis = useCallback(async (tf: string) => {
    setSelectedTF(tf);
    setLoading(true);
    setError(null);
    setSelectedElement(null);
    setPickerState(null);

    try {
      const result = await fetchAnalysis(tf);
      setAnalysis(result);
    } catch (err: any) {
      setError(err.message || "Failed to fetch analysis");
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-refresh polling
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const selectedTFRef = useRef(selectedTF);
  selectedTFRef.current = selectedTF;

  useEffect(() => {
    if (!analysis) return;
    const pollMs: Record<string, number> = {
      "M5": 5_000, "M15": 10_000, "H1": 15_000, "H4": 30_000,
      "D1": 60_000, "W1": 60_000, "1M": 120_000,
    };
    const ms = pollMs[selectedTF] ?? 30_000;
    intervalRef.current = setInterval(() => silentRefresh(selectedTFRef.current), ms);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [analysis, selectedTF, silentRefresh]);

  // Live price ticker — polls Binance every 2s
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const t = await fetchLivePrice();
        if (active) setTicker(t);
      } catch { /* silent */ }
    };
    poll(); // immediate first fetch
    const id = setInterval(poll, 2_000);
    return () => { active = false; clearInterval(id); };
  }, []);

  const price = ticker?.price ?? null;
  const priceChangePct = ticker?.changePercent ?? 0;

  return (
    <div className="h-screen flex flex-col bg-[var(--bg-primary)] overflow-hidden">
      {/* ============ HEADER ============ */}
      <header className="glass border-b border-[var(--border-primary)] z-20">
        <div className="flex items-center justify-between px-4 h-12">
          {/* Logo + pair */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-blue-500/20">
                <Activity className="w-4 h-4 text-white" />
              </div>
              <span className="text-sm font-semibold tracking-tight gradient-text">
                TradingMamba
              </span>
            </div>

            <div className="h-5 w-px bg-[var(--border-primary)]" />

            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-[var(--text-primary)]">BTCUSDT</span>
              {price && (
                <div className="flex items-center gap-1.5">
                  <span className="text-sm font-mono font-medium text-[var(--text-primary)]">
                    ${price.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                  </span>
                  <span className={`text-xs font-mono ${priceChangePct >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {priceChangePct >= 0 ? "+" : ""}{priceChangePct.toFixed(2)}%
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Timeframe pills */}
          <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-lg p-0.5">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf}
                onClick={() => runAnalysis(tf)}
                disabled={loading}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-200 ${
                  selectedTF === tf
                    ? "bg-blue-600 text-white shadow-lg shadow-blue-600/20"
                    : "text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-card)]"
                } disabled:opacity-50`}
              >
                {tf}
              </button>
            ))}
          </div>

          {/* Detector toggles */}
          <div className="flex items-center gap-1">
            {DETECTOR_LABELS.map(({ key, label, color }) => (
              <button
                key={key}
                onClick={() => toggleDetector(key)}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all duration-200 border ${
                  visibility[key]
                    ? ""
                    : "border-transparent bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                }`}
                style={
                  visibility[key]
                    ? {
                        color,
                        backgroundColor: `${color}15`,
                        borderColor: `${color}40`,
                      }
                    : undefined
                }
              >
                {label}
              </button>
            ))}
          </div>

          {/* Status cluster */}
          <div className="flex items-center gap-2">
            {analysis && (
              <div className="flex items-center gap-1.5 text-emerald-400 text-xs">
                <Radio className="w-3 h-3 animate-pulse" />
                <span className="font-medium">LIVE</span>
              </div>
            )}

            {analysis && (
              <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium ${
                analysis.trend === "bullish"
                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                  : analysis.trend === "bearish"
                  ? "bg-red-500/10 text-red-400 border border-red-500/20"
                  : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] border border-[var(--border-primary)]"
              }`}>
                <TrendIcon trend={analysis.trend} />
                {analysis.trend.toUpperCase()}
              </div>
            )}

            {analysis?.climax_warning && (
              <div className="flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                <AlertTriangle className="w-3 h-3" />
                CLIMAX
              </div>
            )}

            {analysis?.session?.is_kill_zone && (
              <div className="flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20">
                <Zap className="w-3 h-3" />
                KILL ZONE
              </div>
            )}

            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className={`p-1.5 rounded-md transition-colors ${
                sidebarOpen
                  ? "text-blue-400 bg-blue-500/10"
                  : "text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)]"
              }`}
              title="Toggle panel"
            >
              <BarChart3 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* ============ MAIN CONTENT ============ */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chart area */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 min-h-0 p-2">
            {loading ? (
              <div className="flex items-center justify-center h-full animate-fade-in">
                <div className="text-center">
                  <Loader2 className="w-10 h-10 text-blue-500 animate-spin mx-auto mb-4" />
                  <p className="text-sm text-[var(--text-secondary)]">
                    Analyzing {selectedTF} structure...
                  </p>
                  <p className="text-xs text-[var(--text-muted)] mt-1">
                    Detecting swings, IDM, BOS, CHoCH, FVG, OB
                  </p>
                </div>
              </div>
            ) : error ? (
              <div className="flex items-center justify-center h-full animate-fade-in">
                <div className="text-center max-w-sm">
                  <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center mx-auto mb-4">
                    <AlertTriangle className="w-6 h-6 text-red-400" />
                  </div>
                  <p className="text-sm text-red-400 mb-1">Connection Error</p>
                  <p className="text-xs text-[var(--text-muted)] mb-4">{error}</p>
                  <button
                    onClick={() => runAnalysis(selectedTF)}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded-lg transition-colors shadow-lg shadow-blue-600/20"
                  >
                    Retry
                  </button>
                </div>
              </div>
            ) : !analysis ? (
              <div className="flex items-center justify-center h-full animate-fade-in">
                <div className="text-center max-w-md">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-cyan-400/20 border border-blue-500/20 flex items-center justify-center mx-auto mb-6">
                    <Activity className="w-8 h-8 text-blue-400" />
                  </div>
                  <h2 className="text-xl font-semibold gradient-text mb-2">
                    TradingMamba
                  </h2>
                  <p className="text-sm text-[var(--text-secondary)] mb-1">
                    Smart Money Concepts Detection Engine
                  </p>
                  <p className="text-xs text-[var(--text-muted)] mb-6">
                    Select a timeframe to analyze BTCUSDT structure
                  </p>
                  <button
                    onClick={() => runAnalysis("H4")}
                    className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-blue-600/25"
                  >
                    Analyze H4
                  </button>
                </div>
              </div>
            ) : (
              <Chart data={analysis} visibility={visibility} livePrice={price} onElementClick={handleChartClick} />
            )}
          </div>

          {/* Radial element picker — shows when multiple elements overlap */}
          {pickerState && (
            <ElementPicker
              candidates={pickerState.candidates}
              x={pickerState.x}
              y={pickerState.y}
              onPick={handlePickerSelect}
              onDismiss={() => setPickerState(null)}
            />
          )}

          {/* Analysis Panel — shows when a chart element is clicked */}
          {analysis && selectedElement && (
            <AnalysisPanel
              element={selectedElement}
              data={analysis}
              onClose={() => { setSelectedElement(null); setPickerState(null); }}
            />
          )}

          {/* Signals bar */}
          {analysis && analysis.signals.length > 0 && (
            <div className="border-t border-[var(--border-primary)] bg-[var(--bg-secondary)] animate-fade-in">
              <div className="px-4 py-3">
                <div className="flex items-center gap-2 mb-2.5">
                  <Zap className="w-3.5 h-3.5 text-amber-400" />
                  <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                    Active Signals
                  </h3>
                  <span className="text-xs font-mono text-[var(--text-muted)] bg-[var(--bg-tertiary)] px-1.5 py-0.5 rounded">
                    {analysis.signals.length}
                  </span>
                </div>
                <div className="flex gap-3 overflow-x-auto pb-1">
                  {analysis.signals.map((sig, i) => (
                    <div key={i} className="min-w-[300px] animate-slide-in-right" style={{ animationDelay: `${i * 50}ms` }}>
                      <SignalCard signal={sig} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right sidebar */}
        {sidebarOpen && (
          <div className="w-80 border-l border-[var(--border-primary)] bg-[var(--bg-secondary)] overflow-hidden flex flex-col animate-slide-in-right">
            <div className="flex items-center justify-between px-4 h-10 border-b border-[var(--border-primary)] flex-shrink-0">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                <h2 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                  Detections
                </h2>
              </div>
              {analysis && (
                <span className="text-xs font-mono text-[var(--text-muted)]">
                  {analysis.timeframe}
                </span>
              )}
            </div>
            <div className="flex-1 overflow-y-auto">
              <DetectionPanel data={analysis} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
