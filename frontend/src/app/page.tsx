"use client";

import { useState, useCallback } from "react";
import Chart from "@/components/Chart";
import DetectionPanel from "@/components/DetectionPanel";
import SignalCard from "@/components/SignalCard";
import { fetchAnalysis } from "@/lib/api";
import { AnalysisResult } from "@/lib/types";

const TIMEFRAMES = ["W1", "D1", "H4", "M15"] as const;

export default function Dashboard() {
  const [selectedTF, setSelectedTF] = useState<string>("H4");
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = useCallback(async (tf: string) => {
    setSelectedTF(tf);
    setLoading(true);
    setError(null);

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

  return (
    <div className="h-screen flex flex-col bg-[#0a0a0f]">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 border-b border-gray-800 bg-[#0d0d14]">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold text-white tracking-tight">
            TradingMamba
          </h1>
          <span className="text-xs text-gray-500 border border-gray-700 rounded px-1.5 py-0.5">
            SMC Engine
          </span>
          <span className="text-xs text-gray-600">BTCUSDT</span>
        </div>

        {/* Timeframe selector */}
        <div className="flex items-center gap-1">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf}
              onClick={() => runAnalysis(tf)}
              disabled={loading}
              className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
                selectedTF === tf
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white"
              } disabled:opacity-50`}
            >
              {tf}
            </button>
          ))}
        </div>

        {/* Status */}
        <div className="flex items-center gap-3 text-xs">
          {analysis && (
            <span className={`px-2 py-0.5 rounded font-medium ${
              analysis.trend === "bullish"
                ? "bg-green-900/50 text-green-400"
                : analysis.trend === "bearish"
                ? "bg-red-900/50 text-red-400"
                : "bg-gray-800 text-gray-400"
            }`}>
              {analysis.trend.toUpperCase()}
            </span>
          )}
          {analysis?.session?.is_kill_zone && (
            <span className="px-2 py-0.5 rounded bg-yellow-900/50 text-yellow-400 font-medium">
              KILL ZONE
            </span>
          )}
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chart area */}
        <div className="flex-1 flex flex-col">
          {/* Chart */}
          <div className="flex-1 p-2">
            {loading ? (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <div className="animate-spin w-8 h-8 border-2 border-gray-600 border-t-blue-500 rounded-full mx-auto mb-3" />
                  <p className="text-sm">Fetching {selectedTF} candles & running analysis...</p>
                </div>
              </div>
            ) : error ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <p className="text-red-400 text-sm mb-2">Error: {error}</p>
                  <p className="text-gray-500 text-xs">
                    Make sure the backend is running on localhost:8000
                  </p>
                  <button
                    onClick={() => runAnalysis(selectedTF)}
                    className="mt-3 px-4 py-1.5 bg-gray-800 text-white text-xs rounded hover:bg-gray-700"
                  >
                    Retry
                  </button>
                </div>
              </div>
            ) : !analysis ? (
              <div className="flex items-center justify-center h-full text-gray-600">
                <div className="text-center">
                  <p className="text-lg mb-2">TradingMamba SMC Engine</p>
                  <p className="text-sm text-gray-500 mb-4">
                    Select a timeframe to analyze BTCUSDT
                  </p>
                  <button
                    onClick={() => runAnalysis("H4")}
                    className="px-6 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-500"
                  >
                    Analyze H4
                  </button>
                </div>
              </div>
            ) : (
              <Chart data={analysis} />
            )}
          </div>

          {/* Signals bar at bottom */}
          {analysis && analysis.signals.length > 0 && (
            <div className="border-t border-gray-800 p-3 bg-[#0d0d14]">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                Active Signals ({analysis.signals.length})
              </h3>
              <div className="flex gap-3 overflow-x-auto">
                {analysis.signals.map((sig, i) => (
                  <div key={i} className="min-w-[280px]">
                    <SignalCard signal={sig} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right sidebar — Detection Panel */}
        <div className="w-72 border-l border-gray-800 bg-[#0d0d14] overflow-y-auto">
          <div className="sticky top-0 bg-[#0d0d14] border-b border-gray-800 px-4 py-2">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Detections
            </h2>
          </div>
          <DetectionPanel data={analysis} />
        </div>
      </div>
    </div>
  );
}
