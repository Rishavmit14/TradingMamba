"use client";

import { TradingSignal } from "@/lib/types";

interface SignalCardProps {
  signal: TradingSignal;
}

const gradeColors: Record<string, string> = {
  A: "bg-green-500 text-white",
  B: "bg-blue-500 text-white",
  C: "bg-yellow-500 text-black",
  D: "bg-red-500 text-white",
};

export default function SignalCard({ signal }: SignalCardProps) {
  const isBull = signal.direction === "bullish";

  return (
    <div className={`rounded-lg border p-3 ${
      isBull ? "border-green-800 bg-green-950/30" : "border-red-800 bg-red-950/30"
    }`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`text-lg font-bold ${isBull ? "text-green-400" : "text-red-400"}`}>
            {isBull ? "LONG" : "SHORT"}
          </span>
          <span className={`px-2 py-0.5 text-xs font-bold rounded ${gradeColors[signal.grade]}`}>
            {signal.grade}
          </span>
        </div>
        <span className="text-gray-400 text-xs font-mono">{signal.timeframe}</span>
      </div>

      <div className="grid grid-cols-3 gap-2 mb-2 text-xs">
        <div>
          <span className="text-gray-500 block">Entry</span>
          <span className="text-white font-mono">${signal.entry_price.toLocaleString()}</span>
        </div>
        <div>
          <span className="text-gray-500 block">SL</span>
          <span className="text-red-400 font-mono">${signal.stop_loss.toLocaleString()}</span>
        </div>
        <div>
          <span className="text-gray-500 block">TP</span>
          <span className="text-green-400 font-mono">${signal.take_profit.toLocaleString()}</span>
        </div>
      </div>

      <div className="flex items-center justify-between text-xs mb-2">
        <span className="text-gray-400">R:R {signal.risk_reward_ratio}</span>
        <span className="text-gray-400">Confidence: {signal.confidence_score}%</span>
      </div>

      <div className="flex flex-wrap gap-1">
        {signal.confluences.map((c, i) => (
          <span key={i} className="px-1.5 py-0.5 text-xs bg-gray-800 text-gray-300 rounded">
            {c}
          </span>
        ))}
      </div>

      {signal.climax_warning && (
        <div className="mt-2 text-yellow-400 text-xs bg-yellow-900/20 rounded px-2 py-1">
          Climax warning active
        </div>
      )}
    </div>
  );
}
