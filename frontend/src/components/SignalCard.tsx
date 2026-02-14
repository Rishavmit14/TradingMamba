"use client";

import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Shield,
  Target,
  ArrowRight,
} from "lucide-react";
import { TradingSignal } from "@/lib/types";

interface SignalCardProps {
  signal: TradingSignal;
}

const gradeConfig: Record<string, { bg: string; text: string; border: string; label: string }> = {
  A: { bg: "bg-emerald-500/15", text: "text-emerald-400", border: "border-emerald-500/30", label: "A" },
  B: { bg: "bg-blue-500/15", text: "text-blue-400", border: "border-blue-500/30", label: "B" },
  C: { bg: "bg-amber-500/15", text: "text-amber-400", border: "border-amber-500/30", label: "C" },
  D: { bg: "bg-red-500/15", text: "text-red-400", border: "border-red-500/30", label: "D" },
};

const mssConfig: Record<string, { bg: string; text: string; border: string; label: string } | null> = {
  a_plus_plus: { bg: "bg-yellow-500/15", text: "text-yellow-400", border: "border-yellow-500/30", label: "MSS A++" },
  a_plus: { bg: "bg-cyan-500/15", text: "text-cyan-400", border: "border-cyan-500/30", label: "MSS A+" },
  standard: { bg: "bg-slate-500/15", text: "text-slate-400", border: "border-slate-500/30", label: "MSS" },
};

const styleConfig: Record<string, { bg: string; text: string; border: string; label: string }> = {
  positional:  { bg: "bg-purple-500/15", text: "text-purple-400", border: "border-purple-500/30", label: "Positional" },
  swing:       { bg: "bg-indigo-500/15", text: "text-indigo-400", border: "border-indigo-500/30", label: "Swing" },
  short_term:  { bg: "bg-sky-500/15",    text: "text-sky-400",    border: "border-sky-500/30",    label: "Short-Term" },
  intraday:    { bg: "bg-teal-500/15",   text: "text-teal-400",   border: "border-teal-500/30",   label: "Intraday" },
  day_trading: { bg: "bg-orange-500/15", text: "text-orange-400", border: "border-orange-500/30", label: "Day Trading" },
  scalping:    { bg: "bg-rose-500/15",   text: "text-rose-400",   border: "border-rose-500/30",   label: "Scalping" },
};

export default function SignalCard({ signal }: SignalCardProps) {
  const isBull = signal.direction === "bullish";
  const grade = gradeConfig[signal.grade] || gradeConfig.D;
  const mss = signal.mss_quality ? mssConfig[signal.mss_quality] : null;
  const style = signal.trading_style ? styleConfig[signal.trading_style] : null;

  return (
    <div className={`glass-card rounded-xl p-3.5 transition-all hover:border-[var(--border-hover)] ${
      isBull ? "hover:shadow-emerald-500/5" : "hover:shadow-red-500/5"
    } hover:shadow-lg`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
            isBull ? "bg-emerald-500/10" : "bg-red-500/10"
          }`}>
            {isBull
              ? <TrendingUp className="w-4 h-4 text-emerald-400" />
              : <TrendingDown className="w-4 h-4 text-red-400" />
            }
          </div>
          <div>
            <span className={`text-sm font-bold ${isBull ? "text-emerald-400" : "text-red-400"}`}>
              {isBull ? "LONG" : "SHORT"}
            </span>
            <span className="text-xs text-[var(--text-muted)] ml-2 font-mono">{signal.timeframe}</span>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {style && (
            <div className={`px-2 py-1 rounded-md text-xs font-bold border ${style.bg} ${style.text} ${style.border}`}>
              {style.label}
            </div>
          )}
          {mss && (
            <div className={`px-2 py-1 rounded-md text-xs font-bold border ${mss.bg} ${mss.text} ${mss.border}`}>
              {mss.label}
            </div>
          )}
          <div className={`px-2 py-1 rounded-md text-xs font-bold border ${grade.bg} ${grade.text} ${grade.border}`}>
            Grade {grade.label}
          </div>
        </div>
      </div>

      {/* Price levels */}
      <div className="grid grid-cols-3 gap-2 mb-3">
        <div className="bg-[var(--bg-tertiary)] rounded-lg px-2.5 py-2">
          <div className="flex items-center gap-1 mb-1">
            <ArrowRight className="w-3 h-3 text-[var(--text-muted)]" />
            <span className="text-xs text-[var(--text-muted)]">Entry</span>
          </div>
          <span className="text-xs font-mono font-medium text-[var(--text-primary)]">
            ${signal.entry_price.toLocaleString()}
          </span>
        </div>
        <div className="bg-red-500/5 rounded-lg px-2.5 py-2">
          <div className="flex items-center gap-1 mb-1">
            <Shield className="w-3 h-3 text-red-400/60" />
            <span className="text-xs text-red-400/60">SL</span>
          </div>
          <span className="text-xs font-mono font-medium text-red-400">
            ${signal.stop_loss.toLocaleString()}
          </span>
        </div>
        <div className="bg-emerald-500/5 rounded-lg px-2.5 py-2">
          <div className="flex items-center gap-1 mb-1">
            <Target className="w-3 h-3 text-emerald-400/60" />
            <span className="text-xs text-emerald-400/60">TP</span>
          </div>
          <span className="text-xs font-mono font-medium text-emerald-400">
            ${signal.take_profit.toLocaleString()}
          </span>
        </div>
      </div>

      {/* R:R and Confidence */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="text-xs text-[var(--text-muted)]">
            R:R <span className="font-mono font-medium text-[var(--text-secondary)]">{signal.risk_reward_ratio}</span>
          </span>
          <span className="text-xs text-[var(--text-muted)]">
            Conf <span className="font-mono font-medium text-[var(--text-secondary)]">{signal.confidence_score}%</span>
          </span>
        </div>
      </div>

      {/* Confluences */}
      <div className="flex flex-wrap gap-1">
        {signal.confluences.map((c, i) => (
          <span
            key={i}
            className="px-2 py-0.5 text-xs bg-[var(--bg-tertiary)] text-[var(--text-secondary)] rounded-md border border-[var(--border-primary)]"
          >
            {c}
          </span>
        ))}
      </div>

      {/* Warnings */}
      {signal.vsa_absorption && (
        <div className="mt-2.5 flex items-center gap-1.5 bg-emerald-500/5 border border-emerald-500/15 rounded-lg px-2.5 py-1.5">
          <TrendingUp className="w-3 h-3 text-emerald-400" />
          <span className="text-xs text-emerald-400">VSA Absorption confirmed</span>
        </div>
      )}

      {signal.is_counter_trend && (
        <div className="mt-2 flex items-center gap-1.5 bg-purple-500/5 border border-purple-500/15 rounded-lg px-2.5 py-1.5">
          <ArrowRight className="w-3 h-3 text-purple-400 rotate-180" />
          <span className="text-xs text-purple-400">Counter-trend signal</span>
        </div>
      )}
    </div>
  );
}
