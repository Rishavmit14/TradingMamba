"use client";

import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Shield,
  Target,
  ArrowRight,
  Search,
} from "lucide-react";
import { TradingSignal } from "@/lib/types";

interface SignalCardProps {
  signal: TradingSignal;
  onShowDetails?: (signalId: string) => void;
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
};

export default function SignalCard({ signal, onShowDetails }: SignalCardProps) {
  const isBull = signal.direction === "bullish";
  const grade = gradeConfig[signal.grade] || gradeConfig.D;
  const mss = signal.mss_quality ? mssConfig[signal.mss_quality] : null;

  // Use trading_styles[] (merged) if available, otherwise fall back to single trading_style
  const styles = signal.trading_styles?.length
    ? signal.trading_styles
    : signal.trading_style
    ? [signal.trading_style]
    : [];

  return (
    <div className={`glass-card rounded-xl p-3.5 transition-all hover:border-[var(--border-hover)] ${
      isBull ? "hover:shadow-emerald-500/5" : "hover:shadow-red-500/5"
    } hover:shadow-lg ${signal.suppressed ? "opacity-50" : ""}`}>
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
          {styles.map((s) => {
            const cfg = styleConfig[s];
            return cfg ? (
              <div key={s} className={`px-1.5 py-0.5 rounded-md text-[10px] font-bold border ${cfg.bg} ${cfg.text} ${cfg.border}`}>
                {cfg.label}
              </div>
            ) : null;
          })}
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

      {/* Price levels — Entry + SL */}
      <div className="grid grid-cols-2 gap-2 mb-2">
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
      </div>

      {/* TP levels */}
      <div className={`grid gap-2 mb-3 ${
        (signal.take_profits?.length ?? 0) >= 3 ? "grid-cols-3" :
        (signal.take_profits?.length ?? 0) === 2 ? "grid-cols-2" : "grid-cols-1"
      }`}>
        {(signal.take_profits && signal.take_profits.length > 0 ? signal.take_profits : [
          { price: signal.take_profit, rr: signal.risk_reward_ratio, label: "TP1" }
        ]).map((tp, i) => (
          <div key={i} className={`rounded-lg px-2.5 py-2 ${
            i === 0 ? "bg-emerald-500/10" : "bg-emerald-500/5"
          }`}>
            <div className="flex items-center gap-1 mb-1">
              <Target className={`w-3 h-3 ${i === 0 ? "text-emerald-400" : "text-emerald-400/50"}`} />
              <span className={`text-xs ${i === 0 ? "text-emerald-400" : "text-emerald-400/50"}`}>{tp.label}</span>
              <span className={`text-[10px] font-mono ml-auto ${i === 0 ? "text-emerald-400/70" : "text-emerald-400/40"}`}>{tp.rr}R</span>
            </div>
            <span className={`text-xs font-mono font-medium ${i === 0 ? "text-emerald-400" : "text-emerald-400/70"}`}>
              ${tp.price.toLocaleString()}
            </span>
          </div>
        ))}
      </div>

      {/* Confidence */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="text-xs text-[var(--text-muted)]">
            Conf <span className="font-mono font-medium text-[var(--text-secondary)]">{signal.confidence_score}%</span>
          </span>
        </div>
      </div>

      {/* Confluences */}
      <div className="flex flex-wrap gap-1">
        {signal.confluences.map((c, i) => {
          const isFutures = c.startsWith("Futures:");
          const isFuturesWarn = isFutures && c.endsWith("\u26a0");
          return (
            <span
              key={i}
              className={`px-2 py-0.5 text-xs rounded-md border ${
                isFuturesWarn
                  ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                  : isFutures
                  ? "bg-cyan-500/10 text-cyan-400 border-cyan-500/20"
                  : "bg-[var(--bg-tertiary)] text-[var(--text-secondary)] border-[var(--border-primary)]"
              }`}
            >
              {c}
            </span>
          );
        })}
        {signal.quant_confluences?.map((c, i) => (
          <span
            key={`q${i}`}
            className="px-2 py-0.5 text-xs rounded-md border bg-violet-500/10 text-violet-400 border-violet-500/20"
          >
            {c}
          </span>
        ))}
      </div>

      {/* Quant Score (quant mode only) */}
      {signal.quant_score && (
        <div className="mb-3 p-2.5 rounded-lg bg-violet-500/5 border border-violet-500/15">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-bold text-violet-400 uppercase tracking-wider">Quant Score</span>
            <div className={`px-2 py-0.5 rounded-md text-xs font-bold border ${
              signal.quant_score.combined_score > 20
                ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                : signal.quant_score.combined_score < -20
                ? "bg-red-500/15 text-red-400 border-red-500/30"
                : "bg-slate-500/15 text-slate-400 border-slate-500/30"
            }`}>
              {signal.quant_score.combined_score > 0 ? "+" : ""}{signal.quant_score.combined_score.toFixed(0)}
            </div>
          </div>
          <div className="grid grid-cols-4 gap-1.5 text-[10px]">
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Alpha</div>
              <div className="font-mono font-medium text-violet-300">{signal.quant_score.alpha_score.toFixed(0)}</div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Micro</div>
              <div className="font-mono font-medium text-violet-300">{signal.quant_score.micro_score.toFixed(0)}</div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Risk</div>
              <div className="font-mono font-medium text-violet-300">{signal.quant_score.risk_tradability.toFixed(0)}</div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Exec</div>
              <div className="font-mono font-medium text-violet-300">{signal.quant_score.execution_score.toFixed(0)}</div>
            </div>
          </div>
          {(signal.quant_score.confirmations > 0 || signal.quant_score.contradictions > 0) && (
            <div className="mt-1.5 flex items-center gap-2 text-[10px]">
              {signal.quant_score.confirmations > 0 && (
                <span className="text-emerald-400">{signal.quant_score.confirmations} confirm{signal.quant_score.confirmations > 1 ? "s" : ""}</span>
              )}
              {signal.quant_score.contradictions > 0 && (
                <span className="text-red-400">{signal.quant_score.contradictions} contradict{signal.quant_score.contradictions > 1 ? "s" : ""}</span>
              )}
              {signal.quant_score.grade_change !== 0 && (
                <span className={signal.quant_score.grade_change > 0 ? "text-emerald-400" : "text-red-400"}>
                  Grade {signal.quant_score.grade_change > 0 ? "+" : ""}{signal.quant_score.grade_change}
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* ATR SL/TP + Position Size (quant mode) */}
      {(signal.atr_stop_loss != null && signal.atr_stop_loss > 0) && (
        <div className="grid grid-cols-3 gap-1.5 mb-2">
          <div className="bg-red-500/5 rounded-lg px-2 py-1.5 text-center">
            <div className="text-[10px] text-red-400/60">ATR SL</div>
            <div className="text-[10px] font-mono font-medium text-red-400">${signal.atr_stop_loss.toLocaleString()}</div>
          </div>
          <div className="bg-emerald-500/5 rounded-lg px-2 py-1.5 text-center">
            <div className="text-[10px] text-emerald-400/60">ATR TP</div>
            <div className="text-[10px] font-mono font-medium text-emerald-400">${(signal.atr_take_profit ?? 0).toLocaleString()}</div>
          </div>
          {(signal.position_size_pct != null && signal.position_size_pct > 0) && (
            <div className="bg-violet-500/5 rounded-lg px-2 py-1.5 text-center">
              <div className="text-[10px] text-violet-400/60">Size</div>
              <div className="text-[10px] font-mono font-medium text-violet-400">{signal.position_size_pct}%</div>
            </div>
          )}
        </div>
      )}

      {/* Warnings */}
      {signal.suppressed && (
        <div className="mt-2 flex items-center gap-1.5 bg-amber-500/10 border border-amber-500/20 rounded-lg px-2.5 py-1.5">
          <AlertTriangle className="w-3 h-3 text-amber-400" />
          <span className="text-xs text-amber-400">Suppressed — extreme volatility</span>
        </div>
      )}

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

      {/* Signal age indicator */}
      {(signal.bars_active != null && signal.bars_active > 0) && (
        <div className="mt-2 flex items-center justify-between text-[10px] text-[var(--text-muted)]">
          <span>Active for {signal.bars_active} cycle{signal.bars_active > 1 ? "s" : ""}</span>
          {styles.length > 1 && (
            <span className="font-medium text-amber-400">{styles.length} styles agree</span>
          )}
        </div>
      )}

      {/* Show Details button */}
      {signal.signal_id && onShowDetails && (
        <button
          onClick={() => onShowDetails(signal.signal_id!)}
          className="mt-2.5 w-full flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs font-medium text-blue-400 bg-blue-500/5 hover:bg-blue-500/10 border border-blue-500/15 hover:border-blue-500/30 rounded-lg transition-all"
        >
          <Search className="w-3 h-3" />
          Show Details
        </button>
      )}
    </div>
  );
}
