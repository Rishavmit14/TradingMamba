"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  Zap,
  TrendingUp,
  TrendingDown,
  Minus,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  ChevronDown,
  ChevronRight,
  Radio,
  Shield,
  Target,
  ArrowRight,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { fetchDetailedSignals, fetchResolvedSignals, deleteResolvedSignal, clearResolvedSignals } from "@/lib/api";
import {
  DetailedSignals,
  EngineMode,
  TradingSignal,
  ChecklistItem,
  ChecklistStatus,
  TrendState,
  ResolvedSignal,
  SignalStoreStats,
} from "@/lib/types";

// ── V20 Trading Style Metadata ──

interface TradingStyleMeta {
  key: string;
  label: string;
  biasTF: string;
  setupTF: string;
  entryTF: string;
  color: string;
  bgColor: string;
  borderColor: string;
}

const TRADING_STYLES: TradingStyleMeta[] = [
  { key: "positional",  label: "Positional",  biasTF: "1M", setupTF: "W1",  entryTF: "D1",  color: "text-purple-400",  bgColor: "bg-purple-500/15",  borderColor: "border-purple-500/30" },
  { key: "swing",       label: "Swing",       biasTF: "W1", setupTF: "D1",  entryTF: "H4",  color: "text-indigo-400",  bgColor: "bg-indigo-500/15",  borderColor: "border-indigo-500/30" },
  { key: "short_term",  label: "Short-Term",  biasTF: "D1", setupTF: "H4",  entryTF: "H1",  color: "text-sky-400",     bgColor: "bg-sky-500/15",     borderColor: "border-sky-500/30" },
  { key: "intraday",    label: "Intraday",    biasTF: "H4", setupTF: "H1",  entryTF: "M15", color: "text-teal-400",    bgColor: "bg-teal-500/15",    borderColor: "border-teal-500/30" },
];

// ── Entry Method Config (for badges on cards) ──

const ENTRY_METHOD_COLORS: Record<string, { color: string; bg: string; border: string; label: string }> = {
  mss:            { color: "text-emerald-400", bg: "bg-emerald-500/15", border: "border-emerald-500/30", label: "MSS" },
  sbc:            { color: "text-blue-400",    bg: "bg-blue-500/15",    border: "border-blue-500/30",    label: "SBC" },
  pullback_break: { color: "text-purple-400",  bg: "bg-purple-500/15",  border: "border-purple-500/30",  label: "Pullback" },
  scob:           { color: "text-amber-400",   bg: "bg-amber-500/15",   border: "border-amber-500/30",   label: "SCOB" },
};

// ── Helpers ──

function TrendPill({ trend, label }: { trend: TrendState; label: string }) {
  const config: Record<string, { bg: string; text: string; icon: typeof TrendingUp }> = {
    bullish: { bg: "bg-emerald-500/10 border-emerald-500/20", text: "text-emerald-400", icon: TrendingUp },
    bearish: { bg: "bg-red-500/10 border-red-500/20", text: "text-red-400", icon: TrendingDown },
    ranging: { bg: "bg-[var(--bg-tertiary)] border-[var(--border-primary)]", text: "text-[var(--text-muted)]", icon: Minus },
  };
  const c = config[trend] || config.ranging;
  const Icon = c.icon;
  return (
    <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border ${c.bg} ${c.text}`}>
      <Icon className="w-3 h-3" />
      <span>{label}</span>
      <span className="font-bold uppercase">{trend}</span>
    </div>
  );
}

function StatusIcon({ status }: { status: ChecklistStatus }) {
  if (status === "passed") return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
  if (status === "failed") return <XCircle className="w-4 h-4 text-red-400" />;
  return <Clock className="w-4 h-4 text-amber-400" />;
}

const gradeConfig: Record<string, { bg: string; text: string; border: string }> = {
  A: { bg: "bg-emerald-500/15", text: "text-emerald-400", border: "border-emerald-500/30" },
  B: { bg: "bg-blue-500/15", text: "text-blue-400", border: "border-blue-500/30" },
  C: { bg: "bg-amber-500/15", text: "text-amber-400", border: "border-amber-500/30" },
  D: { bg: "bg-red-500/15", text: "text-red-400", border: "border-red-500/30" },
};

// ── Signal Card (enhanced for Signals tab) ──

function SignalCardFull({ signal }: { signal: TradingSignal }) {
  const isBull = signal.direction === "bullish";
  const grade = gradeConfig[signal.grade] || gradeConfig.D;
  const method = signal.entry_method ? ENTRY_METHOD_COLORS[signal.entry_method] : null;

  // Use merged trading_styles[] if available
  const styles = signal.trading_styles?.length
    ? signal.trading_styles
    : signal.trading_style
    ? [signal.trading_style]
    : [];

  return (
    <div className={`glass-card rounded-xl p-4 transition-all hover:border-[var(--border-hover)] ${
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
          {signal.is_counter_trend && (
            <span className="px-1.5 py-0.5 text-[10px] font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20 rounded">
              CT
            </span>
          )}
          {method && (
            <div className={`px-2 py-1 rounded-md text-[10px] font-bold border ${method.bg} ${method.color} ${method.border}`}>
              {method.label}
            </div>
          )}
          <div className={`px-2 py-1 rounded-md text-xs font-bold border ${grade.bg} ${grade.text} ${grade.border}`}>
            Grade {signal.grade}
          </div>
        </div>
      </div>

      {/* Multi-style badges */}
      {styles.length > 1 && (
        <div className="flex items-center gap-1 mb-3">
          <span className="text-[10px] text-[var(--text-muted)]">Styles:</span>
          {styles.map((s) => {
            const meta = TRADING_STYLES.find((ts) => ts.key === s);
            return meta ? (
              <span key={s} className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${meta.bgColor} ${meta.color} ${meta.borderColor}`}>
                {meta.label}
              </span>
            ) : null;
          })}
        </div>
      )}

      {/* Price levels — Entry + SL */}
      <div className="grid grid-cols-2 gap-2 mb-2">
        <div className="bg-[var(--bg-tertiary)] rounded-lg px-2.5 py-2">
          <div className="flex items-center gap-1 mb-1">
            <ArrowRight className="w-3 h-3 text-[var(--text-muted)]" />
            <span className="text-[10px] text-[var(--text-muted)]">Entry</span>
          </div>
          <span className="text-xs font-mono font-medium text-[var(--text-primary)]">
            ${signal.entry_price.toLocaleString()}
          </span>
        </div>
        <div className="bg-red-500/5 rounded-lg px-2.5 py-2">
          <div className="flex items-center gap-1 mb-1">
            <Shield className="w-3 h-3 text-red-400/60" />
            <span className="text-[10px] text-red-400/60">SL</span>
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
              <span className={`text-[10px] ${i === 0 ? "text-emerald-400" : "text-emerald-400/50"}`}>{tp.label}</span>
              <span className={`text-[10px] font-mono ml-auto ${i === 0 ? "text-emerald-400/70" : "text-emerald-400/40"}`}>{tp.rr}R</span>
            </div>
            <span className={`text-xs font-mono font-medium ${i === 0 ? "text-emerald-400" : "text-emerald-400/70"}`}>
              ${tp.price.toLocaleString()}
            </span>
          </div>
        ))}
      </div>

      {/* Confidence + Type */}
      <div className="flex items-center gap-4 mb-3">
        <span className="text-xs text-[var(--text-muted)]">
          Conf <span className="font-mono font-medium text-[var(--text-secondary)]">{signal.confidence_score}%</span>
        </span>
        <span className="text-xs text-[var(--text-muted)]">
          Type <span className="font-mono font-medium text-[var(--text-secondary)]">{signal.pattern_type}</span>
        </span>
      </div>

      {/* Confluences */}
      <div className="flex flex-wrap gap-1">
        {signal.confluences.map((c, i) => (
          <span
            key={i}
            className="px-2 py-0.5 text-[10px] bg-[var(--bg-tertiary)] text-[var(--text-secondary)] rounded-md border border-[var(--border-primary)]"
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

      {/* Signal age */}
      {(signal.bars_active != null && signal.bars_active > 0) && (
        <div className="mt-2 text-[10px] text-[var(--text-muted)]">
          Active for {signal.bars_active} cycle{signal.bars_active > 1 ? "s" : ""}
        </div>
      )}
    </div>
  );
}

// ── Main Component ──

export default function SignalsTab({ mode = "smc" as EngineMode }: { mode?: EngineMode }) {
  const [data, setData] = useState<DetailedSignals | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [v23Open, setV23Open] = useState(false);
  const [resolvedSignals, setResolvedSignals] = useState<ResolvedSignal[]>([]);
  const [storeStats, setStoreStats] = useState<SignalStoreStats | null>(null);
  const [outcomesOpen, setOutcomesOpen] = useState(true);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    setError(null);
    try {
      const result = await fetchDetailedSignals(mode);
      setData(result);

      // Fetch resolved signals separately — don't block main signals if this fails
      try {
        const resolvedData = await fetchResolvedSignals(20, mode);
        setResolvedSignals(resolvedData.resolved);
        setStoreStats(resolvedData.stats);
      } catch {
        // Resolved endpoint may not be available yet — silently ignore
      }
    } catch (err: any) {
      if (!silent) setError(err.message || "Failed to fetch signals");
    } finally {
      if (!silent) setLoading(false);
    }
  }, [mode]);

  const handleDeleteOne = useCallback(async (signalId: string) => {
    try {
      await deleteResolvedSignal(signalId, mode);
      setResolvedSignals((prev) => prev.filter((rs) => rs.signal_id !== signalId));
      // Re-fetch stats after delete
      try {
        const resolvedData = await fetchResolvedSignals(20, mode);
        setStoreStats(resolvedData.stats);
      } catch { /* silent */ }
    } catch { /* silent */ }
  }, [mode]);

  const handleClearAll = useCallback(async () => {
    try {
      await clearResolvedSignals(mode);
      setResolvedSignals([]);
      setStoreStats((prev) => prev ? { ...prev, resolved_count: 0, sl_hits: 0, tp_hits: 0, expired: 0, win_rate: 0 } : null);
    } catch { /* silent */ }
  }, [mode]);

  // Initial fetch + 30s auto-refresh
  useEffect(() => {
    fetchData();
    intervalRef.current = setInterval(() => fetchData(true), 30_000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [fetchData]);

  // Use all_style_signals (all 6 trading styles) — fall back to signals (intraday-only)
  const allSignals = data?.all_style_signals?.length ? data.all_style_signals : (data?.signals ?? []);

  // Group signals by trading style — use trading_styles[] (merged) for correct multi-style placement
  const signalsByStyle: Record<string, TradingSignal[]> = {};
  for (const style of TRADING_STYLES) {
    signalsByStyle[style.key] = [];
  }
  signalsByStyle["other"] = [];
  for (const sig of allSignals) {
    const styles = sig.trading_styles?.length
      ? sig.trading_styles
      : sig.trading_style
      ? [sig.trading_style]
      : ["other"];
    let placed = false;
    for (const s of styles) {
      if (signalsByStyle[s]) {
        signalsByStyle[s].push(sig);
        placed = true;
      }
    }
    if (!placed) {
      signalsByStyle["other"].push(sig);
    }
  }

  // Multi-style alignment detection — use trading_styles[] for accurate counting
  const bullStyleSet = new Set<string>();
  const bearStyleSet = new Set<string>();
  for (const sig of allSignals) {
    const styles = sig.trading_styles?.length
      ? sig.trading_styles
      : sig.trading_style
      ? [sig.trading_style]
      : [];
    for (const s of styles) {
      if (sig.direction === "bullish") bullStyleSet.add(s);
      else bearStyleSet.add(s);
    }
  }
  const alignedCount = Math.max(bullStyleSet.size, bearStyleSet.size);
  const alignedDir = bullStyleSet.size >= bearStyleSet.size ? "bullish" : "bearish";

  const v24Score = data?.checklist_v24?.filter((c) => c.status === "passed").length ?? 0;
  const v24Total = data?.checklist_v24?.length ?? 6;

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center h-full animate-fade-in">
        <div className="text-center">
          <Loader2 className="w-10 h-10 text-blue-500 animate-spin mx-auto mb-4" />
          <p className="text-sm text-[var(--text-secondary)]">Loading multi-TF analysis...</p>
          <p className="text-xs text-[var(--text-muted)] mt-1">Fetching W1, D1, H4, H1, M15 data</p>
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="flex items-center justify-center h-full animate-fade-in">
        <div className="text-center max-w-sm">
          <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center mx-auto mb-4">
            <AlertTriangle className="w-6 h-6 text-red-400" />
          </div>
          <p className="text-sm text-red-400 mb-1">Connection Error</p>
          <p className="text-xs text-[var(--text-muted)] mb-4">{error}</p>
          <button
            onClick={() => fetchData()}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      {/* ═══ SECTION A: Market Context Bar ═══ */}
      {data && (
        <div className="glass-card rounded-xl p-4 animate-fade-in">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Radio className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
              <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                Multi-Timeframe Context
              </h3>
            </div>
            <button
              onClick={() => fetchData()}
              className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
            >
              <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </button>
          </div>

          <div className="flex flex-wrap gap-2 mb-3">
            <TrendPill trend={data.context.w1_trend} label="W1" />
            <TrendPill trend={data.context.d1_trend} label="D1" />
            <TrendPill trend={data.context.h4_trend} label="H4" />
            <TrendPill trend={data.context.h1_trend} label="H1" />
            <TrendPill trend={data.context.m15_trend} label="M15" />
          </div>

          <div className="flex flex-wrap gap-2">
            {/* Session */}
            {data.context.session && (
              <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border ${
                data.context.session.is_kill_zone
                  ? "bg-purple-500/10 text-purple-400 border-purple-500/20"
                  : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] border-[var(--border-primary)]"
              }`}>
                <Zap className="w-3 h-3" />
                {data.context.session.name.toUpperCase()}
                {data.context.session.is_kill_zone && " (Kill Zone)"}
              </div>
            )}

            {/* Phase */}
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-[var(--bg-tertiary)] text-[var(--text-secondary)] border border-[var(--border-primary)]">
              Phase: {data.context.current_phase}
            </div>

            {/* P/D Zone */}
            {data.context.premium_discount && (
              <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border ${
                data.context.premium_discount.zone === "discount"
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                  : data.context.premium_discount.zone === "premium"
                  ? "bg-red-500/10 text-red-400 border-red-500/20"
                  : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] border-[var(--border-primary)]"
              }`}>
                {data.context.premium_discount.zone.toUpperCase()} zone ({data.context.premium_discount.depth_pct}%)
              </div>
            )}

            {/* VSA */}
            {data.context.vsa_active && (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <TrendingUp className="w-3 h-3" />
                VSA ABSORPTION
              </div>
            )}
          </div>
        </div>
      )}

      {/* ═══ SECTION B: Multi-Style Alignment ═══ */}
      {alignedCount >= 2 && (
        <div className={`glass-card rounded-xl p-4 animate-fade-in ${
          alignedDir === "bullish"
            ? "border-emerald-500/20"
            : "border-red-500/20"
        }`}>
          <div className="flex items-center gap-3">
            {alignedDir === "bullish"
              ? <TrendingUp className="w-5 h-5 text-emerald-400" />
              : <TrendingDown className="w-5 h-5 text-red-400" />
            }
            <div>
              <span className={`text-sm font-bold ${alignedDir === "bullish" ? "text-emerald-400" : "text-red-400"}`}>
                {alignedCount} Trading Styles Aligned {alignedDir.toUpperCase()}
              </span>
              {alignedCount >= 3 && (
                <p className="text-xs text-[var(--text-muted)] mt-0.5">
                  Strong multi-style confluence — consider holding for bigger targets
                </p>
              )}
            </div>
            <div className="ml-auto flex items-center gap-1">
              {Array.from(alignedDir === "bullish" ? bullStyleSet : bearStyleSet).map(style => {
                const meta = TRADING_STYLES.find(s => s.key === style);
                return meta ? (
                  <span key={style} className={`px-2 py-0.5 rounded text-[10px] font-bold border ${meta.bgColor} ${meta.color} ${meta.borderColor}`}>
                    {meta.label}
                  </span>
                ) : null;
              })}
            </div>
          </div>
        </div>
      )}

      {/* ═══ SECTION C: Signals by Trading Style ═══ */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-amber-400" />
          <h2 className="text-sm font-semibold text-[var(--text-primary)]">
            Active Signals by Trading Style
          </h2>
          {data && (
            <span className="text-xs font-mono text-[var(--text-muted)] bg-[var(--bg-tertiary)] px-1.5 py-0.5 rounded">
              {allSignals.length} total
            </span>
          )}
        </div>

        {TRADING_STYLES.map((style) => {
          const signals = signalsByStyle[style.key] || [];

          return (
            <div key={style.key} className="glass-card rounded-xl overflow-hidden animate-fade-in">
              {/* Style Header */}
              <div className="px-4 py-3 border-b border-[var(--border-primary)]">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`px-2.5 py-1 rounded-lg text-xs font-bold border ${style.bgColor} ${style.color} ${style.borderColor}`}>
                      {style.label}
                    </div>
                    <div className="text-[10px] text-[var(--text-muted)] font-mono">
                      {style.biasTF} → {style.setupTF} → {style.entryTF}
                    </div>
                  </div>
                  <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                    signals.length > 0
                      ? `${style.bgColor} ${style.color}`
                      : "bg-[var(--bg-tertiary)] text-[var(--text-muted)]"
                  }`}>
                    {signals.length}
                  </span>
                </div>
              </div>

              {/* Signals Grid */}
              <div className="p-4">
                {signals.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                    {signals.map((sig, i) => (
                      <SignalCardFull key={i} signal={sig} />
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-[var(--text-muted)] text-center py-3">
                    No active signals for {style.label}
                  </p>
                )}
              </div>
            </div>
          );
        })}

        {/* Other / unclassified signals */}
        {signalsByStyle["other"]?.length > 0 && (
          <div className="glass-card rounded-xl p-4">
            <h3 className="text-xs font-semibold text-[var(--text-muted)] mb-3">Other Signals</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {signalsByStyle["other"].map((sig, i) => (
                <SignalCardFull key={i} signal={sig} />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ═══ SECTION D: V24 6-Rule Checklist ═══ */}
      {data?.checklist_v24 && (
        <div className="glass-card rounded-xl p-4 animate-fade-in">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-blue-400" />
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                V24 — 6-Rule Trading Framework
              </h3>
              <span className="text-[10px] text-[var(--text-muted)]">
                &ldquo;If all 6 rules fulfilled, ANY concept works&rdquo;
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className={`text-sm font-bold font-mono ${
                v24Score >= 5 ? "text-emerald-400" : v24Score >= 3 ? "text-amber-400" : "text-red-400"
              }`}>
                {v24Score}/{v24Total}
              </span>
            </div>
          </div>

          {/* Progress bar */}
          <div className="w-full h-1.5 bg-[var(--bg-tertiary)] rounded-full mb-4 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                v24Score >= 5 ? "bg-emerald-400" : v24Score >= 3 ? "bg-amber-400" : "bg-red-400"
              }`}
              style={{ width: `${(v24Score / v24Total) * 100}%` }}
            />
          </div>

          {/* Checklist items */}
          <div className="space-y-2">
            {data.checklist_v24.map((item) => (
              <div
                key={item.rule}
                className={`flex items-start gap-3 px-3 py-2.5 rounded-lg border transition-colors ${
                  item.status === "passed"
                    ? "bg-emerald-500/5 border-emerald-500/10"
                    : item.status === "failed"
                    ? "bg-red-500/5 border-red-500/10"
                    : "bg-amber-500/5 border-amber-500/10"
                }`}
              >
                <StatusIcon status={item.status} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-[var(--text-muted)]">R{item.rule}</span>
                    <span className="text-xs font-medium text-[var(--text-primary)]">{item.name}</span>
                  </div>
                  <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{item.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ═══ SECTION E: V23 Master Checklist (Collapsible) ═══ */}
      {data?.checklist_v23 && (
        <div className="glass-card rounded-xl overflow-hidden animate-fade-in">
          <button
            onClick={() => setV23Open(!v23Open)}
            className="w-full flex items-center justify-between px-4 py-3 hover:bg-[var(--bg-tertiary)] transition-colors"
          >
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-purple-400" />
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                V23 — 11-Step Master Trading Checklist
              </h3>
              <span className="text-xs font-mono text-[var(--text-muted)] bg-[var(--bg-tertiary)] px-1.5 py-0.5 rounded">
                {data.checklist_v23.filter((c) => c.status === "passed").length}/{data.checklist_v23.length} passed
              </span>
            </div>
            {v23Open ? (
              <ChevronDown className="w-4 h-4 text-[var(--text-muted)]" />
            ) : (
              <ChevronRight className="w-4 h-4 text-[var(--text-muted)]" />
            )}
          </button>

          {v23Open && (
            <div className="px-4 pb-4 space-y-2 border-t border-[var(--border-primary)] pt-3">
              {data.checklist_v23.map((item) => (
                <div
                  key={item.step}
                  className={`flex items-start gap-3 px-3 py-2 rounded-lg border ${
                    item.status === "passed"
                      ? "bg-emerald-500/5 border-emerald-500/10"
                      : item.status === "failed"
                      ? "bg-red-500/5 border-red-500/10"
                      : "bg-amber-500/5 border-amber-500/10"
                  }`}
                >
                  <StatusIcon status={item.status} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-[var(--text-muted)]">S{item.step}</span>
                      <span className="text-xs font-medium text-[var(--text-primary)]">{item.name}</span>
                    </div>
                    <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{item.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ═══ SECTION F: Recent Outcomes (Resolved Signals) ═══ */}
      {(resolvedSignals.length > 0 || (storeStats && storeStats.resolved_count > 0)) && (
        <div className="glass-card rounded-xl overflow-hidden animate-fade-in">
          <button
            onClick={() => setOutcomesOpen(!outcomesOpen)}
            className="w-full flex items-center justify-between px-4 py-3 hover:bg-[var(--bg-tertiary)] transition-colors"
          >
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                Recent Outcomes
              </h3>
              {storeStats && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    {storeStats.tp_hits} TP
                  </span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">
                    {storeStats.sl_hits} SL
                  </span>
                  {storeStats.win_rate > 0 && (
                    <span className={`text-[10px] font-mono font-bold ${storeStats.win_rate >= 50 ? "text-emerald-400" : "text-red-400"}`}>
                      {storeStats.win_rate}% WR
                    </span>
                  )}
                </div>
              )}
            </div>
            <div className="flex items-center gap-2">
              {outcomesOpen && resolvedSignals.length > 0 && (
                <button
                  onClick={(e) => { e.stopPropagation(); handleClearAll(); }}
                  className="flex items-center gap-1 px-2 py-1 text-[10px] font-medium rounded-md bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition-colors"
                >
                  <Trash2 className="w-3 h-3" />
                  Clear All
                </button>
              )}
              {outcomesOpen ? (
                <ChevronDown className="w-4 h-4 text-[var(--text-muted)]" />
              ) : (
                <ChevronRight className="w-4 h-4 text-[var(--text-muted)]" />
              )}
            </div>
          </button>

          {outcomesOpen && resolvedSignals.length > 0 && (
            <div className="px-4 pb-4 border-t border-[var(--border-primary)] pt-3">
              <div className="space-y-2">
                {resolvedSignals.map((rs) => {
                  const isBull = rs.direction === "bullish";
                  const isWin = rs.status === "tp_hit";
                  const isLoss = rs.status === "sl_hit";
                  return (
                    <div
                      key={rs.signal_id}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-lg border ${
                        isWin
                          ? "bg-emerald-500/5 border-emerald-500/15"
                          : isLoss
                          ? "bg-red-500/5 border-red-500/15"
                          : "bg-[var(--bg-tertiary)] border-[var(--border-primary)]"
                      }`}
                    >
                      {/* Direction */}
                      <div className={`w-6 h-6 rounded flex items-center justify-center ${
                        isBull ? "bg-emerald-500/10" : "bg-red-500/10"
                      }`}>
                        {isBull
                          ? <TrendingUp className="w-3 h-3 text-emerald-400" />
                          : <TrendingDown className="w-3 h-3 text-red-400" />
                        }
                      </div>

                      {/* Outcome badge */}
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        isWin
                          ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                          : isLoss
                          ? "bg-red-500/15 text-red-400 border border-red-500/30"
                          : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] border border-[var(--border-primary)]"
                      }`}>
                        {isWin ? "TP HIT" : isLoss ? "SL HIT" : "EXPIRED"}
                      </span>

                      {/* Price info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 text-[10px]">
                          <span className="text-[var(--text-muted)]">
                            Entry <span className="font-mono text-[var(--text-secondary)]">${rs.entry_price.toLocaleString()}</span>
                          </span>
                          <span className="text-[var(--text-muted)]">→</span>
                          <span className={isWin ? "text-emerald-400" : isLoss ? "text-red-400" : "text-[var(--text-muted)]"}>
                            {isWin ? `TP $${rs.take_profit.toLocaleString()}` : isLoss ? `SL $${rs.stop_loss.toLocaleString()}` : "Expired"}
                          </span>
                        </div>
                        <div className="flex items-center gap-1 mt-0.5">
                          {rs.trading_styles.map((s) => {
                            const meta = TRADING_STYLES.find((ts) => ts.key === s);
                            return meta ? (
                              <span key={s} className={`px-1 py-0 rounded text-[8px] font-bold ${meta.color}`}>
                                {meta.label}
                              </span>
                            ) : null;
                          })}
                        </div>
                      </div>

                      {/* Grade + R:R */}
                      <div className="text-right">
                        <span className={`text-[10px] font-bold ${
                          gradeConfig[rs.grade]?.text || "text-[var(--text-muted)]"
                        }`}>
                          {rs.grade}
                        </span>
                        <span className="text-[10px] text-[var(--text-muted)] ml-1.5">
                          R:R {rs.risk_reward_ratio}
                        </span>
                      </div>

                      {/* Delete button */}
                      <button
                        onClick={() => handleDeleteOne(rs.signal_id)}
                        className="p-1 rounded hover:bg-red-500/15 text-[var(--text-muted)] hover:text-red-400 transition-colors"
                        title="Delete outcome"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  );
                })}
              </div>

              {resolvedSignals.length === 0 && (
                <p className="text-xs text-[var(--text-muted)] text-center py-4">
                  No resolved signals yet — signals will appear here after SL/TP is hit
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
