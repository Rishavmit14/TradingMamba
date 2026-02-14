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
  ArrowUpRight,
  Repeat2,
  GitBranch,
  Crosshair,
  Radio,
  Shield,
  Target,
  ArrowRight,
  RefreshCw,
} from "lucide-react";
import { fetchDetailedSignals } from "@/lib/api";
import {
  DetailedSignals,
  TradingSignal,
  ChecklistItem,
  ChecklistStatus,
  TrendState,
} from "@/lib/types";

// ── Entry Method Metadata (from V17/V22/V23/V24 knowledge) ──

interface EntryMethodMeta {
  key: string;
  name: string;
  fullName: string;
  description: string;
  source: string;
  icon: typeof ArrowUpRight;
  color: string;
}

const ENTRY_METHODS: EntryMethodMeta[] = [
  {
    key: "mss",
    name: "MSS",
    fullName: "Market Structure Shift",
    description:
      "Body closes beyond swept liquidity, expansion leg forms with FVG. Wait for IDM before entry.",
    source: "V17 — 80-88% accuracy",
    icon: ArrowUpRight,
    color: "#10b981",
  },
  {
    key: "sbc",
    name: "SBC",
    fullName: "Sweep Based Change",
    description:
      "Sweep one side + first candle body close on opposite side = trade direction. Works on major liquidity sweeps.",
    source: "V22 — Wyckoff-based",
    icon: Repeat2,
    color: "#3b82f6",
  },
  {
    key: "pullback_break",
    name: "Pullback Break",
    fullName: "Trend Continuation",
    description:
      "Valid BOS continuation. After confirmed BOS, enter on pullback to new zone. Same-TF: 1:1-1:2 R:R.",
    source: "V23 — Master Checklist",
    icon: GitBranch,
    color: "#a855f7",
  },
  {
    key: "scob",
    name: "SCOB",
    fullName: "Sweep Change of Bias",
    description:
      "Only valid at POI zone or after major liquidity sweep. Never standalone.",
    source: "V19/V23",
    icon: Crosshair,
    color: "#f59e0b",
  },
];

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
        <div className="flex items-center gap-2">
          {signal.is_counter_trend && (
            <span className="px-1.5 py-0.5 text-[10px] font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20 rounded">
              CT
            </span>
          )}
          <div className={`px-2 py-1 rounded-md text-xs font-bold border ${grade.bg} ${grade.text} ${grade.border}`}>
            Grade {signal.grade}
          </div>
        </div>
      </div>

      {/* Price levels */}
      <div className="grid grid-cols-3 gap-2 mb-3">
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
        <div className="bg-emerald-500/5 rounded-lg px-2.5 py-2">
          <div className="flex items-center gap-1 mb-1">
            <Target className="w-3 h-3 text-emerald-400/60" />
            <span className="text-[10px] text-emerald-400/60">TP</span>
          </div>
          <span className="text-xs font-mono font-medium text-emerald-400">
            ${signal.take_profit.toLocaleString()}
          </span>
        </div>
      </div>

      {/* R:R and Confidence */}
      <div className="flex items-center gap-4 mb-3">
        <span className="text-xs text-[var(--text-muted)]">
          R:R <span className="font-mono font-medium text-[var(--text-secondary)]">{signal.risk_reward_ratio}</span>
        </span>
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
    </div>
  );
}

// ── Main Component ──

export default function SignalsTab() {
  const [data, setData] = useState<DetailedSignals | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [v23Open, setV23Open] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    setError(null);
    try {
      const result = await fetchDetailedSignals();
      setData(result);
    } catch (err: any) {
      if (!silent) setError(err.message || "Failed to fetch signals");
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  // Initial fetch + 30s auto-refresh
  useEffect(() => {
    fetchData();
    intervalRef.current = setInterval(() => fetchData(true), 30_000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [fetchData]);

  // Group signals by entry method
  const signalsByMethod: Record<string, TradingSignal[]> = {};
  for (const method of ENTRY_METHODS) {
    signalsByMethod[method.key] = [];
  }
  signalsByMethod["other"] = [];
  if (data?.signals) {
    for (const sig of data.signals) {
      const key = sig.entry_method || "other";
      if (signalsByMethod[key]) {
        signalsByMethod[key].push(sig);
      } else {
        signalsByMethod["other"].push(sig);
      }
    }
  }

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

      {/* ═══ SECTION B: Signals by Entry Strategy ═══ */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-amber-400" />
          <h2 className="text-sm font-semibold text-[var(--text-primary)]">
            Active Signals by Entry Strategy
          </h2>
          {data && (
            <span className="text-xs font-mono text-[var(--text-muted)] bg-[var(--bg-tertiary)] px-1.5 py-0.5 rounded">
              {data.signals.length} total
            </span>
          )}
        </div>

        {ENTRY_METHODS.map((method) => {
          const signals = signalsByMethod[method.key] || [];
          const Icon = method.icon;

          return (
            <div key={method.key} className="glass-card rounded-xl overflow-hidden animate-fade-in">
              {/* Method Header */}
              <div className="px-4 py-3 border-b border-[var(--border-primary)]">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div
                      className="w-8 h-8 rounded-lg flex items-center justify-center"
                      style={{ backgroundColor: `${method.color}15` }}
                    >
                      <Icon className="w-4 h-4" style={{ color: method.color }} />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold" style={{ color: method.color }}>
                          {method.name}
                        </span>
                        <span className="text-xs text-[var(--text-secondary)]">
                          {method.fullName}
                        </span>
                      </div>
                      <p className="text-[10px] text-[var(--text-muted)] mt-0.5 max-w-lg">
                        {method.description}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[var(--text-muted)]">{method.source}</span>
                    <span
                      className="text-xs font-mono font-bold px-2 py-0.5 rounded"
                      style={{
                        color: signals.length > 0 ? method.color : "var(--text-muted)",
                        backgroundColor: signals.length > 0 ? `${method.color}15` : "var(--bg-tertiary)",
                      }}
                    >
                      {signals.length}
                    </span>
                  </div>
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
                    No active signals for this entry model
                  </p>
                )}
              </div>
            </div>
          );
        })}

        {/* Other / unclassified signals */}
        {signalsByMethod["other"]?.length > 0 && (
          <div className="glass-card rounded-xl p-4">
            <h3 className="text-xs font-semibold text-[var(--text-muted)] mb-3">Other Signals</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {signalsByMethod["other"].map((sig, i) => (
                <SignalCardFull key={i} signal={sig} />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ═══ SECTION C: V24 6-Rule Checklist ═══ */}
      {data?.checklist_v24 && (
        <div className="glass-card rounded-xl p-4 animate-fade-in">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-blue-400" />
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                V24 — 6-Rule Trading Framework
              </h3>
              <span className="text-[10px] text-[var(--text-muted)]">
                "If all 6 rules fulfilled, ANY concept works"
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

      {/* ═══ SECTION D: V23 Master Checklist (Collapsible) ═══ */}
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
    </div>
  );
}
