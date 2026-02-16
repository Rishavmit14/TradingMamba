"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Play,
  Loader2,
  TrendingUp,
  TrendingDown,
  Minus,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Zap,
  Shield,
  Activity,
  FlaskConical,
  Info,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { QuantBacktestResult, QuantAlertRecord } from "@/lib/types";
import { runQuantBacktest, fetchLatestQuantBacktest } from "@/lib/api";

// ── Stat Card ──

function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="glass-card rounded-xl p-4 flex-1 min-w-[130px]">
      <div className="text-xs text-[var(--text-muted)] mb-1">{label}</div>
      <div className={`text-xl font-bold font-mono ${color || "text-[var(--text-primary)]"}`}>{value}</div>
      {sub && <div className="text-xs text-[var(--text-muted)] mt-0.5">{sub}</div>}
    </div>
  );
}

// ── Reliability Gauge ──

function ReliabilityGauge({ score }: { score: number }) {
  const color = score >= 65 ? "text-emerald-400" : score >= 45 ? "text-amber-400" : "text-red-400";
  const bg = score >= 65 ? "bg-emerald-500/10" : score >= 45 ? "bg-amber-500/10" : "bg-red-500/10";
  const label = score >= 65 ? "RELIABLE" : score >= 45 ? "MODERATE" : "WEAK";

  return (
    <div className={`glass-card rounded-xl p-4 flex-1 min-w-[130px] ${bg}`}>
      <div className="text-xs text-[var(--text-muted)] mb-1">Reliability Score</div>
      <div className={`text-2xl font-bold font-mono ${color}`}>{score.toFixed(1)}</div>
      <div className={`text-xs font-semibold mt-0.5 ${color}`}>{label}</div>
    </div>
  );
}

// ── Severity Badge ──

function SeverityBadge({ severity }: { severity: string }) {
  const styles: Record<string, string> = {
    critical: "bg-red-500/15 text-red-400 border-red-500/30",
    warning: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    info: "bg-blue-500/15 text-blue-400 border-blue-500/30",
  };
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded border font-semibold uppercase ${styles[severity] || styles.info}`}>
      {severity}
    </span>
  );
}

// ── Direction Badge ──

function DirectionBadge({ direction }: { direction: string }) {
  if (direction === "bullish") {
    return (
      <span className="flex items-center gap-0.5 text-emerald-400 text-xs font-semibold">
        <TrendingUp size={12} /> UP
      </span>
    );
  }
  if (direction === "bearish") {
    return (
      <span className="flex items-center gap-0.5 text-red-400 text-xs font-semibold">
        <TrendingDown size={12} /> DOWN
      </span>
    );
  }
  return (
    <span className="flex items-center gap-0.5 text-yellow-400 text-xs font-semibold">
      <Minus size={12} /> —
    </span>
  );
}

// ── Correct Icon ──

function CorrectIcon({ value }: { value: boolean | null }) {
  if (value === true) return <CheckCircle size={14} className="text-emerald-400" />;
  if (value === false) return <XCircle size={14} className="text-red-400" />;
  return <Minus size={14} className="text-[var(--text-muted)]" />;
}

// ── Hit Rate Bar ──

function HitRateBar({ rate, label }: { rate: number; label: string }) {
  const color = rate >= 60 ? "bg-emerald-500" : rate >= 45 ? "bg-amber-500" : "bg-red-500";
  const textColor = rate >= 60 ? "text-emerald-400" : rate >= 45 ? "text-amber-400" : "text-red-400";
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-[var(--text-muted)] w-8">{label}</span>
      <div className="flex-1 h-2 bg-[var(--bg-primary)] rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${Math.min(rate, 100)}%` }} />
      </div>
      <span className={`text-xs font-mono font-semibold w-12 text-right ${textColor}`}>{rate.toFixed(1)}%</span>
    </div>
  );
}

// ── Format timestamp ──

function formatTs(ts: number): string {
  return new Date(ts).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

// ── Main Component ──

export default function QuantBacktestTab() {
  const [result, setResult] = useState<QuantBacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startDate, setStartDate] = useState("2025-11-01");
  const [endDate, setEndDate] = useState("2026-02-01");
  const [expandedTimeline, setExpandedTimeline] = useState(false);
  const [timelinePage, setTimelinePage] = useState(0);
  const TIMELINE_PAGE_SIZE = 50;

  // Load cached results on mount
  useEffect(() => {
    fetchLatestQuantBacktest().then(r => {
      if (r) setResult(r);
    }).catch(() => {});
  }, []);

  const handleRun = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await runQuantBacktest({ start_date: startDate, end_date: endDate });
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Backtest failed");
    } finally {
      setLoading(false);
    }
  }, [startDate, endDate]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {/* ── Config Bar ── */}
      <div className="glass-card rounded-xl p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <FlaskConical className="w-4 h-4 text-fuchsia-400" />
            <span className="text-sm font-semibold text-[var(--text-primary)]">Quant Algo Backtest</span>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-xs text-[var(--text-muted)]">From</label>
            <input
              type="date"
              value={startDate}
              onChange={e => setStartDate(e.target.value)}
              className="bg-[var(--bg-primary)] border border-[var(--border-primary)] rounded px-2 py-1 text-xs text-[var(--text-primary)] font-mono"
              disabled={loading}
            />
            <label className="text-xs text-[var(--text-muted)]">To</label>
            <input
              type="date"
              value={endDate}
              onChange={e => setEndDate(e.target.value)}
              className="bg-[var(--bg-primary)] border border-[var(--border-primary)] rounded px-2 py-1 text-xs text-[var(--text-primary)] font-mono"
              disabled={loading}
            />
          </div>

          <button
            onClick={handleRun}
            disabled={loading}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold bg-fuchsia-600 hover:bg-fuchsia-500 text-white disabled:opacity-50 transition-colors"
          >
            {loading ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            {loading ? "Running..." : "Run Backtest"}
          </button>

          <div className="flex items-center gap-1 text-xs text-[var(--text-muted)]">
            <Info size={12} />
            <span>7/10 algos tested (Options, Liquidation, OFI excluded — real-time only)</span>
          </div>
        </div>

        {error && (
          <div className="mt-3 text-xs text-red-400 flex items-center gap-1">
            <AlertTriangle size={12} />
            {error}
          </div>
        )}

        {loading && (
          <div className="mt-3 text-xs text-fuchsia-400 flex items-center gap-1.5">
            <Loader2 size={12} className="animate-spin" />
            <span>Downloading historical data and replaying algorithms... This may take 1-5 minutes.</span>
          </div>
        )}
      </div>

      {/* ── Results ── */}
      {result && (
        <>
          {/* Summary Cards */}
          <div className="flex flex-wrap gap-3">
            <StatCard
              label="Total Alerts"
              value={result.total_alerts.toString()}
              sub={`${result.directional_alerts} directional, ${result.non_directional_alerts} non-directional`}
            />
            <StatCard
              label="Hit Rate @ 4h"
              value={`${result.hit_rate_4h}%`}
              color={result.hit_rate_4h >= 55 ? "text-emerald-400" : result.hit_rate_4h >= 45 ? "text-amber-400" : "text-red-400"}
              sub="Directional accuracy"
            />
            <StatCard
              label="Hit Rate @ 24h"
              value={`${result.hit_rate_24h}%`}
              color={result.hit_rate_24h >= 55 ? "text-emerald-400" : result.hit_rate_24h >= 45 ? "text-amber-400" : "text-red-400"}
              sub="Directional accuracy"
            />
            <ReliabilityGauge score={result.reliability_score} />
            <StatCard
              label="Avg Move on Alert"
              value={`${result.avg_abs_move_24h.toFixed(2)}%`}
              sub="24h absolute move"
              color="text-blue-400"
            />
            <StatCard
              label="Combos Fired"
              value={Object.values(result.by_combo || {}).reduce((s, c) => s + c.count, 0).toString()}
              sub="Perfect Storm events"
              color="text-fuchsia-400"
            />
          </div>

          {/* Breakdown: 2-column layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Left: Per Alert Type */}
            <div className="glass-card rounded-xl p-4">
              <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3 flex items-center gap-1.5">
                <Activity size={14} className="text-cyan-400" />
                Alert Type Performance
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-[var(--text-muted)] border-b border-[var(--border-primary)]">
                      <th className="text-left py-2 pr-2">Alert</th>
                      <th className="text-center px-1">#</th>
                      <th className="text-center px-1">Sev</th>
                      <th className="text-center px-1">4h HR</th>
                      <th className="text-center px-1">24h HR</th>
                      <th className="text-right pl-1">Avg Move</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.by_alert_type || {})
                      .sort(([, a], [, b]) => b.count - a.count)
                      .map(([id, info]) => {
                        const hrColor = (hr: number) => hr >= 60 ? "text-emerald-400" : hr >= 45 ? "text-amber-400" : "text-red-400";
                        return (
                          <tr key={id} className="border-b border-[var(--border-primary)]/30 hover:bg-[var(--bg-card)]">
                            <td className="py-1.5 pr-2 text-[var(--text-secondary)] font-medium">{info.title}</td>
                            <td className="text-center px-1 font-mono">{info.count}</td>
                            <td className="text-center px-1"><SeverityBadge severity={info.severity} /></td>
                            <td className={`text-center px-1 font-mono font-semibold ${hrColor(info.hit_rate_4h)}`}>
                              {info.hit_rate_4h.toFixed(1)}%
                            </td>
                            <td className={`text-center px-1 font-mono font-semibold ${hrColor(info.hit_rate_24h)}`}>
                              {info.hit_rate_24h.toFixed(1)}%
                            </td>
                            <td className="text-right pl-1 font-mono text-[var(--text-muted)]">
                              {info.avg_move_24h.toFixed(2)}%
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>

              {/* Combo Performance */}
              {Object.keys(result.by_combo || {}).length > 0 && (
                <div className="mt-4">
                  <h4 className="text-xs font-semibold text-fuchsia-400 mb-2 flex items-center gap-1">
                    <Zap size={12} />
                    Perfect Storm Combos
                  </h4>
                  {Object.entries(result.by_combo || {}).map(([id, info]) => (
                    <div key={id} className="flex items-center justify-between py-1.5 border-b border-[var(--border-primary)]/20">
                      <span className="text-xs text-[var(--text-secondary)] font-medium">{info.title}</span>
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-mono text-[var(--text-muted)]">{info.count}x</span>
                        <span className={`text-xs font-mono font-semibold ${info.hit_rate_24h >= 60 ? "text-emerald-400" : "text-amber-400"}`}>
                          {info.hit_rate_24h.toFixed(1)}% @ 24h
                        </span>
                        <span className="text-xs font-mono text-blue-400">+{info.avg_favorable.toFixed(2)}% fav</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Right: By Severity + By Direction + Algos */}
            <div className="space-y-4">
              {/* By Severity */}
              <div className="glass-card rounded-xl p-4">
                <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3 flex items-center gap-1.5">
                  <Shield size={14} className="text-amber-400" />
                  By Severity
                </h3>
                <div className="space-y-2">
                  {(["critical", "warning", "info"] as const).map(sev => {
                    const info = result.by_severity?.[sev];
                    if (!info || info.count === 0) return null;
                    return (
                      <div key={sev} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <SeverityBadge severity={sev} />
                          <span className="text-xs font-mono text-[var(--text-muted)]">{info.count} alerts</span>
                        </div>
                        <div className="flex items-center gap-4">
                          <HitRateBar rate={info.hit_rate_4h} label="4h" />
                          <HitRateBar rate={info.hit_rate_24h} label="24h" />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* By Direction */}
              <div className="glass-card rounded-xl p-4">
                <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3 flex items-center gap-1.5">
                  <TrendingUp size={14} className="text-emerald-400" />
                  By Direction
                </h3>
                <div className="space-y-2">
                  {(["bullish", "bearish"] as const).map(dir => {
                    const info = result.by_direction?.[dir];
                    if (!info || info.count === 0) return null;
                    return (
                      <div key={dir} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <DirectionBadge direction={dir} />
                          <span className="text-xs font-mono text-[var(--text-muted)]">{info.count} alerts</span>
                        </div>
                        <div className="flex items-center gap-4">
                          <HitRateBar rate={info.hit_rate_4h} label="4h" />
                          <HitRateBar rate={info.hit_rate_24h} label="24h" />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Algos Active */}
              <div className="glass-card rounded-xl p-4">
                <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-2 flex items-center gap-1.5">
                  <FlaskConical size={14} className="text-fuchsia-400" />
                  Algorithms Tested
                </h3>
                <div className="flex flex-wrap gap-1.5">
                  {["VPIN", "Funding OU", "Kyle-Amihud", "Hurst", "Vol Regime", "Smart/Retail", "Bayesian"].map(name => (
                    <span key={name} className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      {name}
                    </span>
                  ))}
                  {["Options", "Liquidation", "OFI"].map(name => (
                    <span key={name} className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--bg-primary)] text-[var(--text-muted)] border border-[var(--border-primary)] line-through">
                      {name}
                    </span>
                  ))}
                </div>
                <p className="text-[10px] text-[var(--text-muted)] mt-2">
                  Excluded algos need real-time data (WebSocket/Deribit) — not available historically
                </p>
              </div>
            </div>
          </div>

          {/* Alert Timeline */}
          <div className="glass-card rounded-xl p-4">
            <button
              onClick={() => setExpandedTimeline(!expandedTimeline)}
              className="flex items-center gap-2 w-full text-left"
            >
              {expandedTimeline ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                Alert Timeline
              </h3>
              <span className="text-xs text-[var(--text-muted)]">
                — {result.timeline.length} events
              </span>
            </button>

            {expandedTimeline && result.timeline.length > 0 && (
              <div className="mt-3">
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-[var(--text-muted)] border-b border-[var(--border-primary)]">
                        <th className="text-left py-2 pr-2">Time</th>
                        <th className="text-left px-1">Alert</th>
                        <th className="text-center px-1">Dir</th>
                        <th className="text-center px-1">Sev</th>
                        <th className="text-right px-1">Price</th>
                        <th className="text-right px-1">1h</th>
                        <th className="text-right px-1">4h</th>
                        <th className="text-right px-1">24h</th>
                        <th className="text-center px-1">4h?</th>
                        <th className="text-center px-1">24h?</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.timeline
                        .slice(timelinePage * TIMELINE_PAGE_SIZE, (timelinePage + 1) * TIMELINE_PAGE_SIZE)
                        .map((row: QuantAlertRecord, idx: number) => {
                          const isCombo = row.combo !== null;
                          const moveColor = (m: number) => m >= 0.5 ? "text-emerald-400" : m <= -0.5 ? "text-red-400" : "text-[var(--text-muted)]";
                          return (
                            <tr
                              key={`${row.timestamp}-${row.alert_id}-${idx}`}
                              className={`border-b border-[var(--border-primary)]/20 hover:bg-[var(--bg-card)] ${isCombo ? "bg-fuchsia-500/5" : ""}`}
                            >
                              <td className="py-1.5 pr-2 font-mono text-[var(--text-muted)] whitespace-nowrap">{formatTs(row.timestamp)}</td>
                              <td className="px-1 text-[var(--text-secondary)] font-medium">
                                {row.title}
                                {isCombo && <span className="ml-1 text-[10px] text-fuchsia-400 font-semibold">COMBO</span>}
                              </td>
                              <td className="text-center px-1"><DirectionBadge direction={row.direction} /></td>
                              <td className="text-center px-1"><SeverityBadge severity={row.severity} /></td>
                              <td className="text-right px-1 font-mono text-[var(--text-secondary)]">${row.trigger_price.toLocaleString()}</td>
                              <td className={`text-right px-1 font-mono ${moveColor(row.move_1h_pct)}`}>{row.move_1h_pct >= 0 ? "+" : ""}{row.move_1h_pct.toFixed(2)}%</td>
                              <td className={`text-right px-1 font-mono ${moveColor(row.move_4h_pct)}`}>{row.move_4h_pct >= 0 ? "+" : ""}{row.move_4h_pct.toFixed(2)}%</td>
                              <td className={`text-right px-1 font-mono ${moveColor(row.move_24h_pct)}`}>{row.move_24h_pct >= 0 ? "+" : ""}{row.move_24h_pct.toFixed(2)}%</td>
                              <td className="text-center px-1"><CorrectIcon value={row.correct_4h} /></td>
                              <td className="text-center px-1"><CorrectIcon value={row.correct_24h} /></td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                {result.timeline.length > TIMELINE_PAGE_SIZE && (
                  <div className="flex items-center justify-between mt-3">
                    <span className="text-xs text-[var(--text-muted)]">
                      Showing {timelinePage * TIMELINE_PAGE_SIZE + 1}–{Math.min((timelinePage + 1) * TIMELINE_PAGE_SIZE, result.timeline.length)} of {result.timeline.length}
                    </span>
                    <div className="flex gap-1">
                      <button
                        onClick={() => setTimelinePage(p => Math.max(0, p - 1))}
                        disabled={timelinePage === 0}
                        className="px-2 py-1 text-xs rounded border border-[var(--border-primary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] disabled:opacity-30"
                      >
                        Prev
                      </button>
                      <button
                        onClick={() => setTimelinePage(p => p + 1)}
                        disabled={(timelinePage + 1) * TIMELINE_PAGE_SIZE >= result.timeline.length}
                        className="px-2 py-1 text-xs rounded border border-[var(--border-primary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] disabled:opacity-30"
                      >
                        Next
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {expandedTimeline && result.timeline.length === 0 && (
              <p className="mt-3 text-xs text-[var(--text-muted)]">No alerts triggered during the backtest period.</p>
            )}
          </div>

          {/* Meta info */}
          <div className="text-xs text-[var(--text-muted)] flex flex-wrap gap-4">
            <span>Period: {result.start_date} → {result.end_date}</span>
            <span>Steps: {result.total_hours}h</span>
            <span>Runtime: {result.elapsed_seconds}s</span>
            <span>Run: {result.run_timestamp}</span>
          </div>
        </>
      )}

      {/* Empty state */}
      {!result && !loading && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <FlaskConical size={48} className="text-[var(--text-muted)] mb-4 opacity-30" />
          <h3 className="text-sm font-semibold text-[var(--text-secondary)] mb-1">No Quant Backtest Results</h3>
          <p className="text-xs text-[var(--text-muted)] max-w-md">
            Run a backtest to replay 7/10 institutional algorithms over historical Binance data.
            Each alert threshold crossing is tested against actual price outcomes at T+1h, T+4h, T+24h.
          </p>
        </div>
      )}
    </div>
  );
}
