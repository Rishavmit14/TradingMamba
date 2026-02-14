"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Play, Loader2, TrendingUp, TrendingDown, Minus, CheckCircle, XCircle, Clock } from "lucide-react";
import { createChart, IChartApi, ISeriesApi } from "lightweight-charts";
import { BacktestResult, TradeRecord } from "@/lib/types";
import { runBacktest, fetchLatestBacktest } from "@/lib/api";

interface BacktestTabProps {
  result: BacktestResult | null;
  onResult: (r: BacktestResult) => void;
}

function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="glass-card rounded-xl p-4 flex-1 min-w-[140px]">
      <div className="text-xs text-[var(--text-muted)] mb-1">{label}</div>
      <div className={`text-xl font-bold font-mono ${color || "text-[var(--text-primary)]"}`}>{value}</div>
      {sub && <div className="text-xs text-[var(--text-muted)] mt-0.5">{sub}</div>}
    </div>
  );
}

function EquityCurve({ data }: { data: { timestamp: number; pnl: number }[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 250,
      layout: {
        background: { color: "transparent" },
        textColor: "#5c6370",
        fontSize: 11,
        fontFamily: "var(--font-mono), monospace",
      },
      grid: {
        vertLines: { color: "#1e223020" },
        horzLines: { color: "#1e223020" },
      },
      rightPriceScale: {
        borderColor: "#1e2230",
      },
      timeScale: {
        borderColor: "#1e2230",
        timeVisible: true,
      },
      crosshair: {
        horzLine: { color: "#3b82f640", style: 2 },
        vertLine: { color: "#3b82f640", style: 2 },
      },
    });

    const series = chart.addLineSeries({
      color: data[data.length - 1]?.pnl >= 0 ? "#10b981" : "#ef4444",
      lineWidth: 2,
      crosshairMarkerVisible: true,
      priceFormat: { type: "custom", formatter: (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%` },
    });

    // Deduplicate: lightweight-charts requires strictly increasing times.
    // Multiple trades can share a timestamp; keep the last (final cumulative PnL).
    const byTime = new Map<number, number>();
    for (const d of data) {
      byTime.set(Math.floor(d.timestamp / 1000), d.pnl);
    }
    const lineData = Array.from(byTime.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([t, v]) => ({ time: t as any, value: v }));
    series.setData(lineData);
    chart.timeScale().fitContent();
    chartRef.current = chart;

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [data]);

  return <div ref={containerRef} className="w-full" />;
}

function OutcomeBadge({ outcome }: { outcome: string }) {
  if (outcome === "win") {
    return (
      <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-md">
        <CheckCircle className="w-3 h-3" /> WIN
      </span>
    );
  }
  if (outcome === "loss") {
    return (
      <span className="inline-flex items-center gap-1 text-xs font-medium text-red-400 bg-red-500/10 border border-red-500/20 px-2 py-0.5 rounded-md">
        <XCircle className="w-3 h-3" /> LOSS
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs font-medium text-[var(--text-muted)] bg-[var(--bg-tertiary)] border border-[var(--border-primary)] px-2 py-0.5 rounded-md">
      <Clock className="w-3 h-3" /> TIMEOUT
    </span>
  );
}

function GradeBadge({ grade }: { grade: string }) {
  const colors: Record<string, string> = {
    A: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    B: "text-blue-400 bg-blue-500/10 border-blue-500/20",
    C: "text-amber-400 bg-amber-500/10 border-amber-500/20",
    D: "text-red-400 bg-red-500/10 border-red-500/20",
  };
  return (
    <span className={`inline-flex items-center text-xs font-bold px-2 py-0.5 rounded-md border ${colors[grade] || ""}`}>
      {grade}
    </span>
  );
}

function formatTimestamp(ts: number): string {
  if (!ts) return "—";
  const d = new Date(ts);
  return d.toLocaleDateString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric" }) + " " +
    d.toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hour12: false });
}

export default function BacktestTab({ result, onResult }: BacktestTabProps) {
  const [startDate, setStartDate] = useState("2024-06-01");
  const [endDate, setEndDate] = useState("2025-01-01");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Try loading latest backtest on mount
  useEffect(() => {
    if (!result) {
      fetchLatestBacktest().then((r) => {
        if (r) onResult(r);
      });
    }
  }, []);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const r = await runBacktest({ start_date: startDate, end_date: endDate });
      if ((r as any).error) {
        setError((r as any).error);
      } else {
        onResult(r);
      }
    } catch (err: any) {
      setError(err.message || "Backtest failed");
    } finally {
      setRunning(false);
    }
  }, [startDate, endDate, onResult]);

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-6xl mx-auto p-6 space-y-6">
        {/* Config Panel */}
        <div className="glass-card rounded-xl p-5">
          <h2 className="text-sm font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-4">
            Backtest Configuration
          </h2>
          <div className="flex items-end gap-4 flex-wrap">
            <div>
              <label className="text-xs text-[var(--text-muted)] block mb-1">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                disabled={running}
                className="bg-[var(--bg-tertiary)] border border-[var(--border-primary)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] font-mono focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="text-xs text-[var(--text-muted)] block mb-1">End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                disabled={running}
                className="bg-[var(--bg-tertiary)] border border-[var(--border-primary)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] font-mono focus:outline-none focus:border-blue-500"
              />
            </div>
            <button
              onClick={handleRun}
              disabled={running}
              className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-blue-600/25 disabled:opacity-50"
            >
              {running ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running Backtest...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Backtest
                </>
              )}
            </button>
          </div>
          {error && (
            <div className="mt-3 text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {error}
            </div>
          )}
          {running && (
            <div className="mt-4">
              <div className="text-xs text-[var(--text-muted)] mb-2">
                Running ~200 analysis windows across {startDate} to {endDate}... This may take 5-15 minutes.
              </div>
              <div className="w-full bg-[var(--bg-tertiary)] rounded-full h-2">
                <div className="bg-blue-500 h-2 rounded-full shimmer" style={{ width: "100%" }} />
              </div>
            </div>
          )}
        </div>

        {/* Results */}
        {result && !running && (
          <>
            {/* Stat Cards */}
            <div className="flex gap-3 flex-wrap">
              <StatCard label="Total Trades" value={String(result.total_trades)} sub={`${result.total_signals} signals detected`} />
              <StatCard
                label="Win Rate"
                value={`${result.win_rate}%`}
                sub={`${result.wins}W / ${result.losses}L / ${result.timeouts}T`}
                color={result.win_rate >= 55 ? "text-emerald-400" : result.win_rate >= 45 ? "text-amber-400" : "text-red-400"}
              />
              <StatCard
                label="Profit Factor"
                value={String(result.profit_factor)}
                color={result.profit_factor >= 1.5 ? "text-emerald-400" : result.profit_factor >= 1 ? "text-amber-400" : "text-red-400"}
              />
              <StatCard
                label="Total P&L"
                value={`${result.total_pnl_pct >= 0 ? "+" : ""}${result.total_pnl_pct}%`}
                color={result.total_pnl_pct >= 0 ? "text-emerald-400" : "text-red-400"}
              />
              <StatCard
                label="Max Drawdown"
                value={`-${result.max_drawdown_pct}%`}
                color="text-red-400"
              />
              <StatCard label="Avg R:R" value={String(result.avg_rr)} />
            </div>

            {/* Equity Curve */}
            {result.equity_curve.length > 0 && (
              <div className="glass-card rounded-xl p-4">
                <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">
                  Equity Curve (Cumulative P&L %)
                </h3>
                <EquityCurve data={result.equity_curve} />
              </div>
            )}

            {/* Trade List */}
            <div className="glass-card rounded-xl p-4">
              <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">
                Trade List ({result.trades.length} trades)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-[var(--text-muted)] border-b border-[var(--border-primary)]">
                      <th className="text-left py-2 px-2">#</th>
                      <th className="text-left py-2 px-2">Time</th>
                      <th className="text-left py-2 px-2">Dir</th>
                      <th className="text-left py-2 px-2">Grade</th>
                      <th className="text-right py-2 px-2">Entry</th>
                      <th className="text-right py-2 px-2">SL</th>
                      <th className="text-right py-2 px-2">TP</th>
                      <th className="text-left py-2 px-2">Outcome</th>
                      <th className="text-right py-2 px-2">P&L</th>
                      <th className="text-right py-2 px-2">Bars</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.trades.map((t, i) => (
                      <tr key={i} className="border-b border-[var(--border-primary)]/50 hover:bg-[var(--bg-tertiary)]/50">
                        <td className="py-2 px-2 text-[var(--text-muted)]">{i + 1}</td>
                        <td className="py-2 px-2 font-mono">{formatTimestamp(t.entry_timestamp)}</td>
                        <td className="py-2 px-2">
                          {t.direction === "bullish" ? (
                            <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
                          ) : (
                            <TrendingDown className="w-3.5 h-3.5 text-red-400" />
                          )}
                        </td>
                        <td className="py-2 px-2"><GradeBadge grade={t.grade} /></td>
                        <td className="py-2 px-2 text-right font-mono">${t.entry_price.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                        <td className="py-2 px-2 text-right font-mono text-red-400/60">${t.stop_loss.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                        <td className="py-2 px-2 text-right font-mono text-emerald-400/60">${t.take_profit.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                        <td className="py-2 px-2"><OutcomeBadge outcome={t.outcome} /></td>
                        <td className={`py-2 px-2 text-right font-mono font-medium ${t.pnl_pct >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                          {t.pnl_pct >= 0 ? "+" : ""}{t.pnl_pct.toFixed(2)}%
                        </td>
                        <td className="py-2 px-2 text-right font-mono text-[var(--text-muted)]">{t.bars_held}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {/* Empty state */}
        {!result && !running && (
          <div className="flex items-center justify-center h-96">
            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-cyan-400/20 border border-blue-500/20 flex items-center justify-center mx-auto mb-6">
                <Play className="w-8 h-8 text-blue-400" />
              </div>
              <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-2">No Backtest Results</h3>
              <p className="text-sm text-[var(--text-muted)]">
                Configure a date range and run a backtest to see results.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
