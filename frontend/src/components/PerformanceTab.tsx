"use client";

import { TrendingUp, TrendingDown, CheckCircle, XCircle, Timer, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { BacktestResult, GradeBreakdown, ConfluenceEdge } from "@/lib/types";

interface PerformanceTabProps {
  result: BacktestResult | null;
}

function SectionHeader({ title, sub }: { title: string; sub?: string }) {
  return (
    <div className="mb-3">
      <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">{title}</h3>
      {sub && <p className="text-xs text-[var(--text-muted)] mt-0.5">{sub}</p>}
    </div>
  );
}

function RecommendBadge({ rec }: { rec: string }) {
  if (rec === "TRADE") {
    return (
      <span className="inline-flex items-center gap-1 text-xs font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-md">
        <CheckCircle className="w-3 h-3" /> TRADE
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs font-bold text-red-400 bg-red-500/10 border border-red-500/20 px-2 py-0.5 rounded-md">
      <XCircle className="w-3 h-3" /> SKIP
    </span>
  );
}

function EdgeBar({ edge }: { edge: number }) {
  const absEdge = Math.min(Math.abs(edge), 50);
  const positive = edge >= 0;
  return (
    <div className="flex items-center gap-2 w-32">
      <div className="flex-1 h-2 bg-[var(--bg-tertiary)] rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${positive ? "bg-emerald-500" : "bg-red-500"}`}
          style={{ width: `${(absEdge / 50) * 100}%` }}
        />
      </div>
      <span className={`text-xs font-mono font-medium ${positive ? "text-emerald-400" : "text-red-400"}`}>
        {edge >= 0 ? "+" : ""}{edge.toFixed(1)}%
      </span>
    </div>
  );
}

function WinRateBar({ rate, label }: { rate: number; label?: string }) {
  return (
    <div className="flex items-center gap-2">
      {label && <span className="text-xs text-[var(--text-muted)] w-12 text-right">{label}</span>}
      <div className="flex-1 h-3 bg-[var(--bg-tertiary)] rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${rate >= 55 ? "bg-emerald-500" : rate >= 45 ? "bg-amber-500" : "bg-red-500"}`}
          style={{ width: `${Math.min(rate, 100)}%` }}
        />
      </div>
      <span className={`text-xs font-mono font-medium w-12 ${rate >= 55 ? "text-emerald-400" : rate >= 45 ? "text-amber-400" : "text-red-400"}`}>
        {rate.toFixed(1)}%
      </span>
    </div>
  );
}

export default function PerformanceTab({ result }: PerformanceTabProps) {
  if (!result) {
    return (
      <div className="h-full overflow-y-auto">
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-500/20 to-pink-400/20 border border-purple-500/20 flex items-center justify-center mx-auto mb-6">
              <TrendingUp className="w-8 h-8 text-purple-400" />
            </div>
            <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-2">No Performance Data</h3>
            <p className="text-sm text-[var(--text-muted)]">
              Run a backtest first to see performance analytics.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const sortedConfluences = [...result.by_confluence].sort((a, b) => b.edge - a.edge);
  const entryMethods = Object.entries(result.by_entry_method);
  const sessions = Object.entries(result.by_session);

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-6xl mx-auto p-6 space-y-6">
        {/* Overview bar */}
        <div className="glass-card rounded-xl p-4 flex items-center justify-between flex-wrap gap-3">
          <div>
            <div className="text-xs text-[var(--text-muted)]">Backtest Period</div>
            <div className="text-sm font-mono text-[var(--text-primary)]">{result.start_date} to {result.end_date}</div>
          </div>
          <div className="flex gap-6">
            <div className="text-center">
              <div className="text-xs text-[var(--text-muted)]">Trades</div>
              <div className="text-lg font-bold font-mono text-[var(--text-primary)]">{result.total_trades}</div>
            </div>
            <div className="text-center">
              <div className="text-xs text-[var(--text-muted)]">Win Rate</div>
              <div className={`text-lg font-bold font-mono ${result.win_rate >= 55 ? "text-emerald-400" : result.win_rate >= 45 ? "text-amber-400" : "text-red-400"}`}>
                {result.win_rate}%
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-[var(--text-muted)]">P&L</div>
              <div className={`text-lg font-bold font-mono ${result.total_pnl_pct >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                {result.total_pnl_pct >= 0 ? "+" : ""}{result.total_pnl_pct}%
              </div>
            </div>
          </div>
        </div>

        {/* Grade Breakdown */}
        <div className="glass-card rounded-xl p-5">
          <SectionHeader title="Grade Breakdown" sub="Win rate and P&L by signal grade — identifies which grades to trade and which to skip" />
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-[var(--text-muted)] border-b border-[var(--border-primary)]">
                  <th className="text-left py-2 px-2">Grade</th>
                  <th className="text-right py-2 px-2">Trades</th>
                  <th className="text-right py-2 px-2">Wins</th>
                  <th className="text-right py-2 px-2">Losses</th>
                  <th className="text-right py-2 px-2">Timeouts</th>
                  <th className="text-left py-2 px-3 w-48">Win Rate</th>
                  <th className="text-right py-2 px-2">Avg P&L</th>
                  <th className="text-left py-2 px-2">Action</th>
                </tr>
              </thead>
              <tbody>
                {result.by_grade.map((g) => (
                  <tr key={g.grade} className="border-b border-[var(--border-primary)]/50 hover:bg-[var(--bg-tertiary)]/50">
                    <td className="py-2.5 px-2">
                      <span className={`inline-flex items-center text-xs font-bold px-2.5 py-1 rounded-md border ${
                        g.grade === "A" ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/20" :
                        g.grade === "B" ? "text-blue-400 bg-blue-500/10 border-blue-500/20" :
                        g.grade === "C" ? "text-amber-400 bg-amber-500/10 border-amber-500/20" :
                        "text-red-400 bg-red-500/10 border-red-500/20"
                      }`}>
                        {g.grade}
                      </span>
                    </td>
                    <td className="py-2.5 px-2 text-right font-mono">{g.trades}</td>
                    <td className="py-2.5 px-2 text-right font-mono text-emerald-400">{g.wins}</td>
                    <td className="py-2.5 px-2 text-right font-mono text-red-400">{g.losses}</td>
                    <td className="py-2.5 px-2 text-right font-mono text-[var(--text-muted)]">{g.timeouts}</td>
                    <td className="py-2.5 px-3">
                      <WinRateBar rate={g.win_rate} />
                    </td>
                    <td className={`py-2.5 px-2 text-right font-mono font-medium ${g.avg_pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {g.avg_pnl >= 0 ? "+" : ""}{g.avg_pnl.toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-2">
                      <RecommendBadge rec={g.recommendation} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Confluence Edge Analysis */}
        <div className="glass-card rounded-xl p-5">
          <SectionHeader title="Confluence Edge Analysis" sub="Win rate when each confluence is present vs absent — higher edge = stronger signal filter" />
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-[var(--text-muted)] border-b border-[var(--border-primary)]">
                  <th className="text-left py-2 px-2">Confluence</th>
                  <th className="text-right py-2 px-2">Present WR</th>
                  <th className="text-right py-2 px-2">Absent WR</th>
                  <th className="text-left py-2 px-2 w-40">Edge</th>
                  <th className="text-right py-2 px-2">Trades (present)</th>
                  <th className="text-right py-2 px-2">Trades (absent)</th>
                </tr>
              </thead>
              <tbody>
                {sortedConfluences.map((c) => (
                  <tr key={c.name} className="border-b border-[var(--border-primary)]/50 hover:bg-[var(--bg-tertiary)]/50">
                    <td className="py-2.5 px-2 font-medium text-[var(--text-primary)]">{c.name}</td>
                    <td className={`py-2.5 px-2 text-right font-mono ${c.present_wr >= 55 ? "text-emerald-400" : "text-[var(--text-secondary)]"}`}>
                      {c.present_wr.toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-2 text-right font-mono text-[var(--text-muted)]">
                      {c.absent_wr.toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-2">
                      <EdgeBar edge={c.edge} />
                    </td>
                    <td className="py-2.5 px-2 text-right font-mono text-[var(--text-muted)]">{c.present_count}</td>
                    <td className="py-2.5 px-2 text-right font-mono text-[var(--text-muted)]">{c.absent_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Entry Method + Session — side by side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Entry Methods */}
          <div className="glass-card rounded-xl p-5">
            <SectionHeader title="Entry Methods" sub="Performance by entry type" />
            <div className="space-y-3">
              {entryMethods.map(([method, stats]) => (
                <div key={method} className="flex items-center justify-between">
                  <div className="flex items-center gap-2 flex-1">
                    <span className="text-xs font-medium text-[var(--text-primary)] uppercase w-28 font-mono">{method}</span>
                    <div className="flex-1">
                      <WinRateBar rate={stats.win_rate} />
                    </div>
                  </div>
                  <div className="flex items-center gap-3 ml-3">
                    <span className="text-xs text-[var(--text-muted)] font-mono">{stats.trades} trades</span>
                    <span className={`text-xs font-mono font-medium ${stats.avg_pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {stats.avg_pnl >= 0 ? "+" : ""}{stats.avg_pnl.toFixed(2)}%
                    </span>
                  </div>
                </div>
              ))}
              {entryMethods.length === 0 && (
                <p className="text-xs text-[var(--text-muted)]">No entry method data available.</p>
              )}
            </div>
          </div>

          {/* Session Analysis */}
          <div className="glass-card rounded-xl p-5">
            <SectionHeader title="Session Analysis" sub="Kill zone vs off-hours performance" />
            <div className="space-y-3">
              {sessions.map(([session, stats]) => (
                <div key={session} className="flex items-center justify-between">
                  <div className="flex items-center gap-2 flex-1">
                    <span className="text-xs font-medium text-[var(--text-primary)] w-28 capitalize">{session}</span>
                    <div className="flex-1">
                      <WinRateBar rate={stats.win_rate} />
                    </div>
                  </div>
                  <span className="text-xs text-[var(--text-muted)] font-mono ml-3">{stats.trades} trades</span>
                </div>
              ))}
              {sessions.length === 0 && (
                <p className="text-xs text-[var(--text-muted)]">No session data available.</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
