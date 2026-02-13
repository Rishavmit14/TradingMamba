"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  Wallet,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  XCircle,
  CheckCircle,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  Send,
  Settings,
  RotateCcw,
  Wifi,
  WifiOff,
} from "lucide-react";
import {
  fetchDemoAccount,
  fetchDemoTrades,
  fetchDemoEquity,
  fetchTelegramStatus,
  fetchLivePrice,
  takeDemoTrade,
  skipDemoTrade,
  closeDemoTrade,
  resetDemoAccount,
  updateDemoSettings,
  sendTelegramTest,
} from "@/lib/api";
import type {
  DemoAccount,
  DemoTrade,
  DemoEquityPoint,
  TelegramStatus,
  SignalGrade,
} from "@/lib/types";

// ── Helpers ──

function GradeBadge({ grade }: { grade: SignalGrade }) {
  const colors: Record<string, string> = {
    A: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    B: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    C: "bg-amber-500/20 text-amber-400 border-amber-500/30",
    D: "bg-red-500/20 text-red-400 border-red-500/30",
    M: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  };
  return (
    <span
      className={`px-1.5 py-0.5 text-[10px] font-bold rounded border ${colors[grade] || colors.D}`}
    >
      {grade === "M" ? "MAN" : grade}
    </span>
  );
}

function OutcomeBadge({ outcome }: { outcome: string | null }) {
  if (!outcome) return null;
  const map: Record<string, { color: string; label: string }> = {
    win: { color: "bg-emerald-500/20 text-emerald-400", label: "WIN" },
    loss: { color: "bg-red-500/20 text-red-400", label: "LOSS" },
    manual_close: { color: "bg-amber-500/20 text-amber-400", label: "MANUAL" },
  };
  const m = map[outcome] || { color: "bg-gray-500/20 text-gray-400", label: outcome.toUpperCase() };
  return (
    <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded ${m.color}`}>
      {m.label}
    </span>
  );
}

function StatCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
}) {
  return (
    <div className="glass-card rounded-xl p-4 flex-1 min-w-[130px]">
      <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] mb-1">
        {label}
      </div>
      <div className={`text-lg font-bold ${color || "text-[var(--text-primary)]"}`}>
        {value}
      </div>
      {sub && (
        <div className="text-[10px] text-[var(--text-muted)] mt-0.5">{sub}</div>
      )}
    </div>
  );
}

// ── Priority scoring (mirrors backend logic) ──

function computePriorityScore(t: DemoTrade): number {
  const gradeScores: Record<string, number> = { A: 100, B: 75, C: 50, D: 25 };
  const gradeS = gradeScores[t.grade] ?? 25;
  const confS = Math.min(t.confidence_score, 100);
  const rrS = Math.min((t.risk_reward_ratio / 3.0) * 100, 100);
  const confCountS = Math.min((t.confluences.length / 5.0) * 100, 100);

  let score = gradeS * 0.35 + confS * 0.25 + rrS * 0.20 + confCountS * 0.10;

  // Check for counter-trend / climax from confluences text (flags not on DemoTrade directly)
  // These are stored in signal_data but we don't have them on the trade object yet
  return Math.round(Math.max(0, Math.min(100, score)));
}

function PriorityBadge({ rank, score }: { rank: number; score: number }) {
  if (rank === 1) {
    return (
      <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
        #1 BEST
        <span className="text-amber-400/60">{score}</span>
      </span>
    );
  }
  return (
    <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-[var(--bg-tertiary)] text-[var(--text-muted)] border border-[var(--border-primary)]">
      #{rank} <span className="opacity-60">{score}</span>
    </span>
  );
}

// ── Main Component ──

export default function DemoTab() {
  const [account, setAccount] = useState<DemoAccount | null>(null);
  const [trades, setTrades] = useState<DemoTrade[]>([]);
  const [equity, setEquity] = useState<DemoEquityPoint[]>([]);
  const [telegram, setTelegram] = useState<TelegramStatus | null>(null);
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<number | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [riskInput, setRiskInput] = useState("");
  const pollRef = useRef<ReturnType<typeof setInterval>>();

  const fetchAll = useCallback(async () => {
    try {
      const [acct, allTrades, eq, tg, price] = await Promise.all([
        fetchDemoAccount(),
        fetchDemoTrades(undefined, 100),
        fetchDemoEquity(),
        fetchTelegramStatus(),
        fetchLivePrice().then((p) => p.price).catch(() => null),
      ]);
      setAccount(acct);
      setTrades(allTrades);
      setEquity(eq);
      setTelegram(tg);
      setLivePrice(price);
      if (acct && !riskInput) setRiskInput(String(acct.risk_per_trade_pct));
    } catch {
      // Silent fail — keep stale data
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
    pollRef.current = setInterval(fetchAll, 5000);
    return () => clearInterval(pollRef.current);
  }, [fetchAll]);

  // ── Actions ──

  const handleTake = async (tradeId: number) => {
    setActionLoading(tradeId);
    try {
      await takeDemoTrade(tradeId);
      await fetchAll();
    } catch {}
    setActionLoading(null);
  };

  const handleSkip = async (tradeId: number) => {
    setActionLoading(tradeId);
    try {
      await skipDemoTrade(tradeId);
      await fetchAll();
    } catch {}
    setActionLoading(null);
  };

  const handleClose = async (tradeId: number) => {
    setActionLoading(tradeId);
    try {
      await closeDemoTrade(tradeId);
      await fetchAll();
    } catch {}
    setActionLoading(null);
  };

  const handleReset = async () => {
    if (!confirm("Reset demo account? All trades will be cleared.")) return;
    await resetDemoAccount();
    setRiskInput("");
    await fetchAll();
  };

  const handleSaveSettings = async () => {
    const val = parseFloat(riskInput);
    if (isNaN(val) || val <= 0 || val > 10) return;
    await updateDemoSettings({ risk_per_trade_pct: val });
    await fetchAll();
    setShowSettings(false);
  };

  const handleTestTelegram = async () => {
    try {
      await sendTelegramTest();
    } catch {}
  };

  // ── Derived data ──

  const pendingTradesRaw = trades.filter((t) => t.status === "pending");
  // Sort by priority score (highest first) and attach rank
  const pendingScored = pendingTradesRaw
    .map((t) => ({ ...t, _score: computePriorityScore(t) }))
    .sort((a, b) => b._score - a._score);
  const pendingTrades = pendingScored;
  const openTrades = trades.filter((t) => t.status === "open");
  const closedTrades = trades.filter(
    (t) => t.status === "closed" || t.status === "skipped"
  );

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <RefreshCw className="w-6 h-6 animate-spin text-[var(--text-muted)]" />
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-4 md:p-6 space-y-5">
      {/* ── Account Summary Bar ── */}
      <div className="flex flex-wrap gap-3">
        <StatCard
          label="Balance"
          value={account ? `$${account.balance.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "$0"}
          sub={account ? `Initial: $${account.initial_balance.toLocaleString()}` : ""}
        />
        <StatCard
          label="P&L"
          value={
            account
              ? `${account.pnl_total >= 0 ? "+" : ""}$${account.pnl_total.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
              : "$0"
          }
          color={
            account
              ? account.pnl_total >= 0
                ? "text-emerald-400"
                : "text-red-400"
              : undefined
          }
          sub={
            account && account.initial_balance
              ? `${((account.pnl_total / account.initial_balance) * 100).toFixed(1)}%`
              : ""
          }
        />
        <StatCard
          label="Win Rate"
          value={account ? `${account.win_rate.toFixed(1)}%` : "0%"}
          sub={
            account
              ? `${account.wins}W / ${account.losses}L (${account.total_trades} total)`
              : ""
          }
          color={
            account
              ? account.win_rate >= 55
                ? "text-emerald-400"
                : account.win_rate >= 45
                  ? "text-amber-400"
                  : "text-red-400"
              : undefined
          }
        />
        <StatCard
          label="Open"
          value={String(openTrades.length)}
          sub={`${pendingTrades.length} pending`}
        />
        <div className="glass-card rounded-xl p-4 flex-1 min-w-[130px]">
          <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] mb-1">
            Telegram
          </div>
          <div className="flex items-center gap-1.5">
            {telegram?.connected ? (
              <>
                <Wifi className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-medium text-emerald-400">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-[var(--text-muted)]" />
                <span className="text-sm font-medium text-[var(--text-muted)]">Offline</span>
              </>
            )}
          </div>
          {telegram?.connected && telegram.bot_username && (
            <div className="text-[10px] text-[var(--text-muted)] mt-0.5">
              @{telegram.bot_username}
            </div>
          )}
        </div>
      </div>

      {/* ── Pending Signals ── */}
      {pendingTrades.length > 0 && (
        <section>
          <div className="flex items-center gap-2 mb-3">
            <Clock className="w-4 h-4 text-amber-400" />
            <h2 className="text-sm font-semibold text-[var(--text-primary)]">
              Pending Signals
            </h2>
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 font-medium">
              {pendingTrades.length}
            </span>
          </div>
          <div className="space-y-2">
            {pendingTrades.map((t, idx) => {
              const rank = idx + 1;
              const isBest = rank === 1 && pendingTrades.length > 1;
              return (
                <div
                  key={t.id}
                  className={`glass-card rounded-lg p-3 animate-[fadeIn_0.3s_ease-out] ${
                    isBest ? "ring-1 ring-amber-500/40" : ""
                  }`}
                >
                  {/* Top row: direction, grade, rank, entry info, actions */}
                  <div className="flex items-center gap-3">
                    {/* Rank badge (only when multiple pending) */}
                    {pendingTrades.length > 1 && (
                      <PriorityBadge rank={rank} score={t._score} />
                    )}

                    {/* Direction */}
                    <div
                      className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-bold ${
                        t.direction === "bullish"
                          ? "bg-emerald-500/20 text-emerald-400"
                          : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {t.direction === "bullish" ? (
                        <ArrowUpRight className="w-3 h-3" />
                      ) : (
                        <ArrowDownRight className="w-3 h-3" />
                      )}
                      {t.direction === "bullish" ? "LONG" : "SHORT"}
                    </div>

                    <GradeBadge grade={t.grade} />

                    <div className="flex-1 text-xs font-mono text-[var(--text-secondary)]">
                      <span className="text-[var(--text-muted)]">Entry:</span> $
                      {t.entry_price.toLocaleString()}{" "}
                      <span className="text-[var(--text-muted)] ml-2">R:R:</span>{" "}
                      {t.risk_reward_ratio.toFixed(1)}
                      <span className="text-[var(--text-muted)] ml-2">Conf:</span>{" "}
                      {t.confidence_score.toFixed(0)}%
                    </div>

                    <div className="text-[10px] text-[var(--text-muted)]">
                      {t.entry_method?.toUpperCase() || "—"}
                    </div>

                    {/* Action buttons */}
                    <button
                      onClick={() => handleTake(t.id)}
                      disabled={actionLoading === t.id}
                      className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-md bg-emerald-600 hover:bg-emerald-500 text-white transition-colors disabled:opacity-50"
                    >
                      <CheckCircle className="w-3 h-3" />
                      Take
                    </button>
                    <button
                      onClick={() => handleSkip(t.id)}
                      disabled={actionLoading === t.id}
                      className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-md bg-[var(--bg-tertiary)] hover:bg-[var(--bg-card)] text-[var(--text-muted)] transition-colors disabled:opacity-50"
                    >
                      <XCircle className="w-3 h-3" />
                      Skip
                    </button>
                  </div>

                  {/* Bottom row: SL/TP + confluences */}
                  <div className="flex items-center gap-4 mt-2 text-[10px] text-[var(--text-muted)]">
                    <span>
                      SL: <span className="text-red-400">${t.stop_loss.toLocaleString()}</span>
                    </span>
                    <span>
                      TP: <span className="text-emerald-400">${t.take_profit.toLocaleString()}</span>
                    </span>
                    {t.confluences.length > 0 && (
                      <span className="truncate max-w-[300px]">
                        {t.confluences.join(" · ")}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* ── Open Positions ── */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-4 h-4 text-blue-400" />
          <h2 className="text-sm font-semibold text-[var(--text-primary)]">
            Open Positions
          </h2>
          {openTrades.length > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-500/20 text-blue-400 font-medium">
              {openTrades.length}
            </span>
          )}
        </div>

        {openTrades.length === 0 ? (
          <div className="glass-card rounded-lg p-6 text-center text-[var(--text-muted)] text-xs">
            No open positions. Signals will appear when detected.
          </div>
        ) : (
          <div className="space-y-2">
            {openTrades.map((t) => {
              const isLong = t.direction === "bullish";
              const unrealizedPnl = livePrice
                ? isLong
                  ? ((livePrice - t.entry_price) / t.entry_price) * 100
                  : ((t.entry_price - livePrice) / t.entry_price) * 100
                : null;
              const unrealizedUsd =
                livePrice && t.position_size_btc
                  ? (isLong ? 1 : -1) *
                    t.position_size_btc *
                    (livePrice - t.entry_price)
                  : null;

              return (
                <div
                  key={t.id}
                  className="glass-card rounded-lg p-3 animate-[fadeIn_0.3s_ease-out]"
                >
                  <div className="flex items-center gap-3">
                    {/* Direction + Grade */}
                    <div
                      className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-bold ${
                        isLong
                          ? "bg-emerald-500/20 text-emerald-400"
                          : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {isLong ? (
                        <ArrowUpRight className="w-3 h-3" />
                      ) : (
                        <ArrowDownRight className="w-3 h-3" />
                      )}
                      {isLong ? "LONG" : "SHORT"}
                    </div>
                    <GradeBadge grade={t.grade} />

                    {/* Prices */}
                    <div className="flex-1 grid grid-cols-3 gap-2 text-xs font-mono">
                      <div>
                        <span className="text-[var(--text-muted)]">Entry</span>
                        <div className="text-[var(--text-secondary)]">
                          ${t.entry_price.toLocaleString()}
                        </div>
                      </div>
                      <div>
                        <span className="text-red-400/70">SL</span>
                        <div className="text-red-400">
                          ${t.stop_loss.toLocaleString()}
                        </div>
                      </div>
                      <div>
                        <span className="text-emerald-400/70">TP</span>
                        <div className="text-emerald-400">
                          ${t.take_profit.toLocaleString()}
                        </div>
                      </div>
                    </div>

                    {/* Live P&L */}
                    {unrealizedPnl !== null && (
                      <div
                        className={`text-right min-w-[80px] ${
                          unrealizedPnl >= 0 ? "text-emerald-400" : "text-red-400"
                        }`}
                      >
                        <div className="text-sm font-bold">
                          {unrealizedPnl >= 0 ? "+" : ""}
                          {unrealizedPnl.toFixed(2)}%
                        </div>
                        {unrealizedUsd !== null && (
                          <div className="text-[10px] opacity-70">
                            {unrealizedUsd >= 0 ? "+" : ""}$
                            {unrealizedUsd.toFixed(2)}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Close button */}
                    <button
                      onClick={() => handleClose(t.id)}
                      disabled={actionLoading === t.id}
                      className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-md bg-amber-600/20 hover:bg-amber-600/40 text-amber-400 border border-amber-500/30 transition-colors disabled:opacity-50"
                    >
                      <XCircle className="w-3 h-3" />
                      Close
                    </button>
                  </div>

                  {/* Position size + source */}
                  <div className="flex items-center gap-4 mt-2 text-[10px] text-[var(--text-muted)]">
                    {t.position_size_btc && (
                      <span>Size: {t.position_size_btc.toFixed(6)} BTC (${t.position_size_usd?.toLocaleString()})</span>
                    )}
                    <span>R:R {t.risk_reward_ratio.toFixed(1)}</span>
                    <span>{t.entry_method?.toUpperCase() || "—"}</span>
                    <span className={`px-1 py-0.5 rounded text-[10px] font-bold ${
                      t.trade_source === "manual"
                        ? "bg-purple-500/15 text-purple-400"
                        : "bg-blue-500/15 text-blue-400"
                    }`}>
                      {t.trade_source === "manual" ? "Manual" : "Signal"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* ── Equity Curve ── */}
      {equity.length > 1 && (
        <section>
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-[var(--text-muted)]" />
            <h2 className="text-sm font-semibold text-[var(--text-primary)]">
              Equity Curve
            </h2>
          </div>
          <div className="glass-card rounded-lg p-4">
            <MiniEquityCurve data={equity} />
          </div>
        </section>
      )}

      {/* ── Trade History ── */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <Wallet className="w-4 h-4 text-[var(--text-muted)]" />
          <h2 className="text-sm font-semibold text-[var(--text-primary)]">
            Trade History
          </h2>
          <span className="text-[10px] text-[var(--text-muted)]">
            {closedTrades.length} trades
          </span>
        </div>

        {closedTrades.length === 0 ? (
          <div className="glass-card rounded-lg p-6 text-center text-[var(--text-muted)] text-xs">
            No completed trades yet.
          </div>
        ) : (
          <div className="glass-card rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-[var(--border-primary)] text-[var(--text-muted)]">
                    <th className="text-left p-2 pl-3">#</th>
                    <th className="text-left p-2">Time</th>
                    <th className="text-left p-2">Dir</th>
                    <th className="text-left p-2">Grade</th>
                    <th className="text-right p-2">Entry</th>
                    <th className="text-right p-2">Exit</th>
                    <th className="text-left p-2">Status</th>
                    <th className="text-right p-2">P&L</th>
                    <th className="text-left p-2 pr-3">Source</th>
                  </tr>
                </thead>
                <tbody>
                  {closedTrades.map((t, i) => (
                    <tr
                      key={t.id}
                      className="border-b border-[var(--border-primary)]/50 hover:bg-[var(--bg-tertiary)]/30 transition-colors"
                    >
                      <td className="p-2 pl-3 text-[var(--text-muted)] font-mono">
                        {closedTrades.length - i}
                      </td>
                      <td className="p-2 text-[var(--text-muted)] font-mono">
                        {t.closed_at
                          ? new Date(t.closed_at + "Z").toLocaleString("en-US", {
                              month: "short",
                              day: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            })
                          : "—"}
                      </td>
                      <td className="p-2">
                        <span
                          className={`flex items-center gap-0.5 ${
                            t.direction === "bullish"
                              ? "text-emerald-400"
                              : "text-red-400"
                          }`}
                        >
                          {t.direction === "bullish" ? (
                            <TrendingUp className="w-3 h-3" />
                          ) : (
                            <TrendingDown className="w-3 h-3" />
                          )}
                          {t.direction === "bullish" ? "LONG" : "SHORT"}
                        </span>
                      </td>
                      <td className="p-2">
                        <GradeBadge grade={t.grade} />
                      </td>
                      <td className="p-2 text-right font-mono text-[var(--text-secondary)]">
                        ${t.entry_price.toLocaleString()}
                      </td>
                      <td className="p-2 text-right font-mono text-[var(--text-secondary)]">
                        {t.exit_price ? `$${t.exit_price.toLocaleString()}` : "—"}
                      </td>
                      <td className="p-2">
                        {t.status === "skipped" ? (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-500/20 text-gray-400 font-medium">
                            SKIP
                          </span>
                        ) : (
                          <OutcomeBadge outcome={t.outcome} />
                        )}
                      </td>
                      <td
                        className={`p-2 text-right font-mono font-medium ${
                          (t.pnl_usd || 0) >= 0
                            ? "text-emerald-400"
                            : "text-red-400"
                        }`}
                      >
                        {t.status === "skipped"
                          ? "—"
                          : t.pnl_usd != null
                            ? `${t.pnl_usd >= 0 ? "+" : ""}$${t.pnl_usd.toFixed(2)}`
                            : "—"}
                      </td>
                      <td className="p-2 pr-3">
                        <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded ${
                          t.trade_source === "manual"
                            ? "bg-purple-500/20 text-purple-400"
                            : "bg-blue-500/20 text-blue-400"
                        }`}>
                          {t.trade_source === "manual" ? "Manual" : "Signal"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {/* ── Settings ── */}
      <section>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
        >
          <Settings className="w-3.5 h-3.5" />
          Settings
        </button>

        {showSettings && (
          <div className="glass-card rounded-lg p-4 mt-2 space-y-3 animate-[fadeIn_0.2s_ease-out]">
            <div className="flex items-center gap-3">
              <label className="text-xs text-[var(--text-muted)]">Risk per trade:</label>
              <input
                type="number"
                value={riskInput}
                onChange={(e) => setRiskInput(e.target.value)}
                min={0.1}
                max={10}
                step={0.1}
                className="w-20 px-2 py-1 text-xs rounded bg-[var(--bg-tertiary)] border border-[var(--border-primary)] text-[var(--text-primary)] focus:outline-none focus:border-blue-500"
              />
              <span className="text-xs text-[var(--text-muted)]">%</span>
              <button
                onClick={handleSaveSettings}
                className="px-2 py-1 text-xs rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors"
              >
                Save
              </button>
            </div>

            <div className="flex items-center gap-3">
              {telegram?.connected && (
                <button
                  onClick={handleTestTelegram}
                  className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-[var(--bg-tertiary)] hover:bg-[var(--bg-card)] text-[var(--text-muted)] border border-[var(--border-primary)] transition-colors"
                >
                  <Send className="w-3 h-3" />
                  Test Telegram
                </button>
              )}
              <button
                onClick={handleReset}
                className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/30 transition-colors"
              >
                <RotateCcw className="w-3 h-3" />
                Reset Account
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

// ── Mini Equity Curve (SVG-based, no external chart lib) ──

function MiniEquityCurve({ data }: { data: DemoEquityPoint[] }) {
  if (data.length < 2) return null;

  const balances = data.map((d) => d.balance);
  const minB = Math.min(...balances);
  const maxB = Math.max(...balances);
  const range = maxB - minB || 1;

  const w = 600;
  const h = 120;
  const padX = 0;
  const padY = 8;

  const points = data.map((d, i) => {
    const x = padX + (i / (data.length - 1)) * (w - 2 * padX);
    const y = padY + (1 - (d.balance - minB) / range) * (h - 2 * padY);
    return `${x},${y}`;
  });

  const linePoints = points.join(" ");
  const areaPoints = `${padX},${h} ${linePoints} ${w - padX},${h}`;

  const lastBalance = balances[balances.length - 1];
  const firstBalance = balances[0];
  const isPositive = lastBalance >= firstBalance;
  const strokeColor = isPositive ? "#10b981" : "#ef4444";
  const fillColor = isPositive ? "#10b98115" : "#ef444415";

  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-24">
        <polygon points={areaPoints} fill={fillColor} />
        <polyline
          points={linePoints}
          fill="none"
          stroke={strokeColor}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <div className="flex justify-between text-[10px] text-[var(--text-muted)] mt-1 px-1">
        <span>${firstBalance.toLocaleString()}</span>
        <span className={isPositive ? "text-emerald-400" : "text-red-400"}>
          ${lastBalance.toLocaleString()} ({isPositive ? "+" : ""}
          {(((lastBalance - firstBalance) / firstBalance) * 100).toFixed(1)}%)
        </span>
      </div>
    </div>
  );
}
