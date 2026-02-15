"use client";

import { useState, useEffect, useCallback, memo } from "react";
import {
  Loader2,
  RefreshCw,
  Gauge,
  Activity,
  Target,
  Shield,
  BarChart3,
  Waves,
  AlertTriangle,
  Zap,
  TrendingUp,
  TrendingDown,
  ArrowUpDown,
  BookOpen,
  Flame,
  Anchor,
  Eye,
  Info,
} from "lucide-react";
import { fetchQuantIntel, QuantIntelData } from "@/lib/api";

// ── Reusable: Stat Card ──

function StatCard({
  label,
  value,
  sub,
  color,
  icon: Icon,
  bg,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
  icon?: React.ElementType;
  bg?: string;
}) {
  return (
    <div className={`rounded-xl p-4 border border-[var(--border-primary)] ${bg || "bg-[var(--bg-card)]"}`}>
      <div className="flex items-center gap-1.5 mb-1.5">
        {Icon && <Icon className="w-3.5 h-3.5 text-[var(--text-muted)]" />}
        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider font-medium">{label}</div>
      </div>
      <div className={`text-xl font-bold font-mono ${color || "text-[var(--text-primary)]"}`}>
        {value}
      </div>
      {sub && <div className="text-[10px] text-[var(--text-muted)] mt-1">{sub}</div>}
    </div>
  );
}

// ── Fear & Greed Gauge ──

function FearGreedGauge({ value, classification }: { value: number; classification: string }) {
  const pct = Math.min(100, Math.max(0, value));
  const gaugeColor =
    value < 20 ? "from-red-600 to-red-500" :
    value < 40 ? "from-orange-600 to-orange-500" :
    value < 60 ? "from-yellow-600 to-yellow-500" :
    value < 80 ? "from-lime-600 to-lime-500" :
    "from-emerald-600 to-emerald-500";
  const textColor =
    value < 20 ? "text-red-400" :
    value < 40 ? "text-orange-400" :
    value < 60 ? "text-yellow-400" :
    value < 80 ? "text-lime-400" :
    "text-emerald-400";
  const bgColor =
    value < 20 ? "bg-red-500/8" :
    value < 40 ? "bg-orange-500/8" :
    value < 60 ? "bg-yellow-500/8" :
    value < 80 ? "bg-lime-500/8" :
    "bg-emerald-500/8";

  return (
    <div className={`rounded-xl p-5 border border-[var(--border-primary)] ${bgColor}`}>
      <div className="flex items-center gap-2 mb-4">
        <Gauge className="w-4 h-4 text-[var(--text-muted)]" />
        <span className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
          Fear & Greed Index
        </span>
        <span className={`ml-auto text-sm font-bold ${textColor}`}>
          {classification}
        </span>
      </div>

      <div className="flex items-center gap-6">
        {/* Large value */}
        <div className={`text-5xl font-bold font-mono ${textColor}`}>
          {value}
        </div>

        {/* Gauge bar */}
        <div className="flex-1">
          <div className="relative w-full h-3 rounded-full overflow-hidden bg-gradient-to-r from-red-500/20 via-yellow-500/20 to-emerald-500/20">
            <div
              className={`absolute top-0 left-0 h-full rounded-full bg-gradient-to-r ${gaugeColor} transition-all duration-700 shadow-lg`}
              style={{ width: `${pct}%` }}
            />
            <div
              className="absolute top-[-2px] h-[calc(100%+4px)] w-1.5 bg-white rounded-full shadow-lg shadow-white/40 transition-all duration-700"
              style={{ left: `calc(${pct}% - 3px)` }}
            />
          </div>
          <div className="flex justify-between mt-2">
            <span className="text-[10px] text-red-400/70">Extreme Fear</span>
            <span className="text-[10px] text-yellow-400/70">Neutral</span>
            <span className="text-[10px] text-emerald-400/70">Extreme Greed</span>
          </div>
        </div>
      </div>

      <div className="mt-3 text-[10px] text-[var(--text-muted)]">
        {value < 25 ? "Market extremely fearful — contrarian buy opportunity" :
         value < 45 ? "Cautious sentiment — potential accumulation zone" :
         value < 55 ? "Neutral market sentiment" :
         value < 75 ? "Greedy sentiment — watch for overextension" :
         "Extreme greed — contrarian sell risk, expect corrections"}
      </div>
    </div>
  );
}

// ── Deribit Options Section ──

function OptionsSection({ data }: { data: QuantIntelData["options_data"] }) {
  const pc = data.pc_ratio ?? 0;
  const maxPain = data.max_pain ?? 0;
  const maxPainDist = data.max_pain_distance_pct ?? 0;
  const gex = data.net_gex ?? 0;
  const skew = data.skew_25d;
  const spot = (data as any).spot_price ?? 0;

  const pcColor = pc > 1.2 ? "text-red-400" : pc < 0.8 ? "text-emerald-400" : "text-[var(--text-primary)]";
  const pcLabel = pc > 1.3 ? "Very Bearish" : pc > 1.1 ? "Bearish" : pc < 0.7 ? "Very Bullish" : pc < 0.9 ? "Bullish" : "Neutral";

  const gexColor = gex > 0 ? "text-emerald-400" : gex < -100 ? "text-red-400" : "text-[var(--text-secondary)]";
  const gexLabel = gex > 100 ? "Strong Dampening" : gex > 0 ? "Dampening" : gex < -100 ? "Strong Amplifying" : gex < 0 ? "Amplifying" : "Neutral";

  return (
    <div className="rounded-xl border border-cyan-500/15 bg-cyan-500/5 p-5">
      <div className="flex items-center gap-2 mb-4">
        <Target className="w-4 h-4 text-cyan-400" />
        <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">
          Deribit Options Analytics
        </span>
        {spot > 0 && (
          <span className="ml-auto text-[10px] text-[var(--text-muted)] font-mono">
            Spot: ${spot.toLocaleString()}
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        {/* P/C Ratio */}
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Put/Call Ratio</div>
          <div className={`text-2xl font-bold font-mono ${pcColor}`}>{pc.toFixed(3)}</div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">{pcLabel}</div>
          <div className="mt-2 w-full h-1.5 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${pc > 1 ? "bg-red-500" : "bg-emerald-500"}`}
              style={{ width: `${Math.min(100, (pc / 2) * 100)}%` }}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-[9px] text-emerald-400/60">Calls</span>
            <span className="text-[9px] text-[var(--text-muted)]">1.0</span>
            <span className="text-[9px] text-red-400/60">Puts</span>
          </div>
        </div>

        {/* Max Pain */}
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Max Pain</div>
          <div className="text-2xl font-bold font-mono text-cyan-300">
            ${maxPain > 0 ? (maxPain / 1000).toFixed(0) + "K" : "—"}
          </div>
          <div className={`text-[10px] mt-1 ${maxPainDist > 0 ? "text-emerald-400" : "text-red-400"}`}>
            {maxPainDist > 0 ? "+" : ""}{maxPainDist.toFixed(1)}% from spot
          </div>
          <div className="mt-2 text-[9px] text-[var(--text-muted)]">
            {Math.abs(maxPainDist) < 2 ? "Price near max pain — gravitational pull" :
             maxPainDist > 0 ? "Price below max pain — pull higher" :
             "Price above max pain — pull lower"}
          </div>
        </div>

        {/* GEX */}
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">
            Gamma Exposure (GEX)
          </div>
          <div className={`text-2xl font-bold font-mono ${gexColor}`}>
            {gex !== 0 ? `${(gex / 1e6).toFixed(1)}M` : "—"}
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">{gexLabel}</div>
          <div className="mt-2 text-[9px] text-[var(--text-muted)]">
            {gex > 0
              ? "Dealers sell rallies, buy dips — mean-reverting"
              : gex < 0
              ? "Dealers amplify moves — trending conditions"
              : "No gamma exposure data"}
          </div>
        </div>

        {/* 25-Delta Skew */}
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">25-Delta Skew</div>
          <div className={`text-2xl font-bold font-mono ${
            skew != null && skew > 3 ? "text-red-400" :
            skew != null && skew < -3 ? "text-emerald-400" :
            "text-[var(--text-secondary)]"
          }`}>
            {skew != null ? `${skew > 0 ? "+" : ""}${skew.toFixed(1)}` : "—"}
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">
            {skew != null && skew > 5 ? "Puts expensive — fear" :
             skew != null && skew > 2 ? "Mild put demand" :
             skew != null && skew < -5 ? "Calls expensive — greed" :
             skew != null && skew < -2 ? "Mild call demand" :
             "Balanced"}
          </div>
          <div className="mt-2 text-[9px] text-[var(--text-muted)]">
            Put IV - Call IV at 25-delta
          </div>
        </div>
      </div>

      {/* OI Breakdown */}
      {(data.total_put_oi || data.total_call_oi) && (
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-2">Open Interest Distribution</div>
          <div className="flex items-center gap-3">
            <div className="flex-1">
              <div className="flex justify-between text-[10px] mb-1">
                <span className="text-emerald-400">Calls: {((data.total_call_oi ?? 0) / 1000).toFixed(0)}K</span>
                <span className="text-red-400">Puts: {((data.total_put_oi ?? 0) / 1000).toFixed(0)}K</span>
              </div>
              <div className="w-full h-2.5 rounded-full overflow-hidden bg-red-500/20 flex">
                <div
                  className="h-full bg-emerald-500 rounded-l-full transition-all"
                  style={{ width: `${((data.total_call_oi ?? 0) / ((data.total_call_oi ?? 0) + (data.total_put_oi ?? 0))) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── COT Positioning Section ──

function COTSection({ data }: { data: Record<string, any> }) {
  const levNet = data.leveraged_net ?? 0;
  const pct = data.percentile ?? 50;
  const signal = data.signal ?? "neutral";
  const wow = data.wow_change ?? 0;
  const weeks = data.weeks_of_data ?? 0;
  const reportDate = data.report_date ?? "";

  const pctColor =
    pct < 20 ? "text-red-400" :
    pct > 80 ? "text-emerald-400" :
    pct < 40 ? "text-orange-400" :
    pct > 60 ? "text-lime-400" :
    "text-[var(--text-secondary)]";

  const signalLabel = signal.replace(/_/g, " ");

  return (
    <div className="rounded-xl border border-amber-500/15 bg-amber-500/5 p-5">
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-4 h-4 text-amber-400" />
        <span className="text-xs font-semibold text-amber-400 uppercase tracking-wider">
          CME COT — Institutional Positioning
        </span>
        <span className="ml-auto text-[10px] text-[var(--text-muted)] font-mono">
          Report: {reportDate}
        </span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Leveraged Net</div>
          <div className={`text-2xl font-bold font-mono ${levNet > 0 ? "text-emerald-400" : "text-red-400"}`}>
            {levNet > 0 ? "+" : ""}{(levNet / 1000).toFixed(1)}K
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">
            {levNet > 0 ? "Hedge funds net long" : "Hedge funds net short"}
          </div>
        </div>

        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Historical Percentile</div>
          <div className={`text-2xl font-bold font-mono ${pctColor}`}>
            P{pct.toFixed(0)}
          </div>
          <div className="mt-2 w-full h-1.5 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                pct < 20 ? "bg-red-500" :
                pct > 80 ? "bg-emerald-500" :
                "bg-amber-500"
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">{weeks} weeks of data</div>
        </div>

        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Week-over-Week</div>
          <div className={`text-2xl font-bold font-mono ${wow > 0 ? "text-emerald-400" : wow < 0 ? "text-red-400" : "text-[var(--text-secondary)]"}`}>
            {wow > 0 ? "+" : ""}{(wow / 1000).toFixed(1)}K
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">
            {wow > 0 ? "Institutions adding longs" : wow < 0 ? "Institutions reducing" : "No change"}
          </div>
        </div>

        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Signal</div>
          <div className={`text-lg font-bold capitalize ${
            signal.includes("extreme") ? "text-amber-400" :
            signal.includes("moderately") ? "text-[var(--text-secondary)]" :
            "text-[var(--text-muted)]"
          }`}>
            {signalLabel}
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">
            {pct < 20 ? "Contrarian bullish — extreme short positioning" :
             pct > 80 ? "Contrarian bearish — extreme long positioning" :
             "Normal range — no extreme signal"}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── On-Chain Flow Section ──

function OnChainSection({ data }: { data: Record<string, any> }) {
  const inflow = data.inflow_btc ?? 0;
  const outflow = data.outflow_btc ?? 0;
  const net = data.net_flow ?? 0;
  const avg7d = data.avg_7d_net ?? 0;
  const signal = data.signal ?? "neutral";
  const source = data.source ?? "unknown";

  const total = inflow + outflow;
  const inflowPct = total > 0 ? (inflow / total) * 100 : 50;

  return (
    <div className="rounded-xl border border-indigo-500/15 bg-indigo-500/5 p-5">
      <div className="flex items-center gap-2 mb-4">
        <Waves className="w-4 h-4 text-indigo-400" />
        <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">
          On-Chain Exchange Flows
        </span>
        <span className="ml-auto text-[10px] text-[var(--text-muted)]">
          Source: {source} | {data.date ?? ""}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Exchange Inflow</div>
          <div className="text-2xl font-bold font-mono text-red-400">
            {(inflow / 1000).toFixed(1)}K
          </div>
          <div className="text-[10px] text-red-400/70 mt-1">Selling pressure</div>
        </div>

        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Exchange Outflow</div>
          <div className="text-2xl font-bold font-mono text-emerald-400">
            {(outflow / 1000).toFixed(1)}K
          </div>
          <div className="text-[10px] text-emerald-400/70 mt-1">Accumulation</div>
        </div>

        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Net Flow</div>
          <div className={`text-2xl font-bold font-mono ${net > 0 ? "text-red-400" : "text-emerald-400"}`}>
            {net > 0 ? "+" : ""}{(net / 1000).toFixed(1)}K
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">
            7d avg: {avg7d > 0 ? "+" : ""}{(avg7d / 1000).toFixed(1)}K
          </div>
        </div>
      </div>

      {/* Flow bar */}
      <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
        <div className="flex justify-between text-[10px] mb-1.5">
          <span className="text-red-400">Inflow {inflowPct.toFixed(0)}%</span>
          <span className="text-emerald-400">Outflow {(100 - inflowPct).toFixed(0)}%</span>
        </div>
        <div className="w-full h-3 rounded-full overflow-hidden flex">
          <div className="h-full bg-red-500/60 transition-all" style={{ width: `${inflowPct}%` }} />
          <div className="h-full bg-emerald-500/60 transition-all" style={{ width: `${100 - inflowPct}%` }} />
        </div>
        <div className="mt-2 text-[10px] text-[var(--text-muted)]">
          {signal === "strong_inflow" ? "Heavy exchange deposits — significant selling pressure expected" :
           signal === "mild_inflow" ? "Moderate inflows — mild sell pressure" :
           signal === "strong_outflow" ? "Heavy withdrawals — strong accumulation signal" :
           signal === "mild_outflow" ? "Moderate outflows — mild accumulation" :
           "Balanced flows — no directional signal"}
        </div>
      </div>
    </div>
  );
}

// ── Cross-Exchange Funding Section ──

function FundingSection({ data }: { data: QuantIntelData["cross_exchange_funding"] }) {
  const exchanges = [
    { name: "Bybit", rate: data.bybit_rate ?? 0 },
    { name: "OKX", rate: data.okx_rate ?? 0 },
  ];
  const avg = data.avg_rate ?? 0;
  const spread = data.dispersion ?? 0;

  return (
    <div className="rounded-xl border border-[var(--border-primary)] bg-[var(--bg-card)] p-5">
      <div className="flex items-center gap-2 mb-4">
        <ArrowUpDown className="w-4 h-4 text-[var(--text-muted)]" />
        <span className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
          Cross-Exchange Funding Rates
        </span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {exchanges.map((ex) => (
          <div key={ex.name} className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
            <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">{ex.name}</div>
            <div className={`text-xl font-bold font-mono ${ex.rate > 0 ? "text-emerald-400" : ex.rate < 0 ? "text-red-400" : "text-[var(--text-secondary)]"}`}>
              {(ex.rate * 100).toFixed(4)}%
            </div>
            <div className="text-[10px] text-[var(--text-muted)] mt-1">
              {ex.rate > 0.0003 ? "Longs paying" : ex.rate < -0.0003 ? "Shorts paying" : "Neutral"}
            </div>
          </div>
        ))}

        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Average</div>
          <div className={`text-xl font-bold font-mono ${avg > 0 ? "text-emerald-400" : avg < 0 ? "text-red-400" : "text-[var(--text-secondary)]"}`}>
            {(avg * 100).toFixed(4)}%
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">Cross-exchange mean</div>
        </div>

        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Dispersion</div>
          <div className={`text-xl font-bold font-mono ${spread > 0.0002 ? "text-amber-400" : "text-[var(--text-secondary)]"}`}>
            {(spread * 100).toFixed(4)}%
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">
            {spread > 0.0005 ? "High divergence — arb building" : spread > 0.0002 ? "Moderate spread" : "Aligned"}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── L2 Order Book Section ──

function L2Section({ data }: { data: QuantIntelData["l2_depth"] }) {
  const bidWall = data.bid_wall_usd ?? 0;
  const askWall = data.ask_wall_usd ?? 0;
  const imbalance = data.imbalance ?? 0;
  const spread = data.spread_pct ?? 0;
  const bestBid = data.best_bid ?? 0;
  const bestAsk = data.best_ask ?? 0;

  const totalWall = bidWall + askWall;
  const bidPct = totalWall > 0 ? (bidWall / totalWall) * 100 : 50;

  return (
    <div className="rounded-xl border border-[var(--border-primary)] bg-[var(--bg-card)] p-5">
      <div className="flex items-center gap-2 mb-4">
        <BookOpen className="w-4 h-4 text-[var(--text-muted)]" />
        <span className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
          L2 Order Book Depth
        </span>
        {spread > 0 && (
          <span className="ml-auto text-[10px] text-[var(--text-muted)] font-mono">
            Spread: {(spread * 100).toFixed(3)}%
          </span>
        )}
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Bid Wall (±1%)</div>
          <div className="text-xl font-bold font-mono text-emerald-400">
            ${(bidWall / 1e6).toFixed(2)}M
          </div>
          {bestBid > 0 && <div className="text-[10px] text-[var(--text-muted)] mt-1">Best: ${bestBid.toLocaleString()}</div>}
        </div>

        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Ask Wall (±1%)</div>
          <div className="text-xl font-bold font-mono text-red-400">
            ${(askWall / 1e6).toFixed(2)}M
          </div>
          {bestAsk > 0 && <div className="text-[10px] text-[var(--text-muted)] mt-1">Best: ${bestAsk.toLocaleString()}</div>}
        </div>

        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Imbalance</div>
          <div className={`text-xl font-bold font-mono ${imbalance > 0.1 ? "text-emerald-400" : imbalance < -0.1 ? "text-red-400" : "text-[var(--text-secondary)]"}`}>
            {(imbalance * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">
            {imbalance > 0.15 ? "Strong bid support" : imbalance < -0.15 ? "Strong sell pressure" : "Balanced"}
          </div>
        </div>
      </div>

      {/* Imbalance bar */}
      <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
        <div className="flex justify-between text-[10px] mb-1.5">
          <span className="text-emerald-400">Bids {bidPct.toFixed(0)}%</span>
          <span className="text-red-400">Asks {(100 - bidPct).toFixed(0)}%</span>
        </div>
        <div className="w-full h-3 rounded-full overflow-hidden flex">
          <div className="h-full bg-emerald-500/60 transition-all" style={{ width: `${bidPct}%` }} />
          <div className="h-full bg-red-500/60 transition-all" style={{ width: `${100 - bidPct}%` }} />
        </div>
      </div>
    </div>
  );
}

// ── Liquidation Section ──

function LiquidationSection({ data }: { data: QuantIntelData["liquidation_ws"] }) {
  const count = data.event_count ?? 0;
  const recent = data.recent_30m ?? [];
  const running = data.running ?? false;

  const totalLong = recent.filter(e => e.side === "SELL").reduce((s, e) => s + e.qty_usd, 0);
  const totalShort = recent.filter(e => e.side === "BUY").reduce((s, e) => s + e.qty_usd, 0);

  return (
    <div className="rounded-xl border border-[var(--border-primary)] bg-[var(--bg-card)] p-5">
      <div className="flex items-center gap-2 mb-4">
        <Flame className="w-4 h-4 text-orange-400" />
        <span className="text-xs font-semibold text-orange-400 uppercase tracking-wider">
          Liquidation Monitor
        </span>
        <div className={`ml-auto flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] ${
          running ? "bg-emerald-500/15 text-emerald-400" : "bg-red-500/15 text-red-400"
        }`}>
          <div className={`w-1.5 h-1.5 rounded-full ${running ? "bg-emerald-400 animate-pulse" : "bg-red-400"}`} />
          {running ? "WebSocket Live" : "Offline"}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Total Events</div>
          <div className="text-2xl font-bold font-mono text-orange-400">{count}</div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">Since startup</div>
        </div>

        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Longs Liq (30m)</div>
          <div className="text-xl font-bold font-mono text-red-400">
            ${(totalLong / 1e6).toFixed(2)}M
          </div>
          <div className="text-[10px] text-red-400/70 mt-1">{recent.filter(e => e.side === "SELL").length} events</div>
        </div>

        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Shorts Liq (30m)</div>
          <div className="text-xl font-bold font-mono text-emerald-400">
            ${(totalShort / 1e6).toFixed(2)}M
          </div>
          <div className="text-[10px] text-emerald-400/70 mt-1">{recent.filter(e => e.side === "BUY").length} events</div>
        </div>
      </div>

      {/* Recent events table */}
      {recent.length > 0 && (
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-2">Recent Liquidations (30 min)</div>
          <div className="max-h-32 overflow-y-auto space-y-1">
            {recent.slice(0, 10).map((e, i) => (
              <div key={i} className="flex items-center justify-between text-[10px] py-0.5">
                <span className={e.side === "SELL" ? "text-red-400" : "text-emerald-400"}>
                  {e.side === "SELL" ? "Long Liquidated" : "Short Liquidated"}
                </span>
                <span className="font-mono text-[var(--text-secondary)]">${e.price.toLocaleString()}</span>
                <span className="font-mono text-[var(--text-muted)]">${(e.qty_usd / 1000).toFixed(1)}K</span>
                <span className="text-[var(--text-muted)]">
                  {new Date(e.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Whale Activity Section ──

function WhaleSection({ data }: { data: Record<string, any> }) {
  const txCount = data.whale_tx_count ?? 0;
  const totalBTC = data.total_whale_btc ?? 0;
  const recent = data.recent_whales ?? [];
  const signal = data.signal ?? "normal";

  if (txCount === 0) return null;

  return (
    <div className="rounded-xl border border-purple-500/15 bg-purple-500/5 p-5">
      <div className="flex items-center gap-2 mb-4">
        <Eye className="w-4 h-4 text-purple-400" />
        <span className="text-xs font-semibold text-purple-400 uppercase tracking-wider">
          Whale Activity Monitor
        </span>
        <span className={`ml-auto px-2 py-0.5 rounded-full text-[10px] font-medium ${
          signal === "high_activity" ? "bg-purple-500/20 text-purple-300" :
          signal === "active" ? "bg-purple-500/10 text-purple-400" :
          "bg-[var(--bg-tertiary)] text-[var(--text-muted)]"
        }`}>
          {signal === "high_activity" ? "High Activity" : signal === "active" ? "Active" : "Normal"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Large Transactions</div>
          <div className="text-2xl font-bold font-mono text-purple-400">{txCount}</div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">&gt;10 BTC confirmed + &gt;50 BTC mempool</div>
        </div>
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 text-center">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Total Volume</div>
          <div className="text-2xl font-bold font-mono text-purple-400">
            {totalBTC.toFixed(0)} BTC
          </div>
          <div className="text-[10px] text-[var(--text-muted)] mt-1">
            ~${(totalBTC * 97000 / 1e6).toFixed(1)}M USD
          </div>
        </div>
      </div>

      {/* Top whale txs */}
      {recent.length > 0 && (
        <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
          <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-2">Top Whale Transactions</div>
          <div className="max-h-28 overflow-y-auto space-y-1">
            {recent.slice(0, 8).map((tx: any, i: number) => (
              <div key={i} className="flex items-center justify-between text-[10px] py-0.5">
                <span className="font-mono text-purple-400">{tx.btc.toFixed(1)} BTC</span>
                <span className={`px-1.5 py-0.5 rounded text-[9px] ${
                  tx.status === "confirmed" ? "bg-emerald-500/10 text-emerald-400" : "bg-amber-500/10 text-amber-400"
                }`}>
                  {tx.status}
                </span>
                <span className="text-[var(--text-muted)] font-mono">{tx.txid}...</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Interpretation Guide ──

function HowToRead() {
  const items = [
    { icon: Gauge, color: "text-yellow-400", title: "Fear & Greed",
      text: "Contrarian indicator — extreme fear (<20) often signals buying opportunities, extreme greed (>80) signals corrections." },
    { icon: Target, color: "text-cyan-400", title: "Options (P/C, Max Pain, GEX)",
      text: "P/C >1.3 = excess put buying (fear). Max pain acts as price magnet near expiry. Positive GEX = dealers dampen moves (range-bound)." },
    { icon: BarChart3, color: "text-amber-400", title: "COT Positioning",
      text: "Weekly CFTC data shows hedge fund positioning. Extreme percentiles (<P20 or >P80) signal contrarian opportunities." },
    { icon: Waves, color: "text-indigo-400", title: "On-Chain Flows",
      text: "Exchange inflows = coins moving to exchanges (selling pressure). Outflows = withdrawal (accumulation). Net negative = bullish." },
    { icon: ArrowUpDown, color: "text-blue-400", title: "Cross-Exchange Funding",
      text: "Funding divergence between exchanges = arbitrage pressure building. High dispersion signals incoming convergence move." },
    { icon: Flame, color: "text-orange-400", title: "Liquidations",
      text: "Cascading liquidations in one direction = fuel for the opposite move. Large liquidation clusters near SL/TP zones change signal quality." },
    { icon: BookOpen, color: "text-slate-400", title: "L2 Order Book",
      text: "Bid/ask wall imbalance shows support/resistance strength. Large walls can absorb price moves or act as magnets." },
    { icon: Eye, color: "text-purple-400", title: "Whale Activity",
      text: "Large on-chain transactions indicate institutional positioning. Heavy whale deposits to exchanges = bearish, withdrawals = bullish." },
  ];

  return (
    <div className="rounded-xl border border-[var(--border-primary)] bg-[var(--bg-card)] p-5">
      <div className="flex items-center gap-2 mb-4">
        <Info className="w-4 h-4 text-[var(--text-muted)]" />
        <span className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
          How to Read Quant Intelligence
        </span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {items.map((item) => (
          <div key={item.title} className="flex gap-2.5">
            <item.icon className={`w-4 h-4 ${item.color} flex-shrink-0 mt-0.5`} />
            <div>
              <span className="text-xs font-medium text-[var(--text-primary)]">{item.title}</span>
              <p className="text-[10px] text-[var(--text-muted)] mt-0.5 leading-relaxed">{item.text}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main Component ──

function QuantAnalysisTabInner() {
  const [data, setData] = useState<QuantIntelData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const result = await fetchQuantIntel();
      setData(result);
      setError(null);
      setLastUpdated(new Date());
    } catch (err: any) {
      setError(err.message || "Failed to fetch quant data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60_000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Loader2 className="w-10 h-10 text-violet-500 animate-spin mx-auto mb-4" />
          <p className="text-sm text-[var(--text-secondary)]">Loading quant intelligence...</p>
          <p className="text-xs text-[var(--text-muted)] mt-1">Fetching 8 data sources in parallel</p>
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-sm">
          <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center mx-auto mb-4">
            <AlertTriangle className="w-6 h-6 text-red-400" />
          </div>
          <p className="text-sm text-red-400 mb-1">Failed to load quant data</p>
          <p className="text-xs text-[var(--text-muted)] mb-4">{error}</p>
          <button
            onClick={fetchData}
            className="px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-xs font-medium rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const fng = data.fear_greed || {};
  const opts = data.options_data || {};
  const cot = data.cot_data as Record<string, any> || {};
  const onchain = data.onchain_flow as Record<string, any> || {};
  const funding = data.cross_exchange_funding || {};
  const l2 = data.l2_depth || {};
  const liq = data.liquidation_ws || { running: false, event_count: 0, recent_30m: [] };
  const whales = (data as any).whale_transactions || {};

  const fngValue = (fng.value ?? 50) as number;
  const fngClass = (fng.classification ?? "N/A") as string;

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto p-6 space-y-5">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500/20 to-purple-500/20 border border-violet-500/20 flex items-center justify-center">
              <Zap className="w-4 h-4 text-violet-400" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-[var(--text-primary)]">
                Quant Intelligence Dashboard
              </h1>
              <p className="text-xs text-[var(--text-muted)]">
                8 data sources — institutional-grade market analysis
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {lastUpdated && (
              <span className="text-[10px] text-[var(--text-muted)] font-mono">
                Updated {lastUpdated.toLocaleTimeString()}
              </span>
            )}
            <button
              onClick={fetchData}
              className="p-1.5 rounded-md text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] transition-colors"
              title="Refresh now"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Data Source Status Bar */}
        <div className="flex items-center gap-2 flex-wrap">
          {[
            { name: "Fear & Greed", ok: fng.value != null },
            { name: "Options", ok: opts.pc_ratio != null },
            { name: "COT", ok: cot.leveraged_net != null },
            { name: "On-Chain", ok: onchain.net_flow != null },
            { name: "Funding", ok: funding.avg_rate != null },
            { name: "L2 Depth", ok: l2.best_bid != null },
            { name: "Liquidations", ok: liq.running },
            { name: "Whales", ok: whales.whale_tx_count > 0 },
          ].map(({ name, ok }) => (
            <div key={name} className={`flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium ${
              ok ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] border border-[var(--border-primary)]"
            }`}>
              <div className={`w-1.5 h-1.5 rounded-full ${ok ? "bg-emerald-400" : "bg-[var(--text-muted)]/30"}`} />
              {name}
            </div>
          ))}
        </div>

        {/* Fear & Greed Gauge */}
        <FearGreedGauge value={fngValue} classification={fngClass} />

        {/* Deribit Options */}
        {opts.pc_ratio != null && <OptionsSection data={opts} />}

        {/* COT + On-Chain side by side on large screens */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* COT Positioning */}
          {cot.leveraged_net != null && <COTSection data={cot} />}

          {/* On-Chain Flows */}
          {onchain.net_flow != null && <OnChainSection data={onchain} />}
        </div>

        {/* Funding + L2 Depth */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {funding.avg_rate != null && <FundingSection data={funding} />}
          {l2.best_bid != null && <L2Section data={l2} />}
        </div>

        {/* Liquidations + Whale Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <LiquidationSection data={liq} />
          <WhaleSection data={whales} />
        </div>

        {/* How to Read Guide */}
        <HowToRead />
      </div>
    </div>
  );
}

export default memo(QuantAnalysisTabInner);
