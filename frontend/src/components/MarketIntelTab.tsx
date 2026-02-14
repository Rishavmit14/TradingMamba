"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  Loader2,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  BarChart3,
  DollarSign,
  Percent,
  Users,
  Globe,
  Flame,
} from "lucide-react";
import { createChart, IChartApi } from "lightweight-charts";
import { MarketIntelData } from "@/lib/types";
import { fetchMarketIntel } from "@/lib/api";

// ── Stat Card ──

function StatCard({
  label,
  value,
  sub,
  color,
  icon: Icon,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
  icon?: React.ElementType;
}) {
  return (
    <div className="glass-card rounded-xl p-4 flex-1 min-w-[150px]">
      <div className="flex items-center gap-1.5 mb-1">
        {Icon && <Icon className="w-3.5 h-3.5 text-[var(--text-muted)]" />}
        <div className="text-xs text-[var(--text-muted)]">{label}</div>
      </div>
      <div className={`text-xl font-bold font-mono ${color || "text-[var(--text-primary)]"}`}>
        {value}
      </div>
      {sub && <div className="text-xs text-[var(--text-muted)] mt-0.5">{sub}</div>}
    </div>
  );
}

// ── Sentiment Gauge ──

function SentimentGauge({ data }: { data: MarketIntelData }) {
  // Compute sentiment from multiple signals (-5 to +5)
  let score = 0;

  // OI trend: rising = bullish conviction
  const oiHist = data.open_interest.history;
  if (oiHist.length >= 2) {
    const recent = oiHist[oiHist.length - 1].oi_usd;
    const earlier = oiHist[Math.max(0, oiHist.length - 12)].oi_usd;
    if (recent > earlier * 1.01) score += 1;
    else if (recent < earlier * 0.99) score -= 1;
  }

  // Funding rate: positive = crowded long (contrarian bearish), negative = crowded short (contrarian bullish)
  if (data.funding_rate.current > 0.0005) score -= 1;
  else if (data.funding_rate.current < -0.0005) score += 1;

  // Top trader ratio
  const topRatio = data.top_trader_ratio;
  if (topRatio.length > 0) {
    const latest = topRatio[topRatio.length - 1].ratio;
    if (latest > 1.2) score += 1;
    else if (latest < 0.8) score -= 1;
  }

  // Global (retail) ratio — contrarian
  const globalRatio = data.global_ratio;
  if (globalRatio.length > 0) {
    const latest = globalRatio[globalRatio.length - 1].ratio;
    if (latest > 1.5) score -= 1; // too many retail longs = bearish contrarian
    else if (latest < 0.7) score += 1;
  }

  // Taker volume
  const taker = data.taker_volume;
  if (taker.length > 0) {
    const latest = taker[taker.length - 1].ratio;
    if (latest > 1.1) score += 1;
    else if (latest < 0.9) score -= 1;
  }

  const maxScore = 5;
  const pct = ((score + maxScore) / (maxScore * 2)) * 100;
  const label =
    score >= 3 ? "Strong Bullish" :
    score >= 1 ? "Lean Bullish" :
    score <= -3 ? "Strong Bearish" :
    score <= -1 ? "Lean Bearish" :
    "Neutral";
  const labelColor =
    score >= 3 ? "text-emerald-400" :
    score >= 1 ? "text-emerald-400/70" :
    score <= -3 ? "text-red-400" :
    score <= -1 ? "text-red-400/70" :
    "text-[var(--text-muted)]";

  return (
    <div className="glass-card rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
          Composite Sentiment
        </h3>
        <span className={`text-sm font-bold ${labelColor}`}>{label}</span>
      </div>
      <div className="relative w-full h-4 rounded-full overflow-hidden bg-gradient-to-r from-red-500/20 via-[var(--bg-tertiary)] to-emerald-500/20">
        <div
          className="absolute top-0 h-full w-1 bg-white rounded-full shadow-lg shadow-white/30 transition-all duration-500"
          style={{ left: `calc(${pct}% - 2px)` }}
        />
      </div>
      <div className="flex justify-between mt-1.5">
        <span className="text-[10px] text-red-400/60">Bearish</span>
        <span className="text-[10px] text-[var(--text-muted)]">Neutral</span>
        <span className="text-[10px] text-emerald-400/60">Bullish</span>
      </div>
    </div>
  );
}

// ── Mini Chart Component (reusable) ──

function MiniChart({
  title,
  data,
  type = "line",
  color = "#3b82f6",
  negColor,
  formatValue,
}: {
  title: string;
  data: { time: number; value: number }[];
  type?: "line" | "histogram";
  color?: string;
  negColor?: string;
  formatValue?: (v: number) => string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 200,
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

    // Deduplicate timestamps (lightweight-charts needs strictly increasing)
    const byTime = new Map<number, number>();
    for (const d of data) {
      byTime.set(d.time, d.value);
    }
    const sorted = Array.from(byTime.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([t, v]) => ({ time: t as any, value: v }));

    if (type === "histogram") {
      const series = chart.addHistogramSeries({
        color,
        priceFormat: formatValue
          ? { type: "custom" as const, formatter: formatValue }
          : { type: "price" as const, precision: 6 },
      });
      if (negColor) {
        // Color each bar based on value
        const colored = sorted.map((d) => ({
          ...d,
          color: d.value >= 0 ? color : negColor,
        }));
        series.setData(colored);
      } else {
        series.setData(sorted);
      }
    } else {
      const series = chart.addLineSeries({
        color,
        lineWidth: 2,
        crosshairMarkerVisible: true,
        priceFormat: formatValue
          ? { type: "custom" as const, formatter: formatValue }
          : { type: "price" as const, precision: 4 },
      });
      series.setData(sorted);
    }

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
  }, [data, type, color, negColor, formatValue]);

  return (
    <div className="glass-card rounded-xl p-4">
      <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">
        {title}
      </h3>
      <div ref={containerRef} className="w-full" />
    </div>
  );
}

// ── Dual Line Chart (for L/S ratio comparison) ──

function DualLineChart({
  title,
  data1,
  data2,
  label1,
  label2,
  color1 = "#10b981",
  color2 = "#f59e0b",
}: {
  title: string;
  data1: { time: number; value: number }[];
  data2: { time: number; value: number }[];
  label1: string;
  label2: string;
  color1?: string;
  color2?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data1.length === 0) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 200,
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
      rightPriceScale: { borderColor: "#1e2230" },
      timeScale: { borderColor: "#1e2230", timeVisible: true },
      crosshair: {
        horzLine: { color: "#3b82f640", style: 2 },
        vertLine: { color: "#3b82f640", style: 2 },
      },
    });

    const dedup = (arr: { time: number; value: number }[]) => {
      const m = new Map<number, number>();
      for (const d of arr) m.set(d.time, d.value);
      return Array.from(m.entries())
        .sort((a, b) => a[0] - b[0])
        .map(([t, v]) => ({ time: t as any, value: v }));
    };

    const s1 = chart.addLineSeries({ color: color1, lineWidth: 2, title: label1 });
    s1.setData(dedup(data1));

    const s2 = chart.addLineSeries({ color: color2, lineWidth: 2, title: label2 });
    s2.setData(dedup(data2));

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
  }, [data1, data2, label1, label2, color1, color2]);

  return (
    <div className="glass-card rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
          {title}
        </h3>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-0.5 rounded" style={{ background: color1 }} />
            <span className="text-[10px] text-[var(--text-muted)]">{label1}</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-0.5 rounded" style={{ background: color2 }} />
            <span className="text-[10px] text-[var(--text-muted)]">{label2}</span>
          </div>
        </div>
      </div>
      <div ref={containerRef} className="w-full" />
    </div>
  );
}

// ── Taker Volume Stacked Chart ──

function TakerVolumeChart({ data }: { data: MarketIntelData["taker_volume"] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 200,
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
      rightPriceScale: { borderColor: "#1e2230" },
      timeScale: { borderColor: "#1e2230", timeVisible: true },
      crosshair: {
        horzLine: { color: "#3b82f640", style: 2 },
        vertLine: { color: "#3b82f640", style: 2 },
      },
    });

    const dedup = (arr: { time: number; value: number; color?: string }[]) => {
      const m = new Map<number, { value: number; color?: string }>();
      for (const d of arr) m.set(d.time, { value: d.value, color: d.color });
      return Array.from(m.entries())
        .sort((a, b) => a[0] - b[0])
        .map(([t, v]) => ({ time: t as any, ...v }));
    };

    // Net taker volume (buy - sell) as histogram
    const netData = data.map((d) => ({
      time: Math.floor(d.timestamp / 1000),
      value: d.buy_vol - d.sell_vol,
      color: d.buy_vol >= d.sell_vol ? "#10b981" : "#ef4444",
    }));

    const series = chart.addHistogramSeries({
      priceFormat: {
        type: "custom" as const,
        formatter: (v: number) => {
          const abs = Math.abs(v);
          if (abs >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
          if (abs >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
          return `${(v / 1e3).toFixed(0)}K`;
        },
      },
    });
    series.setData(dedup(netData));

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

  return (
    <div className="glass-card rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
          Taker Net Volume (Buy - Sell)
        </h3>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />
            <span className="text-[10px] text-[var(--text-muted)]">Buy dominant</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-sm bg-red-500" />
            <span className="text-[10px] text-[var(--text-muted)]">Sell dominant</span>
          </div>
        </div>
      </div>
      <div ref={containerRef} className="w-full" />
    </div>
  );
}

// ── Helper formatters ──

function formatUSD(v: number): string {
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  return `$${v.toLocaleString()}`;
}

function formatFunding(rate: number): string {
  return `${(rate * 100).toFixed(4)}%`;
}

function formatCountdown(targetMs: number): string {
  const diff = targetMs - Date.now();
  if (diff <= 0) return "Now";
  const h = Math.floor(diff / 3600000);
  const m = Math.floor((diff % 3600000) / 60000);
  return `${h}h ${m}m`;
}

// ── Main Component ──

export default function MarketIntelTab() {
  const [data, setData] = useState<MarketIntelData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval>>();

  const fetchData = useCallback(async () => {
    try {
      const result = await fetchMarketIntel();
      setData(result);
      setError(null);
      setLastUpdated(new Date());
    } catch (err: any) {
      setError(err.message || "Failed to fetch market intel");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    pollRef.current = setInterval(fetchData, 30_000);
    return () => clearInterval(pollRef.current);
  }, [fetchData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Loader2 className="w-10 h-10 text-blue-500 animate-spin mx-auto mb-4" />
          <p className="text-sm text-[var(--text-secondary)]">Loading market intelligence...</p>
          <p className="text-xs text-[var(--text-muted)] mt-1">Fetching Binance Futures data</p>
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-sm">
          <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center mx-auto mb-4">
            <BarChart3 className="w-6 h-6 text-red-400" />
          </div>
          <p className="text-sm text-red-400 mb-1">Failed to load data</p>
          <p className="text-xs text-[var(--text-muted)] mb-4">{error}</p>
          <button
            onClick={fetchData}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) return null;

  // Compute derived values
  const oiHist = data.open_interest.history;
  const oiChange24h =
    oiHist.length >= 24
      ? ((oiHist[oiHist.length - 1].oi_usd - oiHist[oiHist.length - 24].oi_usd) /
          oiHist[oiHist.length - 24].oi_usd) * 100
      : 0;

  const topLatest = data.top_trader_ratio.length > 0
    ? data.top_trader_ratio[data.top_trader_ratio.length - 1]
    : null;

  const globalLatest = data.global_ratio.length > 0
    ? data.global_ratio[data.global_ratio.length - 1]
    : null;

  const takerLatest = data.taker_volume.length > 0
    ? data.taker_volume[data.taker_volume.length - 1]
    : null;

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto p-6 space-y-5">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/20 flex items-center justify-center">
              <BarChart3 className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-[var(--text-primary)]">
                Market Intelligence
              </h1>
              <p className="text-xs text-[var(--text-muted)]">
                BTCUSDT Futures — Binance
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

        {/* ① Headline Stat Cards */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <StatCard
            label="Open Interest"
            value={formatUSD(data.open_interest.current_usd)}
            sub={`${data.open_interest.current.toFixed(0)} BTC`}
            icon={DollarSign}
          />
          <StatCard
            label="OI Change 24h"
            value={`${oiChange24h >= 0 ? "+" : ""}${oiChange24h.toFixed(2)}%`}
            sub={oiChange24h > 0 ? "Rising conviction" : oiChange24h < 0 ? "Declining interest" : "Flat"}
            color={oiChange24h > 0 ? "text-emerald-400" : oiChange24h < 0 ? "text-red-400" : undefined}
            icon={oiChange24h >= 0 ? TrendingUp : TrendingDown}
          />
          <StatCard
            label="Funding Rate"
            value={formatFunding(data.funding_rate.current)}
            sub={`Next: ${formatCountdown(data.funding_rate.next_funding_time)}`}
            color={
              data.funding_rate.current > 0.0001
                ? "text-emerald-400"
                : data.funding_rate.current < -0.0001
                ? "text-red-400"
                : undefined
            }
            icon={Percent}
          />
          <StatCard
            label="Mark Price"
            value={`$${data.funding_rate.mark_price.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            sub={`Premium ${data.premium_index.premium_pct >= 0 ? "+" : ""}${data.premium_index.premium_pct.toFixed(3)}%`}
            icon={DollarSign}
          />
          <StatCard
            label="Top Traders L/S"
            value={topLatest ? topLatest.ratio.toFixed(2) : "—"}
            sub={topLatest ? `${(topLatest.long_pct * 100).toFixed(1)}% L / ${(topLatest.short_pct * 100).toFixed(1)}% S` : undefined}
            color={
              topLatest && topLatest.ratio > 1.1
                ? "text-emerald-400"
                : topLatest && topLatest.ratio < 0.9
                ? "text-red-400"
                : undefined
            }
            icon={Users}
          />
          <StatCard
            label="Retail L/S"
            value={globalLatest ? globalLatest.ratio.toFixed(2) : "—"}
            sub={globalLatest ? `${(globalLatest.long_pct * 100).toFixed(1)}% L / ${(globalLatest.short_pct * 100).toFixed(1)}% S` : undefined}
            color={
              globalLatest && globalLatest.ratio > 1.3
                ? "text-amber-400"
                : globalLatest && globalLatest.ratio < 0.7
                ? "text-cyan-400"
                : undefined
            }
            icon={Globe}
          />
        </div>

        {/* ② Sentiment Gauge */}
        <SentimentGauge data={data} />

        {/* ③④ Charts Row 1: OI History + Funding Rate */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <MiniChart
            title="Open Interest (48h)"
            data={oiHist.map((d) => ({
              time: Math.floor(d.timestamp / 1000),
              value: d.oi_usd,
            }))}
            type="line"
            color={oiChange24h >= 0 ? "#10b981" : "#ef4444"}
            formatValue={(v: number) => formatUSD(v)}
          />
          <MiniChart
            title="Funding Rate History (Last 30)"
            data={data.funding_rate.history.map((d) => ({
              time: Math.floor(d.timestamp / 1000),
              value: d.rate,
            }))}
            type="histogram"
            color="#10b981"
            negColor="#ef4444"
            formatValue={(v: number) => `${(v * 100).toFixed(4)}%`}
          />
        </div>

        {/* ⑤⑥ Charts Row 2: L/S Ratios + Taker Volume */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <DualLineChart
            title="Long/Short Ratio (48h)"
            data1={data.top_trader_ratio.map((d) => ({
              time: Math.floor(d.timestamp / 1000),
              value: d.ratio,
            }))}
            data2={data.global_ratio.map((d) => ({
              time: Math.floor(d.timestamp / 1000),
              value: d.ratio,
            }))}
            label1="Top Traders"
            label2="Retail"
            color1="#3b82f6"
            color2="#f59e0b"
          />
          <TakerVolumeChart data={data.taker_volume} />
        </div>

        {/* Interpretation Guide */}
        <div className="glass-card rounded-xl p-4">
          <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">
            How to Read
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 text-xs text-[var(--text-secondary)]">
            <div className="flex gap-2">
              <DollarSign className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-medium text-[var(--text-primary)]">Open Interest</span> — Rising OI + rising price = new money entering (trend continuation). Falling OI = positions closing.
              </div>
            </div>
            <div className="flex gap-2">
              <Percent className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-medium text-[var(--text-primary)]">Funding Rate</span> — Positive = longs pay shorts (crowded longs). Extreme positive (&gt;0.05%) often precedes dumps.
              </div>
            </div>
            <div className="flex gap-2">
              <Users className="w-3.5 h-3.5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-medium text-[var(--text-primary)]">Top vs Retail</span> — When top traders diverge from retail, follow the top traders. Retail tends to be wrong at extremes.
              </div>
            </div>
            <div className="flex gap-2">
              <Flame className="w-3.5 h-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-medium text-[var(--text-primary)]">Taker Volume</span> — Net buy volume = aggressive buyers. Confirms breakouts and SMC zone taps.
              </div>
            </div>
            <div className="flex gap-2">
              <TrendingUp className="w-3.5 h-3.5 text-purple-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-medium text-[var(--text-primary)]">Premium/Discount</span> — Futures trading above spot (premium) = bullish. Below spot = bearish or hedging.
              </div>
            </div>
            <div className="flex gap-2">
              <BarChart3 className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-medium text-[var(--text-primary)]">Sentiment Gauge</span> — Combines all metrics. Strong extremes suggest contrarian opportunities (SMC liquidity grabs).
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
