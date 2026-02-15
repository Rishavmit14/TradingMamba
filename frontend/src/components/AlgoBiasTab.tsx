"use client";

import { useState, useEffect, useCallback, memo } from "react";
import {
  Brain,
  RefreshCw,
  AlertTriangle,
  Loader2,
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  BarChart3,
  Waves,
  Shield,
  Zap,
  Target,
  Users,
  Gauge,
  LineChart,
  ChevronDown,
  ChevronRight,
  Layers,
} from "lucide-react";
import { fetchAlgoBias } from "@/lib/api";
import { CompositeBias, AlgoBiasResult } from "@/lib/types";

// ── Algo metadata for icons and color theming ──

const ALGO_META: Record<string, { icon: typeof Brain; color: string; label: string }> = {
  vpin: { icon: Activity, color: "cyan", label: "VPIN" },
  funding_ou: { icon: Waves, color: "purple", label: "Funding OU" },
  options_greeks: { icon: Target, color: "amber", label: "Options" },
  kyle_amihud: { icon: BarChart3, color: "indigo", label: "Kyle-Amihud" },
  hurst: { icon: LineChart, color: "teal", label: "Hurst" },
  liquidation: { icon: Zap, color: "red", label: "Liquidation" },
  vol_regime: { icon: Gauge, color: "orange", label: "Vol Regime" },
  ofi: { icon: TrendingUp, color: "emerald", label: "OFI" },
  smart_retail: { icon: Users, color: "violet", label: "Smart/Retail" },
  bayesian_sentiment: { icon: Shield, color: "pink", label: "Bayesian" },
};

const COLOR_MAP: Record<string, { border: string; bg: string; text: string; accent: string }> = {
  cyan: { border: "border-cyan-500/20", bg: "bg-cyan-500/5", text: "text-cyan-400", accent: "bg-cyan-500" },
  purple: { border: "border-purple-500/20", bg: "bg-purple-500/5", text: "text-purple-400", accent: "bg-purple-500" },
  amber: { border: "border-amber-500/20", bg: "bg-amber-500/5", text: "text-amber-400", accent: "bg-amber-500" },
  indigo: { border: "border-indigo-500/20", bg: "bg-indigo-500/5", text: "text-indigo-400", accent: "bg-indigo-500" },
  teal: { border: "border-teal-500/20", bg: "bg-teal-500/5", text: "text-teal-400", accent: "bg-teal-500" },
  red: { border: "border-red-500/20", bg: "bg-red-500/5", text: "text-red-400", accent: "bg-red-500" },
  orange: { border: "border-orange-500/20", bg: "bg-orange-500/5", text: "text-orange-400", accent: "bg-orange-500" },
  emerald: { border: "border-emerald-500/20", bg: "bg-emerald-500/5", text: "text-emerald-400", accent: "bg-emerald-500" },
  violet: { border: "border-violet-500/20", bg: "bg-violet-500/5", text: "text-violet-400", accent: "bg-violet-500" },
  pink: { border: "border-pink-500/20", bg: "bg-pink-500/5", text: "text-pink-400", accent: "bg-pink-500" },
};

// ── Direction helpers ──

function directionColor(dir: string): string {
  if (dir.includes("bullish")) return "text-emerald-400";
  if (dir.includes("bearish")) return "text-red-400";
  return "text-yellow-400";
}

function directionBg(dir: string): string {
  if (dir.includes("bullish")) return "bg-emerald-500/10 border-emerald-500/20";
  if (dir.includes("bearish")) return "bg-red-500/10 border-red-500/20";
  return "bg-yellow-500/10 border-yellow-500/20";
}

function directionLabel(dir: string): string {
  return dir.replace("_", " ").replace(/\b\w/g, c => c.toUpperCase());
}

function DirectionIcon({ dir, size = 16 }: { dir: string; size?: number }) {
  if (dir.includes("bullish")) return <TrendingUp size={size} className="text-emerald-400" />;
  if (dir.includes("bearish")) return <TrendingDown size={size} className="text-red-400" />;
  return <Minus size={size} className="text-yellow-400" />;
}

// ── Composite Gauge (SVG arc) ──

function CompositeGauge({ score, confidence, direction }: { score: number; confidence: number; direction: string }) {
  // Arc from -100 (left) to +100 (right), mapped to 180-degree sweep
  const normalized = (score + 100) / 200; // 0 to 1
  const angle = -180 + normalized * 180; // -180 (left) to 0 (right), -90 = center/top

  const radius = 80;
  const cx = 100;
  const cy = 95;

  // Arc path for background
  const arcPath = (startAngle: number, endAngle: number, r: number) => {
    const start = {
      x: cx + r * Math.cos((startAngle * Math.PI) / 180),
      y: cy + r * Math.sin((startAngle * Math.PI) / 180),
    };
    const end = {
      x: cx + r * Math.cos((endAngle * Math.PI) / 180),
      y: cy + r * Math.sin((endAngle * Math.PI) / 180),
    };
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
  };

  // Needle endpoint
  const needleAngle = (angle * Math.PI) / 180;
  const needleX = cx + (radius - 10) * Math.cos(needleAngle);
  const needleY = cy + (radius - 10) * Math.sin(needleAngle);

  const gaugeColor = score > 25 ? "#34d399" : score < -25 ? "#f87171" : "#facc15";

  return (
    <div className="flex flex-col items-center">
      <svg width="200" height="120" viewBox="0 0 200 120">
        {/* Background arc */}
        <path d={arcPath(-180, 0, radius)} fill="none" stroke="var(--bg-tertiary)" strokeWidth="12" strokeLinecap="round" />

        {/* Colored segments */}
        <path d={arcPath(-180, -144, radius)} fill="none" stroke="#f87171" strokeWidth="12" strokeLinecap="round" opacity="0.3" />
        <path d={arcPath(-144, -108, radius)} fill="none" stroke="#fb923c" strokeWidth="12" opacity="0.2" />
        <path d={arcPath(-108, -72, radius)} fill="none" stroke="#facc15" strokeWidth="12" opacity="0.2" />
        <path d={arcPath(-72, -36, radius)} fill="none" stroke="#a3e635" strokeWidth="12" opacity="0.2" />
        <path d={arcPath(-36, 0, radius)} fill="none" stroke="#34d399" strokeWidth="12" strokeLinecap="round" opacity="0.3" />

        {/* Needle */}
        <line x1={cx} y1={cy} x2={needleX} y2={needleY} stroke={gaugeColor} strokeWidth="2.5" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="4" fill={gaugeColor} />

        {/* Labels */}
        <text x="18" y="100" fill="var(--text-muted)" fontSize="11" textAnchor="middle">-100</text>
        <text x="100" y="12" fill="var(--text-muted)" fontSize="11" textAnchor="middle">0</text>
        <text x="182" y="100" fill="var(--text-muted)" fontSize="11" textAnchor="middle">+100</text>
      </svg>

      {/* Score display */}
      <div className="text-center -mt-2">
        <div className="text-3xl font-bold font-mono" style={{ color: gaugeColor }}>
          {score > 0 ? "+" : ""}{score.toFixed(1)}
        </div>
        <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full border mt-1 ${directionBg(direction)}`}>
          <DirectionIcon dir={direction} size={14} />
          <span className={`text-xs font-semibold ${directionColor(direction)}`}>
            {directionLabel(direction)}
          </span>
        </div>
        <div className="text-xs text-[var(--text-muted)] mt-1.5">
          Confidence: <span className="font-mono font-bold text-[var(--text-secondary)]">{confidence.toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
}

// ── Score Bar (centered at 0, extends left for negative, right for positive) ──

function ScoreBar({ score, color }: { score: number; color: string }) {
  const pct = Math.abs(score);
  const isPositive = score >= 0;

  return (
    <div className="relative w-full h-2 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
      {/* Center marker */}
      <div className="absolute left-1/2 top-0 w-px h-full bg-[var(--text-muted)]/30 z-10" />
      {/* Bar */}
      <div
        className={`absolute top-0 h-full rounded-full transition-all duration-500 ${
          isPositive ? "bg-emerald-500" : "bg-red-500"
        }`}
        style={{
          left: isPositive ? "50%" : `${50 - pct / 2}%`,
          width: `${pct / 2}%`,
        }}
      />
    </div>
  );
}

// ── Confidence Bar ──

function ConfidenceBar({ value, accent }: { value: number; accent: string }) {
  return (
    <div className="w-full h-1.5 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
      <div className={`h-full rounded-full ${accent} transition-all duration-500`}
           style={{ width: `${value}%`, opacity: 0.7 + (value / 100) * 0.3 }} />
    </div>
  );
}

// ── Algorithm Card ──

function AlgoCard({ algo }: { algo: AlgoBiasResult }) {
  const meta = ALGO_META[algo.algo_id] || { icon: Brain, color: "cyan", label: algo.algo_name };
  const colors = COLOR_MAP[meta.color] || COLOR_MAP.cyan;
  const Icon = meta.icon;

  // Pick 2-3 key components to display
  const componentEntries = Object.entries(algo.components || {}).filter(
    ([k]) => !["error"].includes(k)
  ).slice(0, 4);

  return (
    <div className={`rounded-xl p-4 border ${colors.border} ${colors.bg} transition-all hover:border-opacity-40`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`w-7 h-7 rounded-lg ${colors.bg} border ${colors.border} flex items-center justify-center`}>
            <Icon className={`w-3.5 h-3.5 ${colors.text}`} />
          </div>
          <div>
            <div className="text-xs font-semibold text-[var(--text-primary)]">{algo.algo_name}</div>
            <div className="text-[11px] text-[var(--text-muted)] uppercase tracking-wider">{algo.algo_id}</div>
          </div>
        </div>
        <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full border ${directionBg(algo.direction)}`}>
          <DirectionIcon dir={algo.direction} size={12} />
          <span className={`text-xs font-semibold ${directionColor(algo.direction)}`}>
            {algo.direction.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Score bar */}
      <div className="mb-2">
        <div className="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
          <span>Score</span>
          <span className="font-mono font-bold text-[var(--text-secondary)]">
            {algo.score > 0 ? "+" : ""}{algo.score.toFixed(1)}
          </span>
        </div>
        <ScoreBar score={algo.score} color={meta.color} />
      </div>

      {/* Confidence bar */}
      <div className="mb-3">
        <div className="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
          <span>Confidence</span>
          <span className="font-mono font-bold text-[var(--text-secondary)]">{algo.confidence.toFixed(0)}%</span>
        </div>
        <ConfidenceBar value={algo.confidence} accent={colors.accent} />
      </div>

      {/* Key metrics */}
      {componentEntries.length > 0 && (
        <div className="grid grid-cols-2 gap-1.5 mb-2">
          {componentEntries.map(([key, val]) => (
            <div key={key} className="rounded-md bg-[var(--bg-primary)]/50 px-2 py-1">
              <div className="text-xs text-[var(--text-muted)] uppercase truncate">{key.replace(/_/g, " ")}</div>
              <div className="text-xs font-mono font-bold text-[var(--text-secondary)] truncate">
                {typeof val === "number" ? (
                  Math.abs(val) < 0.001 && val !== 0
                    ? val.toExponential(2)
                    : Math.abs(val) > 1000
                    ? val.toLocaleString(undefined, { maximumFractionDigits: 0 })
                    : Number(val.toFixed(4))
                ) : typeof val === "boolean" ? (val ? "YES" : "NO") : String(val)}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Explanation */}
      <div className="text-xs text-[var(--text-muted)] leading-relaxed line-clamp-2">
        {algo.explanation}
      </div>
    </div>
  );
}

// ── Regime Badges ──

function RegimeBadge({ label, value, variant }: { label: string; value: string; variant: "info" | "warn" | "neutral" }) {
  const styles = {
    info: "bg-blue-500/10 border-blue-500/20 text-blue-400",
    warn: "bg-amber-500/10 border-amber-500/20 text-amber-400",
    neutral: "bg-[var(--bg-tertiary)] border-[var(--border-primary)] text-[var(--text-secondary)]",
  };
  return (
    <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs font-medium ${styles[variant]}`}>
      <span className="text-[var(--text-muted)]">{label}:</span>
      <span className="font-semibold uppercase">{value.replace(/_/g, " ")}</span>
    </div>
  );
}

// ── Meta-Algorithm Section ──

function humanizeHurst(h: number): { label: string; detail: string; color: string } {
  if (h > 0.65) return { label: "Strong Trend", detail: "Market is persistently trending — momentum strategies dominate", color: "text-cyan-400" };
  if (h > 0.55) return { label: "Mild Trend", detail: "Slight trending bias — momentum strategies get a moderate edge", color: "text-blue-400" };
  if (h < 0.35) return { label: "Strong Mean Reversion", detail: "Market is reverting aggressively — contrarian strategies dominate", color: "text-amber-400" };
  if (h < 0.45) return { label: "Mild Mean Reversion", detail: "Slight reversion bias — contrarian strategies get a moderate edge", color: "text-yellow-400" };
  return { label: "Random Walk", detail: "No structural bias detected — all strategies weighted equally", color: "text-[var(--text-muted)]" };
}

function humanizeVol(vol: string): { label: string; detail: string; color: string } {
  if (vol === "extreme") return { label: "Extreme Volatility", detail: "All confidence scores reduced by 25% — anything can happen", color: "text-red-400" };
  if (vol === "high") return { label: "High Volatility", detail: "Momentum algos get +15% weight — big moves tend to continue", color: "text-orange-400" };
  if (vol === "low") return { label: "Low Volatility", detail: "Reversion algos get +10% weight — breakouts from compression tend to fade first", color: "text-teal-400" };
  return { label: "Normal Volatility", detail: "No weight adjustment — standard allocation", color: "text-[var(--text-muted)]" };
}

function humanizeEntropy(adj: number): { label: string; detail: string; color: string } {
  if (adj > 0) return { label: "Strong Agreement", detail: "Most algos agree on direction — confidence boosted by +15", color: "text-emerald-400" };
  if (adj < 0) return { label: "High Disagreement", detail: "Algos conflict on direction — confidence penalized by -20", color: "text-red-400" };
  return { label: "Moderate Agreement", detail: "Mixed signals — no confidence adjustment", color: "text-[var(--text-muted)]" };
}

function MetaAlgoSection({ data }: { data: CompositeBias }) {
  const [open, setOpen] = useState(false);
  const meta = data.meta;

  if (!meta || !meta.algo_weights) return null;

  const hurst = humanizeHurst(meta.hurst_value);
  const vol = humanizeVol(data.vol_regime);
  const entropy = humanizeEntropy(meta.entropy_adj);

  // Group algos by category
  const entries = Object.entries(meta.algo_weights);
  const momentumAlgos = entries.filter(([, d]) => d.category === "momentum").sort(([, a], [, b]) => b.adjusted_weight - a.adjusted_weight);
  const reversionAlgos = entries.filter(([, d]) => d.category === "reversion").sort(([, a], [, b]) => b.adjusted_weight - a.adjusted_weight);
  const regimeAlgos = entries.filter(([, d]) => d.category === "regime").sort(([, a], [, b]) => b.adjusted_weight - a.adjusted_weight);

  // Total contributions by category
  const momContrib = momentumAlgos.reduce((s, [, d]) => s + d.contribution, 0);
  const revContrib = reversionAlgos.reduce((s, [, d]) => s + d.contribution, 0);
  const regContrib = regimeAlgos.reduce((s, [, d]) => s + d.contribution, 0);

  // Total weight shift
  const momWeightTotal = momentumAlgos.reduce((s, [, d]) => s + d.adjusted_weight, 0);
  const revWeightTotal = reversionAlgos.reduce((s, [, d]) => s + d.adjusted_weight, 0);
  const regWeightTotal = regimeAlgos.reduce((s, [, d]) => s + d.adjusted_weight, 0);

  // Quick summary line
  const favored = momWeightTotal > revWeightTotal + 5 ? "momentum" : revWeightTotal > momWeightTotal + 5 ? "mean-reversion" : "balanced";

  return (
    <div className="rounded-xl border border-[var(--border-primary)] bg-[var(--bg-card)] overflow-hidden">
      {/* Toggle header */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3 hover:bg-[var(--bg-tertiary)]/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-fuchsia-400" />
          <span className="text-xs font-semibold text-[var(--text-primary)]">Meta-Algorithm</span>
          <span className="text-xs text-[var(--text-muted)]">— how the ensemble combined 10 algos into one score</span>
        </div>
        {open ? (
          <ChevronDown className="w-4 h-4 text-[var(--text-muted)]" />
        ) : (
          <ChevronRight className="w-4 h-4 text-[var(--text-muted)]" />
        )}
      </button>

      {open && (
        <div className="px-5 pb-5 border-t border-[var(--border-primary)] pt-4 space-y-5">

          {/* ── Step 1: Plain English summary ── */}
          <div className="rounded-lg bg-fuchsia-500/5 border border-fuchsia-500/15 p-4">
            <div className="text-xs text-fuchsia-400/70 uppercase tracking-wider font-semibold mb-2">What the ensemble did</div>
            <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed">
              The Hurst exponent detected a <span className={`font-bold ${hurst.color}`}>{hurst.label.toLowerCase()}</span> regime
              (H&nbsp;=&nbsp;{meta.hurst_value.toFixed(3)}), so the ensemble <span className="font-semibold">
              {favored === "momentum" ? "boosted momentum algos (VPIN, OFI, Kyle, Liquidation) and reduced contrarian ones" :
               favored === "mean-reversion" ? "boosted contrarian algos (Funding OU, Bayesian, Smart/Retail) and reduced momentum ones" :
               "kept weights roughly equal"}
              </span>.
              {vol.label !== "Normal Volatility" && <> Volatility is <span className={`font-bold ${vol.color}`}>{vol.label.toLowerCase()}</span> — {vol.detail.split("—")[1]?.trim() || vol.detail}.</>}
              {" "}Algos showed <span className={`font-bold ${entropy.color}`}>{entropy.label.toLowerCase()}</span>
              {meta.entropy_adj !== 0 && <> ({meta.entropy_adj > 0 ? "+" : ""}{meta.entropy_adj} confidence)</>}.
            </p>
          </div>

          {/* ── Step 2: Three decision stages ── */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {/* Stage 1: Hurst */}
            <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 border border-[var(--border-primary)]">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-5 h-5 rounded-full bg-blue-500/15 flex items-center justify-center text-[11px] font-bold text-blue-400">1</div>
                <div className="text-xs font-semibold text-[var(--text-primary)]">Regime Detection</div>
              </div>
              <div className={`text-xs font-bold mb-1 ${hurst.color}`}>{hurst.label}</div>
              <div className="text-xs text-[var(--text-muted)] leading-relaxed">{hurst.detail}</div>
              <div className="mt-2 flex items-center gap-2">
                <span className="text-[11px] text-[var(--text-muted)]">Hurst</span>
                <div className="flex-1 h-1.5 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
                  <div className={`h-full rounded-full transition-all ${meta.hurst_value > 0.5 ? "bg-cyan-500" : "bg-amber-500"}`}
                       style={{ width: `${meta.hurst_value * 100}%` }} />
                </div>
                <span className="text-[11px] font-mono font-bold text-[var(--text-secondary)]">{meta.hurst_value.toFixed(3)}</span>
              </div>
            </div>

            {/* Stage 2: Vol */}
            <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 border border-[var(--border-primary)]">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-5 h-5 rounded-full bg-orange-500/15 flex items-center justify-center text-[11px] font-bold text-orange-400">2</div>
                <div className="text-xs font-semibold text-[var(--text-primary)]">Volatility Modifier</div>
              </div>
              <div className={`text-xs font-bold mb-1 ${vol.color}`}>{vol.label}</div>
              <div className="text-xs text-[var(--text-muted)] leading-relaxed">{vol.detail}</div>
            </div>

            {/* Stage 3: Agreement */}
            <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3 border border-[var(--border-primary)]">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-5 h-5 rounded-full bg-violet-500/15 flex items-center justify-center text-[11px] font-bold text-violet-400">3</div>
                <div className="text-xs font-semibold text-[var(--text-primary)]">Agreement Check</div>
              </div>
              <div className={`text-xs font-bold mb-1 ${entropy.color}`}>{entropy.label}</div>
              <div className="text-xs text-[var(--text-muted)] leading-relaxed">{entropy.detail}</div>
            </div>
          </div>

          {/* ── Step 3: Category scoreboard ── */}
          <div>
            <div className="text-[11px] text-[var(--text-muted)] uppercase tracking-wider font-semibold mb-3">
              Who pulled the score where
            </div>

            {/* Momentum group */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-cyan-500" />
                  <span className="text-xs font-semibold text-cyan-400">Momentum Algos</span>
                  <span className="text-[11px] text-[var(--text-muted)]">— follow the dominant flow</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[11px] text-[var(--text-muted)]">Weight: <span className="font-mono font-bold text-[var(--text-secondary)]">{momWeightTotal.toFixed(0)}%</span></span>
                  <span className={`text-xs font-mono font-bold ${momContrib > 0 ? "text-emerald-400" : momContrib < 0 ? "text-red-400" : "text-[var(--text-muted)]"}`}>
                    {momContrib > 0 ? "+" : ""}{momContrib.toFixed(1)} pts
                  </span>
                </div>
              </div>
              <div className="space-y-1.5 pl-4">
                {momentumAlgos.map(([algoId, detail]) => {
                  const am = ALGO_META[algoId];
                  const colors = COLOR_MAP[am?.color || "cyan"] || COLOR_MAP.cyan;
                  const delta = detail.weight_change;
                  return (
                    <div key={algoId} className="flex items-center gap-2">
                      <span className="w-20 text-xs text-[var(--text-secondary)] truncate">{am?.label || algoId}</span>
                      <div className="flex-1 h-1.5 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
                        <div className={`h-full rounded-full ${colors.accent}`} style={{ width: `${(detail.adjusted_weight / 20) * 100}%`, opacity: 0.7 }} />
                      </div>
                      <span className="w-10 text-[11px] font-mono text-[var(--text-muted)] text-right">{detail.adjusted_weight.toFixed(1)}%</span>
                      <span className={`w-12 text-[11px] font-mono font-bold text-right ${delta > 0.05 ? "text-emerald-400" : delta < -0.05 ? "text-red-400" : "text-[var(--text-muted)]"}`}>
                        {delta > 0 ? "+" : ""}{delta.toFixed(1)}%
                      </span>
                      <span className={`w-12 text-[11px] font-mono font-bold text-right ${detail.contribution > 0 ? "text-emerald-400" : detail.contribution < 0 ? "text-red-400" : "text-[var(--text-muted)]"}`}>
                        {detail.contribution > 0 ? "+" : ""}{detail.contribution.toFixed(1)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Reversion group */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-amber-500" />
                  <span className="text-xs font-semibold text-amber-400">Mean-Reversion Algos</span>
                  <span className="text-[11px] text-[var(--text-muted)]">— fade the crowded side</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[11px] text-[var(--text-muted)]">Weight: <span className="font-mono font-bold text-[var(--text-secondary)]">{revWeightTotal.toFixed(0)}%</span></span>
                  <span className={`text-xs font-mono font-bold ${revContrib > 0 ? "text-emerald-400" : revContrib < 0 ? "text-red-400" : "text-[var(--text-muted)]"}`}>
                    {revContrib > 0 ? "+" : ""}{revContrib.toFixed(1)} pts
                  </span>
                </div>
              </div>
              <div className="space-y-1.5 pl-4">
                {reversionAlgos.map(([algoId, detail]) => {
                  const am = ALGO_META[algoId];
                  const colors = COLOR_MAP[am?.color || "amber"] || COLOR_MAP.amber;
                  const delta = detail.weight_change;
                  return (
                    <div key={algoId} className="flex items-center gap-2">
                      <span className="w-20 text-xs text-[var(--text-secondary)] truncate">{am?.label || algoId}</span>
                      <div className="flex-1 h-1.5 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
                        <div className={`h-full rounded-full ${colors.accent}`} style={{ width: `${(detail.adjusted_weight / 20) * 100}%`, opacity: 0.7 }} />
                      </div>
                      <span className="w-10 text-[11px] font-mono text-[var(--text-muted)] text-right">{detail.adjusted_weight.toFixed(1)}%</span>
                      <span className={`w-12 text-[11px] font-mono font-bold text-right ${delta > 0.05 ? "text-emerald-400" : delta < -0.05 ? "text-red-400" : "text-[var(--text-muted)]"}`}>
                        {delta > 0 ? "+" : ""}{delta.toFixed(1)}%
                      </span>
                      <span className={`w-12 text-[11px] font-mono font-bold text-right ${detail.contribution > 0 ? "text-emerald-400" : detail.contribution < 0 ? "text-red-400" : "text-[var(--text-muted)]"}`}>
                        {detail.contribution > 0 ? "+" : ""}{detail.contribution.toFixed(1)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Regime group */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-[var(--text-muted)]" />
                  <span className="text-xs font-semibold text-[var(--text-secondary)]">Regime Indicators</span>
                  <span className="text-[11px] text-[var(--text-muted)]">— provide context, not direction</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[11px] text-[var(--text-muted)]">Weight: <span className="font-mono font-bold text-[var(--text-secondary)]">{regWeightTotal.toFixed(0)}%</span></span>
                  <span className={`text-xs font-mono font-bold ${regContrib > 0 ? "text-emerald-400" : regContrib < 0 ? "text-red-400" : "text-[var(--text-muted)]"}`}>
                    {regContrib > 0 ? "+" : ""}{regContrib.toFixed(1)} pts
                  </span>
                </div>
              </div>
              <div className="space-y-1.5 pl-4">
                {regimeAlgos.map(([algoId, detail]) => {
                  const am = ALGO_META[algoId];
                  const colors = COLOR_MAP[am?.color || "cyan"] || COLOR_MAP.cyan;
                  const delta = detail.weight_change;
                  return (
                    <div key={algoId} className="flex items-center gap-2">
                      <span className="w-20 text-xs text-[var(--text-secondary)] truncate">{am?.label || algoId}</span>
                      <div className="flex-1 h-1.5 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
                        <div className={`h-full rounded-full ${colors.accent}`} style={{ width: `${(detail.adjusted_weight / 20) * 100}%`, opacity: 0.7 }} />
                      </div>
                      <span className="w-10 text-[11px] font-mono text-[var(--text-muted)] text-right">{detail.adjusted_weight.toFixed(1)}%</span>
                      <span className={`w-12 text-[11px] font-mono font-bold text-right ${delta > 0.05 ? "text-emerald-400" : delta < -0.05 ? "text-red-400" : "text-[var(--text-muted)]"}`}>
                        {delta > 0 ? "+" : ""}{delta.toFixed(1)}%
                      </span>
                      <span className={`w-12 text-[11px] font-mono font-bold text-right ${detail.contribution > 0 ? "text-emerald-400" : detail.contribution < 0 ? "text-red-400" : "text-[var(--text-muted)]"}`}>
                        {detail.contribution > 0 ? "+" : ""}{detail.contribution.toFixed(1)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Column legend */}
            <div className="flex items-center justify-end gap-4 mt-3 text-xs text-[var(--text-muted)]">
              <span>Weight</span>
              <span>Shift</span>
              <span>Score Contribution</span>
            </div>
          </div>

          {/* ── Step 4: Confidence math ── */}
          <div className="rounded-lg bg-[var(--bg-primary)]/50 p-4 border border-[var(--border-primary)]">
            <div className="text-[11px] text-[var(--text-muted)] uppercase tracking-wider font-semibold mb-2">Confidence Calculation</div>
            <div className="flex items-center gap-2 text-[13px] font-mono flex-wrap">
              <span className="text-[var(--text-secondary)]">{meta.base_confidence.toFixed(1)}</span>
              <span className="text-[var(--text-muted)]">base</span>
              <span className={meta.entropy_adj > 0 ? "text-emerald-400" : meta.entropy_adj < 0 ? "text-red-400" : "text-[var(--text-muted)]"}>
                {meta.entropy_adj > 0 ? "+" : ""}{meta.entropy_adj}
              </span>
              <span className="text-[var(--text-muted)]">{meta.entropy_adj > 0 ? "agreement bonus" : meta.entropy_adj < 0 ? "disagreement penalty" : "entropy"}</span>
              {meta.vol_confidence_penalty < 1 && (
                <>
                  <span className="text-red-400">x{meta.vol_confidence_penalty.toFixed(2)}</span>
                  <span className="text-[var(--text-muted)]">extreme vol penalty</span>
                </>
              )}
              <span className="text-[var(--text-muted)]">=</span>
              <span className="text-[var(--text-primary)] font-bold">{data.confidence.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main Component ──

function AlgoBiasTabInner() {
  const [data, setData] = useState<CompositeBias | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const result = await fetchAlgoBias();
      setData(result);
      setError(null);
      setLastUpdated(new Date());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to fetch algo bias data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60_000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Loading state
  if (loading && !data) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Loader2 className="w-10 h-10 text-violet-500 animate-spin mx-auto mb-4" />
          <p className="text-sm text-[var(--text-secondary)]">Running 10 institutional algorithms...</p>
          <p className="text-xs text-[var(--text-muted)] mt-1">VPIN, OU, GEX, Kyle-Lambda, Hurst, OFI, Bayesian...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error && !data) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-sm">
          <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center mx-auto mb-4">
            <AlertTriangle className="w-6 h-6 text-red-400" />
          </div>
          <p className="text-sm text-red-400 mb-1">Algorithm Engine Error</p>
          <p className="text-xs text-[var(--text-muted)] mb-4">{error}</p>
          <button onClick={fetchData} className="px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-xs font-medium rounded-lg transition-colors">
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const algos = data.algos || [];
  const bullCount = algos.filter(a => a.direction === "bullish").length;
  const bearCount = algos.filter(a => a.direction === "bearish").length;
  const neutCount = algos.filter(a => a.direction === "neutral").length;

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto p-6 space-y-5">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 border border-violet-500/20 flex items-center justify-center">
              <Brain className="w-4 h-4 text-violet-400" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-[var(--text-primary)]">Quant Algo Bias</h1>
              <p className="text-xs text-[var(--text-muted)]">10 institutional algorithms — from-scratch ensemble</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {lastUpdated && (
              <span className="text-xs text-[var(--text-muted)]">
                {lastUpdated.toLocaleTimeString()}
              </span>
            )}
            <button
              onClick={fetchData}
              className="w-7 h-7 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-primary)] flex items-center justify-center hover:bg-[var(--bg-tertiary)] transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5 text-[var(--text-muted)]" />
            </button>
          </div>
        </div>

        {/* Data Sources Status */}
        <div className="flex items-center gap-2 flex-wrap">
          {[
            { name: "Taker Vol", ok: algos.some(a => a.algo_id === "vpin" && a.confidence > 0) },
            { name: "Funding", ok: algos.some(a => a.algo_id === "funding_ou" && a.confidence > 0) },
            { name: "Options", ok: algos.some(a => a.algo_id === "options_greeks" && a.confidence > 0) },
            { name: "L2 Depth", ok: algos.some(a => a.algo_id === "ofi" && a.confidence > 0) },
            { name: "Liq WS", ok: algos.some(a => a.algo_id === "liquidation" && a.confidence > 0) },
            { name: "L/S Ratio", ok: algos.some(a => a.algo_id === "smart_retail" && a.confidence > 0) },
            { name: "FnG", ok: algos.some(a => a.algo_id === "bayesian_sentiment" && a.confidence > 0) },
          ].map(({ name, ok }) => (
            <div
              key={name}
              className={`flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium ${
                ok
                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                  : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] border border-[var(--border-primary)]"
              }`}
            >
              <div className={`w-1.5 h-1.5 rounded-full ${ok ? "bg-emerald-400" : "bg-[var(--text-muted)]/30"}`} />
              {name}
            </div>
          ))}
        </div>

        {/* Composite Gauge Section */}
        <div className="rounded-xl p-5 border border-[var(--border-primary)] bg-[var(--bg-card)]">
          <div className="flex flex-col lg:flex-row items-center gap-6">
            {/* Gauge */}
            <div className="flex-shrink-0">
              <CompositeGauge score={data.score} confidence={data.confidence} direction={data.direction} />
            </div>

            {/* Stats */}
            <div className="flex-1 w-full space-y-4">
              {/* Regime badges */}
              <div className="flex flex-wrap gap-2">
                <RegimeBadge
                  label="Regime"
                  value={data.regime}
                  variant={data.regime === "trending" ? "info" : data.regime === "mean_reverting" ? "warn" : "neutral"}
                />
                <RegimeBadge
                  label="Volatility"
                  value={data.vol_regime}
                  variant={data.vol_regime === "extreme" ? "warn" : data.vol_regime === "high" ? "warn" : "neutral"}
                />
                <RegimeBadge
                  label="Entropy"
                  value={data.entropy.toFixed(2)}
                  variant={data.entropy < 0.8 ? "info" : data.entropy > 1.3 ? "warn" : "neutral"}
                />
              </div>

              {/* Agreement grid */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
                  <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Algos Active</div>
                  <div className="text-xl font-bold font-mono text-[var(--text-primary)]">
                    {data.algo_count}<span className="text-sm text-[var(--text-muted)]">/10</span>
                  </div>
                </div>
                <div className="rounded-lg bg-emerald-500/5 p-3">
                  <div className="text-xs text-emerald-400/70 uppercase tracking-wider mb-1">Bullish</div>
                  <div className="text-xl font-bold font-mono text-emerald-400">{bullCount}</div>
                </div>
                <div className="rounded-lg bg-red-500/5 p-3">
                  <div className="text-xs text-red-400/70 uppercase tracking-wider mb-1">Bearish</div>
                  <div className="text-xl font-bold font-mono text-red-400">{bearCount}</div>
                </div>
                <div className="rounded-lg bg-yellow-500/5 p-3">
                  <div className="text-xs text-yellow-400/70 uppercase tracking-wider mb-1">Neutral</div>
                  <div className="text-xl font-bold font-mono text-yellow-400">{neutCount}</div>
                </div>
              </div>

              {/* Agreement bar */}
              <div>
                <div className="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
                  <span>Agreement Distribution</span>
                  <span className="font-mono">{data.agreement_count}/{data.algo_count} agree</span>
                </div>
                <div className="flex h-2.5 rounded-full overflow-hidden bg-[var(--bg-tertiary)]">
                  {bullCount > 0 && (
                    <div className="bg-emerald-500 transition-all duration-500"
                         style={{ width: `${(bullCount / Math.max(data.algo_count, 1)) * 100}%` }} />
                  )}
                  {neutCount > 0 && (
                    <div className="bg-yellow-500 transition-all duration-500"
                         style={{ width: `${(neutCount / Math.max(data.algo_count, 1)) * 100}%` }} />
                  )}
                  {bearCount > 0 && (
                    <div className="bg-red-500 transition-all duration-500"
                         style={{ width: `${(bearCount / Math.max(data.algo_count, 1)) * 100}%` }} />
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Meta-Algorithm Details (expandable) */}
        <MetaAlgoSection data={data} />

        {/* Algorithm Cards Grid */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-4 h-4 text-[var(--text-muted)]" />
            <h2 className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wider">
              Individual Algorithms
            </h2>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {algos.map((algo) => (
              <AlgoCard key={algo.algo_id} algo={algo} />
            ))}
          </div>
        </div>

        {/* How it works section */}
        <div className="rounded-xl p-4 border border-[var(--border-primary)] bg-[var(--bg-card)]">
          <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-2 font-semibold">
            How It Works
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-[var(--text-muted)] leading-relaxed">
            <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
              <span className="font-bold text-[var(--text-secondary)]">10 Algorithms</span> — VPIN (informed flow), Ornstein-Uhlenbeck (funding mean reversion), GEX + Options Greeks, Kyle&apos;s Lambda (price impact), Hurst Exponent (regime), Liquidation Cascade (logistic model), ATR Volatility Regime, Order Flow Imbalance, Smart vs Retail divergence, Bayesian Sentiment Fusion.
            </div>
            <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
              <span className="font-bold text-[var(--text-secondary)]">Adaptive Ensemble</span> — Weights adjust based on Hurst regime (momentum algos boosted in trending, reversion algos in mean-reverting). Vol regime modifies confidence. Shannon entropy measures agreement.
            </div>
            <div className="rounded-lg bg-[var(--bg-primary)]/50 p-3">
              <span className="font-bold text-[var(--text-secondary)]">From Scratch</span> — Every formula built from academic papers (Easley 2012, Kyle 1985, Amihud 2002, Hurst 1951, Cont 2014). No existing codebase math reused. Raw API data only.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const AlgoBiasTab = memo(AlgoBiasTabInner);
export default AlgoBiasTab;
