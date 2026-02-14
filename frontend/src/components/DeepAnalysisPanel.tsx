"use client";

import {
  X,
  TrendingUp,
  TrendingDown,
  Minus,
  ArrowRight,
  Shield,
  Target,
  ChevronDown,
  ChevronRight,
  Zap,
  Layers,
  GitBranch,
  Activity,
  BarChart3,
  Clock,
} from "lucide-react";
import { useState } from "react";
import { DeepAnalysis } from "@/lib/types";

interface DeepAnalysisPanelProps {
  analysis: DeepAnalysis;
  onClose: () => void;
}

const COMPONENT_ICONS: Record<string, { icon: typeof Activity; color: string }> = {
  swing:     { icon: Activity, color: "#f59e0b" },
  bos:       { icon: ArrowRight, color: "#22d3ee" },
  choch:     { icon: TrendingUp, color: "#a855f7" },
  idm:       { icon: GitBranch, color: "#60a5fa" },
  fvg:       { icon: Layers, color: "#10b981" },
  ob:        { icon: Shield, color: "#f97316" },
  liquidity: { icon: Zap, color: "#ec4899" },
  pd:        { icon: BarChart3, color: "#eab308" },
  vsa:       { icon: TrendingUp, color: "#14b8a6" },
};

const ROLE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  Bias:  { bg: "bg-purple-500/10", text: "text-purple-400", border: "border-purple-500/20" },
  Setup: { bg: "bg-blue-500/10", text: "text-blue-400", border: "border-blue-500/20" },
  Entry: { bg: "bg-emerald-500/10", text: "text-emerald-400", border: "border-emerald-500/20" },
};

const STRENGTH_CONFIG: Record<string, { bg: string; text: string }> = {
  strong: { bg: "bg-emerald-500/10", text: "text-emerald-400" },
  medium: { bg: "bg-blue-500/10", text: "text-blue-400" },
  weak:   { bg: "bg-amber-500/10", text: "text-amber-400" },
};

function TrendIcon({ trend }: { trend: string }) {
  if (trend === "bullish") return <TrendingUp className="w-3 h-3 text-emerald-400" />;
  if (trend === "bearish") return <TrendingDown className="w-3 h-3 text-red-400" />;
  return <Minus className="w-3 h-3 text-[var(--text-muted)]" />;
}

function formatPrice(p: number) {
  return "$" + p.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 2 });
}

export default function DeepAnalysisPanel({ analysis, onClose }: DeepAnalysisPanelProps) {
  const [expandedLevels, setExpandedLevels] = useState<Set<string>>(
    new Set(analysis.levels.map(l => l.tf))
  );

  const toggleLevel = (tf: string) => {
    setExpandedLevels(prev => {
      const next = new Set(prev);
      if (next.has(tf)) next.delete(tf);
      else next.add(tf);
      return next;
    });
  };

  const isBull = analysis.signal_summary.direction === "bullish";
  const borderColor = isBull ? "#10b981" : "#ef4444";

  return (
    <div
      className="border-t-2 animate-fade-in max-h-[40vh] overflow-y-auto flex-shrink-0"
      style={{ background: "var(--bg-secondary)", borderColor }}
    >
      <div className="px-4 py-3">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div
              className="w-2.5 h-2.5 rounded-full animate-pulse"
              style={{ backgroundColor: borderColor }}
            />
            <h3 className="text-sm font-bold uppercase tracking-wider" style={{ color: borderColor }}>
              Deep Analysis for {isBull ? "LONG" : "SHORT"} {analysis.signal_summary.style}
            </h3>
            <span className={`text-xs px-2 py-0.5 rounded-md font-bold border ${
              analysis.signal_summary.grade === "A" ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30" :
              analysis.signal_summary.grade === "B" ? "bg-blue-500/15 text-blue-400 border-blue-500/30" :
              analysis.signal_summary.grade === "C" ? "bg-amber-500/15 text-amber-400 border-amber-500/30" :
              "bg-red-500/15 text-red-400 border-red-500/30"
            }`}>
              Grade {analysis.signal_summary.grade}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] transition-colors"
            title="Close panel"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Signal Summary Row */}
        <div className="grid grid-cols-4 gap-2 mb-4">
          <div className="bg-[var(--bg-tertiary)] rounded-lg px-2.5 py-1.5">
            <div className="text-[10px] text-[var(--text-muted)] mb-0.5">Entry</div>
            <div className="text-xs font-mono font-medium text-[var(--text-primary)]">
              {formatPrice(analysis.signal_summary.entry_price)}
            </div>
          </div>
          <div className="bg-red-500/5 rounded-lg px-2.5 py-1.5">
            <div className="text-[10px] text-red-400/60 mb-0.5">SL</div>
            <div className="text-xs font-mono font-medium text-red-400">
              {formatPrice(analysis.signal_summary.stop_loss)}
            </div>
          </div>
          {analysis.signal_summary.take_profits.slice(0, 2).map((tp, i) => (
            <div key={i} className={`rounded-lg px-2.5 py-1.5 ${i === 0 ? "bg-emerald-500/10" : "bg-emerald-500/5"}`}>
              <div className={`text-[10px] mb-0.5 ${i === 0 ? "text-emerald-400" : "text-emerald-400/50"}`}>
                {tp.label} ({tp.rr}R)
              </div>
              <div className={`text-xs font-mono font-medium ${i === 0 ? "text-emerald-400" : "text-emerald-400/70"}`}>
                {formatPrice(tp.price)}
              </div>
            </div>
          ))}
        </div>

        {/* Timing */}
        {analysis.timing && (
          <div className="flex items-center gap-3 mb-4 text-xs">
            <div className="flex items-center gap-1.5 bg-[var(--bg-tertiary)] px-2.5 py-1.5 rounded-lg">
              <Clock className="w-3 h-3 text-[var(--text-muted)]" />
              <span className="text-[var(--text-muted)]">Activated</span>
              <span className="font-mono text-[var(--text-primary)]">
                {new Date(analysis.timing.activated_at).toLocaleString()}
              </span>
            </div>
            <div className="bg-[var(--bg-tertiary)] px-2.5 py-1.5 rounded-lg">
              <span className="text-[var(--text-muted)]">TF </span>
              <span className="font-mono font-medium text-[var(--text-primary)]">
                {analysis.timing.entry_timeframe}
              </span>
            </div>
            {analysis.timing.bars_active > 0 && (
              <div className="bg-[var(--bg-tertiary)] px-2.5 py-1.5 rounded-lg">
                <span className="text-[var(--text-muted)]">Age </span>
                <span className="font-mono font-medium text-[var(--text-primary)]">
                  {analysis.timing.bars_active} cycle{analysis.timing.bars_active > 1 ? "s" : ""}
                </span>
              </div>
            )}
          </div>
        )}

        {/* TF Levels */}
        <div className="space-y-2 mb-4">
          {analysis.levels.map((level) => {
            const roleStyle = ROLE_COLORS[level.role] || ROLE_COLORS.Setup;
            const isExpanded = expandedLevels.has(level.tf);

            return (
              <div key={level.tf} className="border border-[var(--border-primary)] rounded-lg overflow-hidden">
                {/* Level header — clickable */}
                <button
                  onClick={() => toggleLevel(level.tf)}
                  className="w-full flex items-center gap-2 px-3 py-2 bg-[var(--bg-tertiary)] hover:bg-[var(--bg-tertiary)]/80 transition-colors"
                >
                  {isExpanded
                    ? <ChevronDown className="w-3 h-3 text-[var(--text-muted)]" />
                    : <ChevronRight className="w-3 h-3 text-[var(--text-muted)]" />
                  }
                  <span className="text-xs font-bold font-mono text-[var(--text-primary)]">{level.tf}</span>
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${roleStyle.bg} ${roleStyle.text} ${roleStyle.border}`}>
                    {level.role}
                  </span>
                  <TrendIcon trend={level.trend} />
                  <span className={`text-xs font-medium ${
                    level.trend === "bullish" ? "text-emerald-400" :
                    level.trend === "bearish" ? "text-red-400" :
                    "text-[var(--text-muted)]"
                  }`}>
                    {level.trend}
                  </span>
                  <span className="text-[10px] text-[var(--text-muted)] ml-auto">
                    {level.components.length} components
                  </span>
                </button>

                {/* Level body */}
                {isExpanded && (
                  <div className="px-3 py-2 space-y-2">
                    {/* Narrative */}
                    <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                      {level.narrative}
                    </p>

                    {/* Components */}
                    <div className="space-y-1">
                      {level.components.map((comp, i) => {
                        const cfg = COMPONENT_ICONS[comp.type] || COMPONENT_ICONS.swing;
                        const Icon = cfg.icon;
                        return (
                          <div key={i} className="flex items-start gap-2 py-0.5">
                            <Icon className="w-3 h-3 mt-0.5 shrink-0" style={{ color: cfg.color }} />
                            <span className="text-xs text-[var(--text-secondary)]">{comp.detail}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Entry Zone */}
        <div className="mb-4">
          <h4 className="text-[10px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
            Entry Zone
          </h4>
          <div className={`rounded-lg px-3 py-2 border ${isBull ? "bg-emerald-500/5 border-emerald-500/15" : "bg-red-500/5 border-red-500/15"}`}>
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-xs font-bold ${isBull ? "text-emerald-400" : "text-red-400"}`}>
                {analysis.entry_zone.type}
              </span>
              <span className="text-[10px] font-mono text-[var(--text-muted)]">
                {formatPrice(analysis.entry_zone.lower)} – {formatPrice(analysis.entry_zone.upper)}
              </span>
              {analysis.entry_zone.method && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium">
                  {analysis.entry_zone.method.toUpperCase()}
                </span>
              )}
            </div>
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
              {analysis.entry_zone.narrative}
            </p>
          </div>
        </div>

        {/* TP Logic */}
        <div className="mb-4">
          <h4 className="text-[10px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
            Take Profit Logic
          </h4>
          <div className="space-y-1">
            {analysis.tp_logic.map((tp, i) => (
              <div key={i} className="flex items-center justify-between bg-[var(--bg-tertiary)] rounded-lg px-3 py-1.5">
                <div className="flex items-center gap-2">
                  <Target className={`w-3 h-3 ${i === 0 ? "text-emerald-400" : "text-emerald-400/50"}`} />
                  <span className={`text-xs font-bold ${i === 0 ? "text-emerald-400" : "text-emerald-400/60"}`}>
                    {tp.label}
                  </span>
                  <span className="text-xs text-[var(--text-secondary)]">
                    targets {tp.target}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-[var(--text-primary)]">{formatPrice(tp.price)}</span>
                  <span className="text-[10px] font-mono text-emerald-400/70">{tp.rr}R</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Confluences */}
        <div>
          <h4 className="text-[10px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
            Confluences ({analysis.confluences.length})
          </h4>
          <div className="space-y-1">
            {analysis.confluences.map((conf, i) => {
              const strengthStyle = STRENGTH_CONFIG[conf.strength] || STRENGTH_CONFIG.medium;
              return (
                <div key={i} className="flex items-center justify-between bg-[var(--bg-tertiary)] rounded-lg px-3 py-1.5">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span className="text-xs font-medium text-[var(--text-primary)] whitespace-nowrap">
                      {conf.name}
                    </span>
                    <span className="text-xs text-[var(--text-muted)] truncate">
                      {conf.detail}
                    </span>
                  </div>
                  <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded shrink-0 ml-2 ${strengthStyle.bg} ${strengthStyle.text}`}>
                    {conf.strength}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
