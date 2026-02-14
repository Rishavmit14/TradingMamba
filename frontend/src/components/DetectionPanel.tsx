"use client";

import { useState } from "react";
import {
  ChevronDown,
  TrendingUp,
  TrendingDown,
  Target,
  Crosshair,
  Layers,
  ArrowUpDown,
  GitBranch,
  BarChart,
  Gauge,
  Clock,
  AlertTriangle,
} from "lucide-react";
import { AnalysisResult } from "@/lib/types";

interface DetectionPanelProps {
  data: AnalysisResult | null;
}

function Badge({ children, variant }: { children: React.ReactNode; variant: "green" | "red" | "gray" | "amber" | "purple" | "blue" | "cyan" }) {
  const styles = {
    green: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    red: "bg-red-500/10 text-red-400 border-red-500/20",
    gray: "bg-[var(--bg-tertiary)] text-[var(--text-muted)] border-[var(--border-primary)]",
    amber: "bg-amber-500/10 text-amber-400 border-amber-500/20",
    purple: "bg-purple-500/10 text-purple-400 border-purple-500/20",
    blue: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    cyan: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-md border ${styles[variant]}`}>
      {children}
    </span>
  );
}

function StatRow({ label, value, sub, color }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div className="flex justify-between items-center py-1.5">
      <span className="text-[var(--text-muted)] text-xs">{label}</span>
      <div className="text-right flex items-center gap-1">
        <span className={`text-xs font-mono font-medium ${color || "text-[var(--text-primary)]"}`}>{value}</span>
        {sub && <span className="text-[var(--text-muted)] text-xs">{sub}</span>}
      </div>
    </div>
  );
}

function ProgressBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="w-full h-1 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
      <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  );
}

function Section({
  title,
  icon: Icon,
  count,
  color,
  defaultOpen = true,
  children,
}: {
  title: string;
  icon: any;
  count?: number;
  color: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-[var(--border-primary)]">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between w-full px-4 py-2.5 hover:bg-[var(--bg-tertiary)]/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Icon className="w-3.5 h-3.5" style={{ color }} />
          <span className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider">{title}</span>
          {count !== undefined && (
            <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-[var(--bg-tertiary)] text-[var(--text-muted)]">
              {count}
            </span>
          )}
        </div>
        <ChevronDown
          className={`w-3.5 h-3.5 text-[var(--text-muted)] transition-transform duration-200 ${open ? "" : "-rotate-90"}`}
        />
      </button>
      {open && <div className="px-4 pb-3 animate-fade-in">{children}</div>}
    </div>
  );
}

export default function DetectionPanel({ data }: DetectionPanelProps) {
  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center h-full py-12 px-4">
        <div className="w-10 h-10 rounded-xl bg-[var(--bg-tertiary)] flex items-center justify-center mb-3">
          <BarChart className="w-5 h-5 text-[var(--text-muted)]" />
        </div>
        <p className="text-sm text-[var(--text-muted)] text-center">
          Select a timeframe to view analysis
        </p>
      </div>
    );
  }

  const validBOS = data.bos_events.filter((b) => b.valid);
  const invalidBOS = data.bos_events.filter((b) => !b.valid);
  const validFVGs = data.fvgs.filter((f) => f.valid && !f.mitigated);
  const validOBs = data.order_blocks.filter((o) => o.valid && !o.mitigated);
  const takenIDMs = data.inducements.filter((i) => i.status === "taken");
  const activeIDMs = data.inducements.filter((i) => i.status === "active");
  const validSwings = data.swings.filter((s) => s.is_valid_smc);

  return (
    <div className="text-sm">
      {/* Trend */}
      <Section title="Trend" icon={TrendingUp} color="#10b981" defaultOpen={true}>
        <div className="flex items-center gap-2 mb-2">
          <Badge variant={data.trend === "bullish" ? "green" : data.trend === "bearish" ? "red" : "gray"}>
            {data.trend === "bullish" && <TrendingUp className="w-3 h-3 mr-1" />}
            {data.trend === "bearish" && <TrendingDown className="w-3 h-3 mr-1" />}
            {data.trend.toUpperCase()}
          </Badge>
        </div>
        {data.vsa_active && (
          <div className="flex items-center gap-2 bg-emerald-500/5 border border-emerald-500/15 rounded-lg px-3 py-2">
            <BarChart className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
            <div>
              <p className="text-xs font-medium text-emerald-400">VSA Absorption</p>
              <p className="text-xs text-[var(--text-muted)]">{data.vsa_absorptions.length} pattern(s) detected</p>
            </div>
          </div>
        )}
      </Section>

      {/* Swings */}
      <Section title="Swings" icon={ArrowUpDown} count={data.swings.length} color="#f59e0b" defaultOpen={true}>
        <div className="grid grid-cols-2 gap-2 mb-2">
          {(["HH", "HL", "LH", "LL"] as const).map((cls) => {
            const count = data.swings.filter((s) => s.classification === cls).length;
            const isUp = cls === "HH" || cls === "HL";
            return (
              <div key={cls} className={`flex items-center justify-between px-2.5 py-1.5 rounded-md ${isUp ? "bg-emerald-500/5" : "bg-red-500/5"}`}>
                <span className={`font-mono text-xs font-semibold ${isUp ? "text-emerald-400" : "text-red-400"}`}>{cls}</span>
                <span className="text-xs font-mono text-[var(--text-secondary)]">{count}</span>
              </div>
            );
          })}
        </div>
        <div className="space-y-1">
          <StatRow label="Valid SMC" value={validSwings.length} sub={`/ ${data.swings.length}`} />
          <ProgressBar value={validSwings.length} max={data.swings.length} color="#f59e0b" />
        </div>
      </Section>

      {/* Inducements */}
      <Section title="Inducements" icon={Target} count={data.inducements.length} color="#60a5fa" defaultOpen={true}>
        <div className="space-y-1">
          <StatRow label="Active" value={activeIDMs.length} color="text-blue-400" />
          <StatRow label="Taken" value={takenIDMs.length} color="text-emerald-400" />
          <StatRow label="Transferred" value={data.inducements.filter((i) => i.status === "transferred").length} color="text-amber-400" />
        </div>
        {data.inducements.length > 0 && (
          <div className="mt-2">
            <ProgressBar value={takenIDMs.length} max={data.inducements.length} color="#60a5fa" />
            <p className="text-xs text-[var(--text-muted)] mt-1">
              {data.inducements.length > 0 ? Math.round((takenIDMs.length / data.inducements.length) * 100) : 0}% swept
            </p>
          </div>
        )}
      </Section>

      {/* Liquidity */}
      <Section title="Liquidity" icon={Layers} count={data.liquidity_pools.length} color="#8b5cf6" defaultOpen={false}>
        <div className="space-y-1">
          <StatRow label="Swept" value={data.liquidity_pools.filter((l) => l.swept).length} sub={`/ ${data.liquidity_pools.length}`} />
          <StatRow label="Sweeps" value={data.liquidity_pools.filter((l) => l.event_type === "sweep").length} />
          <StatRow label="Grabs" value={data.liquidity_pools.filter((l) => l.event_type === "grab").length} />
        </div>
      </Section>

      {/* BOS */}
      <Section title="BOS" icon={Crosshair} count={data.bos_events.length} color="#22d3ee" defaultOpen={true}>
        <div className="flex gap-2 mb-2">
          <div className="flex-1 bg-emerald-500/5 rounded-md px-2.5 py-2 text-center">
            <p className="text-lg font-mono font-semibold text-emerald-400">{validBOS.length}</p>
            <p className="text-xs text-[var(--text-muted)]">Valid</p>
          </div>
          <div className="flex-1 bg-red-500/5 rounded-md px-2.5 py-2 text-center">
            <p className="text-lg font-mono font-semibold text-red-400">{invalidBOS.length}</p>
            <p className="text-xs text-[var(--text-muted)]">Invalid</p>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-emerald-400 flex items-center gap-1"><TrendingUp className="w-3 h-3" /> Bull</span>
            <span className="text-xs font-mono text-[var(--text-secondary)]">{data.bos_events.filter((b) => b.direction === "bullish").length}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-red-400 flex items-center gap-1"><TrendingDown className="w-3 h-3" /> Bear</span>
            <span className="text-xs font-mono text-[var(--text-secondary)]">{data.bos_events.filter((b) => b.direction === "bearish").length}</span>
          </div>
        </div>
      </Section>

      {/* CHoCH */}
      <Section title={`CHoCH${data.choch_events.filter(c => c.is_mss).length > 0 ? ` (${data.choch_events.filter(c => c.is_mss).length} MSS)` : ""}`} icon={GitBranch} count={data.choch_events.length} color="#a855f7" defaultOpen={true}>
        {data.choch_events.length === 0 ? (
          <p className="text-xs text-[var(--text-muted)] py-1">None detected</p>
        ) : (
          <div className="space-y-2">
            {data.choch_events.map((ch, i) => (
              <div key={i} className="glass-card rounded-lg p-2.5">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <Badge variant={ch.direction === "bullish" ? "green" : "red"}>
                    {ch.direction === "bullish" ? <TrendingUp className="w-3 h-3 mr-0.5" /> : <TrendingDown className="w-3 h-3 mr-0.5" />}
                    {ch.direction}
                  </Badge>
                  {ch.is_mss && <Badge variant="amber">MSS</Badge>}
                  {ch.is_fake && <Badge variant="gray">FAKE</Badge>}
                  {!ch.is_mss && ch.confirmed && <Badge variant="purple">CONFIRMED</Badge>}
                  {ch.has_vsa_confluence && <Badge variant="green">VSA</Badge>}
                </div>
                <div className="mt-1.5 flex items-center justify-between">
                  <span className="text-xs text-[var(--text-muted)]">Confidence</span>
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${ch.confidence * 100}%`,
                          backgroundColor: ch.confidence > 0.7 ? "#10b981" : ch.confidence > 0.4 ? "#f59e0b" : "#ef4444",
                        }}
                      />
                    </div>
                    <span className="text-xs font-mono text-[var(--text-secondary)]">{(ch.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* FVGs */}
      <Section title="FVG" icon={Layers} count={data.fvgs.length} color="#10b981" defaultOpen={false}>
        <div className="space-y-1">
          <StatRow label="Valid (active)" value={validFVGs.length} color="text-emerald-400" />
          <StatRow label="Mitigated" value={data.fvgs.filter((f) => f.mitigated).length} color="text-[var(--text-muted)]" />
          <StatRow label="From extreme" value={data.fvgs.filter((f) => f.from_extreme_candle).length} color="text-amber-400" />
        </div>
      </Section>

      {/* Order Blocks */}
      <Section title="Order Blocks" icon={BarChart} count={data.order_blocks.length} color="#3b82f6" defaultOpen={false}>
        <div className="space-y-1">
          <StatRow label="Valid (active)" value={validOBs.length} color="text-blue-400" />
          <StatRow label="Mitigated" value={data.order_blocks.filter((o) => o.mitigated).length} color="text-[var(--text-muted)]" />
          <StatRow label="Traps" value={data.order_blocks.filter((o) => o.is_trap).length} color="text-red-400" />
        </div>
      </Section>

      {/* Premium/Discount */}
      <Section title="Premium / Discount" icon={Gauge} color="#eab308" defaultOpen={true}>
        {data.premium_discount ? (
          <>
            <div className="flex items-center gap-2 mb-2">
              <Badge
                variant={
                  data.premium_discount.zone === "discount" ? "green"
                  : data.premium_discount.zone === "premium" ? "red"
                  : "amber"
                }
              >
                {data.premium_discount.zone.toUpperCase()}
              </Badge>
              <span className="text-xs text-[var(--text-muted)]">{data.premium_discount.depth_pct.toFixed(1)}% deep</span>
            </div>
            <div className="space-y-1">
              <StatRow label="Range High" value={`$${data.premium_discount.swing_high.toLocaleString()}`} color="text-red-400" />
              <StatRow label="Equilibrium" value={`$${data.premium_discount.equilibrium.toLocaleString()}`} color="text-amber-400" />
              <StatRow label="Range Low" value={`$${data.premium_discount.swing_low.toLocaleString()}`} color="text-emerald-400" />
            </div>
            {/* Visual zone indicator */}
            <div className="mt-2 relative h-2 rounded-full overflow-hidden bg-gradient-to-r from-emerald-500/20 via-amber-500/20 to-red-500/20">
              <div
                className="absolute top-0 w-1.5 h-full bg-white rounded-full shadow"
                style={{ left: `${Math.min(Math.max(data.premium_discount.depth_pct, 2), 98)}%` }}
              />
            </div>
          </>
        ) : (
          <p className="text-xs text-[var(--text-muted)] py-1">Insufficient data</p>
        )}
      </Section>

      {/* Session */}
      <Section title="Session" icon={Clock} color="#8b5cf6" defaultOpen={false}>
        {data.session ? (
          <div className="space-y-1">
            <StatRow label="Session" value={data.session.name.toUpperCase()} />
            <StatRow
              label="Kill Zone"
              value={data.session.is_kill_zone ? "YES" : "No"}
              color={data.session.is_kill_zone ? "text-purple-400" : "text-[var(--text-muted)]"}
            />
            <StatRow label="Volatility" value={data.session.volatility_expectation} />
          </div>
        ) : (
          <p className="text-xs text-[var(--text-muted)] py-1">N/A</p>
        )}
      </Section>
    </div>
  );
}
