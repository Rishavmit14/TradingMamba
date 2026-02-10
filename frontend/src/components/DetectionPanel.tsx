"use client";

import { AnalysisResult } from "@/lib/types";

interface DetectionPanelProps {
  data: AnalysisResult | null;
}

function Badge({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <span className={`inline-block px-2 py-0.5 text-xs font-medium rounded ${color}`}>
      {children}
    </span>
  );
}

function StatRow({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="flex justify-between items-center py-1">
      <span className="text-gray-400 text-sm">{label}</span>
      <div className="text-right">
        <span className="text-white text-sm font-mono">{value}</span>
        {sub && <span className="text-gray-500 text-xs ml-1">{sub}</span>}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border-b border-gray-800 pb-3 mb-3">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">{title}</h3>
      {children}
    </div>
  );
}

export default function DetectionPanel({ data }: DetectionPanelProps) {
  if (!data) {
    return (
      <div className="p-4 text-gray-500 text-sm">
        Select a timeframe to view analysis...
      </div>
    );
  }

  const trendColors: Record<string, string> = {
    bullish: "bg-green-900/50 text-green-400",
    bearish: "bg-red-900/50 text-red-400",
    ranging: "bg-gray-800 text-gray-400",
  };

  const validBOS = data.bos_events.filter((b) => b.valid);
  const invalidBOS = data.bos_events.filter((b) => !b.valid);
  const validFVGs = data.fvgs.filter((f) => f.valid && !f.mitigated);
  const validOBs = data.order_blocks.filter((o) => o.valid && !o.mitigated);
  const takenIDMs = data.inducements.filter((i) => i.status === "taken");
  const activeIDMs = data.inducements.filter((i) => i.status === "active");
  const sweptPools = data.liquidity_pools.filter((l) => l.swept);

  return (
    <div className="p-4 overflow-y-auto h-full text-sm">
      {/* Trend */}
      <Section title="Trend">
        <div className="flex items-center gap-2 mb-2">
          <Badge color={trendColors[data.trend]}>{data.trend.toUpperCase()}</Badge>
          <span className="text-gray-400 text-xs">{data.timeframe}</span>
        </div>
        {data.climax_warning && (
          <div className="bg-yellow-900/30 border border-yellow-700/50 rounded px-2 py-1 text-yellow-400 text-xs">
            CLIMAX WARNING — ratio {data.climax_ratio.toFixed(2)}
          </div>
        )}
      </Section>

      {/* Swings */}
      <Section title={`Swings (${data.swings.length})`}>
        <div className="grid grid-cols-2 gap-1">
          {(["HH", "HL", "LH", "LL"] as const).map((cls) => {
            const count = data.swings.filter((s) => s.classification === cls).length;
            const color = cls === "HH" || cls === "HL" ? "text-green-400" : "text-red-400";
            return (
              <div key={cls} className="flex justify-between">
                <span className={`font-mono ${color}`}>{cls}</span>
                <span className="text-gray-400">{count}</span>
              </div>
            );
          })}
        </div>
        <div className="mt-1">
          <StatRow
            label="Valid SMC"
            value={data.swings.filter((s) => s.is_valid_smc).length}
            sub={`/ ${data.swings.length}`}
          />
        </div>
      </Section>

      {/* Inducements */}
      <Section title={`Inducements (${data.inducements.length})`}>
        <StatRow label="Active" value={activeIDMs.length} />
        <StatRow label="Taken" value={takenIDMs.length} />
        <StatRow label="Transferred" value={data.inducements.filter((i) => i.status === "transferred").length} />
      </Section>

      {/* Liquidity */}
      <Section title={`Liquidity (${data.liquidity_pools.length})`}>
        <StatRow label="Swept" value={sweptPools.length} sub={`/ ${data.liquidity_pools.length}`} />
        <StatRow
          label="Sweeps"
          value={data.liquidity_pools.filter((l) => l.event_type === "sweep").length}
        />
        <StatRow
          label="Grabs"
          value={data.liquidity_pools.filter((l) => l.event_type === "grab").length}
        />
      </Section>

      {/* BOS */}
      <Section title={`BOS (${data.bos_events.length})`}>
        <StatRow label="Valid" value={validBOS.length} />
        <StatRow label="Invalid" value={invalidBOS.length} />
        <div className="mt-1 grid grid-cols-2 gap-1">
          <div className="flex justify-between">
            <span className="text-green-400 font-mono text-xs">Bullish</span>
            <span className="text-gray-400">{data.bos_events.filter((b) => b.direction === "bullish").length}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-red-400 font-mono text-xs">Bearish</span>
            <span className="text-gray-400">{data.bos_events.filter((b) => b.direction === "bearish").length}</span>
          </div>
        </div>
      </Section>

      {/* CHoCH */}
      <Section title={`CHoCH (${data.choch_events.length})`}>
        {data.choch_events.length === 0 ? (
          <span className="text-gray-600 text-xs">None detected</span>
        ) : (
          data.choch_events.map((ch, i) => (
            <div key={i} className="bg-gray-900 rounded p-2 mb-1">
              <div className="flex items-center gap-2">
                <Badge color={ch.direction === "bullish" ? "bg-green-900/50 text-green-400" : "bg-red-900/50 text-red-400"}>
                  {ch.direction}
                </Badge>
                {ch.is_fake && <Badge color="bg-gray-700 text-gray-400">FAKE</Badge>}
                {ch.confirmed && <Badge color="bg-purple-900/50 text-purple-400">CONFIRMED</Badge>}
                {ch.has_climax_confluence && <Badge color="bg-yellow-900/50 text-yellow-400">CLIMAX</Badge>}
              </div>
              <div className="text-gray-500 text-xs mt-1">
                Confidence: {(ch.confidence * 100).toFixed(0)}%
              </div>
            </div>
          ))
        )}
      </Section>

      {/* FVGs */}
      <Section title={`FVGs (${data.fvgs.length})`}>
        <StatRow label="Valid (active)" value={validFVGs.length} />
        <StatRow label="Mitigated" value={data.fvgs.filter((f) => f.mitigated).length} />
        <StatRow label="From extreme" value={data.fvgs.filter((f) => f.from_extreme_candle).length} />
      </Section>

      {/* Order Blocks */}
      <Section title={`Order Blocks (${data.order_blocks.length})`}>
        <StatRow label="Valid (active)" value={validOBs.length} />
        <StatRow label="Mitigated" value={data.order_blocks.filter((o) => o.mitigated).length} />
        <StatRow label="Traps" value={data.order_blocks.filter((o) => o.is_trap).length} />
      </Section>

      {/* Premium/Discount */}
      <Section title="Premium / Discount">
        {data.premium_discount ? (
          <>
            <div className="flex items-center gap-2 mb-1">
              <Badge
                color={
                  data.premium_discount.zone === "discount"
                    ? "bg-green-900/50 text-green-400"
                    : data.premium_discount.zone === "premium"
                    ? "bg-red-900/50 text-red-400"
                    : "bg-yellow-900/50 text-yellow-400"
                }
              >
                {data.premium_discount.zone.toUpperCase()}
              </Badge>
              <span className="text-gray-400 text-xs">{data.premium_discount.depth_pct.toFixed(1)}% deep</span>
            </div>
            <StatRow label="Range High" value={`$${data.premium_discount.swing_high.toLocaleString()}`} />
            <StatRow label="Equilibrium" value={`$${data.premium_discount.equilibrium.toLocaleString()}`} />
            <StatRow label="Range Low" value={`$${data.premium_discount.swing_low.toLocaleString()}`} />
          </>
        ) : (
          <span className="text-gray-600 text-xs">Insufficient data</span>
        )}
      </Section>

      {/* Session */}
      <Section title="Session">
        {data.session ? (
          <>
            <StatRow label="Session" value={data.session.name.toUpperCase()} />
            <StatRow label="Kill Zone" value={data.session.is_kill_zone ? "YES" : "No"} />
            <StatRow label="Volatility" value={data.session.volatility_expectation} />
          </>
        ) : (
          <span className="text-gray-600 text-xs">N/A</span>
        )}
      </Section>
    </div>
  );
}
