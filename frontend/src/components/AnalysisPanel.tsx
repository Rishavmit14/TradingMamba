"use client";

import {
  X,
  TrendingUp,
  TrendingDown,
  ArrowUpRight,
  ArrowDownRight,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Info,
  Shield,
  Zap,
  Target,
  Layers,
  GitBranch,
} from "lucide-react";
import {
  AnalysisResult,
  SelectedElement,
  SwingPoint,
  BOS,
  CHoCH,
  Inducement,
  FVG,
  OrderBlock,
} from "@/lib/types";

interface AnalysisPanelProps {
  element: SelectedElement | null;
  data: AnalysisResult;
  onClose: () => void;
}

const TYPE_CONFIG: Record<string, { label: string; color: string; bg: string; border: string }> = {
  swing: { label: "Swing Point", color: "#f59e0b", bg: "rgba(245,158,11,0.08)", border: "rgba(245,158,11,0.25)" },
  bos:   { label: "Break of Structure", color: "#22d3ee", bg: "rgba(34,211,238,0.08)", border: "rgba(34,211,238,0.25)" },
  choch: { label: "Change of Character", color: "#a855f7", bg: "rgba(168,85,247,0.08)", border: "rgba(168,85,247,0.25)" },
  idm:   { label: "Inducement", color: "#60a5fa", bg: "rgba(96,165,250,0.08)", border: "rgba(96,165,250,0.25)" },
  fvg:   { label: "Fair Value Gap", color: "#10b981", bg: "rgba(16,185,129,0.08)", border: "rgba(16,185,129,0.25)" },
  ob:    { label: "Order Block", color: "#f97316", bg: "rgba(249,115,22,0.08)", border: "rgba(249,115,22,0.25)" },
};

function Badge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${
      ok
        ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
        : "bg-red-500/10 text-red-400 border border-red-500/20"
    }`}>
      {ok ? <CheckCircle2 className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
      {label}
    </span>
  );
}

function Row({ label, value, mono }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-start justify-between py-1.5 border-b border-[var(--border-primary)] last:border-b-0">
      <span className="text-xs text-[var(--text-muted)] shrink-0">{label}</span>
      <span className={`text-xs text-[var(--text-primary)] text-right ${mono ? "font-mono" : ""}`}>{value}</span>
    </div>
  );
}

function Explanation({ text, type }: { text: string; type?: "info" | "warn" | "success" }) {
  const Icon = type === "warn" ? AlertTriangle : type === "success" ? CheckCircle2 : Info;
  const colors = type === "warn"
    ? "bg-amber-500/5 border-amber-500/15 text-amber-300"
    : type === "success"
    ? "bg-emerald-500/5 border-emerald-500/15 text-emerald-300"
    : "bg-blue-500/5 border-blue-500/15 text-blue-300";
  return (
    <div className={`flex items-start gap-2 px-3 py-2 rounded-lg border text-xs leading-relaxed ${colors}`}>
      <Icon className="w-3.5 h-3.5 mt-0.5 shrink-0" />
      <span>{text}</span>
    </div>
  );
}

function formatPrice(p: number) {
  return "$" + p.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 2 });
}

function formatTime(ts: number) {
  return new Date(ts).toLocaleString("en-US", {
    timeZone: "America/New_York",
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

// ─── Swing Analysis ──────────────────────────────

function SwingAnalysis({ swing, data }: { swing: SwingPoint; data: AnalysisResult }) {
  const candle = data.candles[swing.candle_index];
  const isHigh = swing.swing_type === "swing_high";
  const relatedIDMs = data.inducements.filter(i => i.parent_swing_index === swing.candle_index);

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-x-6">
        <Row label="Type" value={isHigh ? "Swing High" : "Swing Low"} />
        <Row label="Classification" value={
          <span className="font-bold">{swing.classification.toUpperCase()}</span>
        } />
        <Row label="Price" value={formatPrice(swing.price)} mono />
        <Row label="Candle" value={`#${swing.candle_index}`} mono />
        {candle && <Row label="Time" value={formatTime(candle.timestamp)} />}
      </div>

      <div className="flex flex-wrap gap-1.5">
        <Badge ok={swing.is_valid_smc} label={swing.is_valid_smc ? "Valid SMC" : "Invalid SMC"} />
        <Badge ok={swing.is_strong} label={swing.is_strong ? "Strong" : "Weak"} />
        <Badge ok={swing.candle_closed_properly} label={swing.candle_closed_properly ? "Body Closed" : "No Body Close"} />
        <Badge ok={swing.idm_taken} label={swing.idm_taken ? "IDM Taken" : "No IDM"} />
      </div>

      {swing.is_valid_smc && swing.idm_taken && (
        <Explanation
          type="success"
          text="This swing is a valid SMC swing point. The inducement (IDM) was taken before this swing formed, confirming structural validity."
        />
      )}
      {!swing.is_valid_smc && (
        <Explanation
          type="warn"
          text="This swing is NOT a valid SMC swing point. It may be an inducement level or part of engineered liquidity. Trading signals from this swing carry higher risk."
        />
      )}
      {swing.is_strong && (
        <Explanation
          type="info"
          text="Strong swing point: the initiating candle created structural momentum without simultaneously sweeping opposing liquidity."
        />
      )}

      {relatedIDMs.length > 0 && (
        <div className="mt-2">
          <h4 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-2">Related Inducements</h4>
          {relatedIDMs.map((idm, i) => (
            <div key={i} className="bg-[var(--bg-tertiary)] rounded-lg px-3 py-2 mb-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs text-blue-400 font-medium">IDM @ {formatPrice(idm.price)}</span>
                <span className={`text-xs px-1.5 py-0.5 rounded ${
                  idm.status === "taken" ? "bg-emerald-500/10 text-emerald-400" :
                  idm.status === "active" ? "bg-blue-500/10 text-blue-400" :
                  "bg-amber-500/10 text-amber-400"
                }`}>
                  {idm.status.toUpperCase()}
                </span>
              </div>
              <div className="flex gap-3 mt-1">
                <span className="text-xs text-[var(--text-muted)]">
                  Body: {idm.body_closed ? "Closed" : "Wick only"}
                </span>
                <span className="text-xs text-[var(--text-muted)]">
                  {idm.is_major ? "Major IDM" : "Minor IDM"}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── BOS Analysis ──────────────────────────────

function BOSAnalysis({ bos, data }: { bos: BOS; data: AnalysisResult }) {
  const breakCandle = data.candles[bos.candle_index];
  const brokenSwing = data.swings.find(s => s.candle_index === bos.broken_swing_index);
  const isBull = bos.direction === "bullish";

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 mb-1">
        {isBull
          ? <ArrowUpRight className="w-4 h-4 text-emerald-400" />
          : <ArrowDownRight className="w-4 h-4 text-red-400" />
        }
        <span className={`text-sm font-bold ${isBull ? "text-emerald-400" : "text-red-400"}`}>
          {isBull ? "Bullish" : "Bearish"} Break of Structure
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-6">
        <Row label="Broken Price" value={formatPrice(bos.broken_price)} mono />
        <Row label="Break Candle" value={`#${bos.candle_index}`} mono />
        {breakCandle && <Row label="Time" value={formatTime(breakCandle.timestamp)} />}
        {brokenSwing && <Row label="Broken Swing" value={`${brokenSwing.classification} @ ${formatPrice(brokenSwing.price)}`} />}
      </div>

      <div className="flex flex-wrap gap-1.5">
        <Badge ok={bos.valid} label={bos.valid ? "Valid BOS" : "Invalid BOS"} />
        <Badge ok={bos.idm_body_closed} label={bos.idm_body_closed ? "IDM Body Closed" : "No IDM Body Close"} />
      </div>

      {bos.valid && bos.idm_body_closed && (
        <Explanation
          type="success"
          text="High-confidence BOS: IDM was taken with a full body close (not just wick), qualifying this as a Swing HH/LL tier break. This is the strongest form of BOS confirmation."
        />
      )}
      {bos.valid && !bos.idm_body_closed && (
        <Explanation
          type="info"
          text="Valid BOS but IDM was not taken with a body close. The break is structurally valid but carries slightly lower confidence than a body-closed BOS."
        />
      )}
      {!bos.valid && (
        <Explanation
          type="warn"
          text={`Invalid BOS${bos.invalidation_reason ? ": " + bos.invalidation_reason : ". The structural requirements for a valid break were not met."}`}
        />
      )}

      {breakCandle && (
        <div className="mt-2">
          <h4 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-2">Break Candle OHLC</h4>
          <div className="grid grid-cols-4 gap-2">
            {[
              { l: "O", v: breakCandle.open },
              { l: "H", v: breakCandle.high },
              { l: "L", v: breakCandle.low },
              { l: "C", v: breakCandle.close },
            ].map(({ l, v }) => (
              <div key={l} className="bg-[var(--bg-tertiary)] rounded-lg px-2 py-1.5 text-center">
                <div className="text-xs text-[var(--text-muted)]">{l}</div>
                <div className="text-xs font-mono text-[var(--text-primary)]">{formatPrice(v)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── CHoCH Analysis ──────────────────────────────

function CHoCHAnalysis({ choch, data }: { choch: CHoCH; data: AnalysisResult }) {
  const breakCandle = data.candles[choch.candle_index];
  const brokenSwing = data.swings.find(s => s.candle_index === choch.broken_swing_index);
  const isBull = choch.direction === "bullish";
  const modelLabel = choch.model === "sweep" ? "Sweep-Based (SBC)" : choch.model === "swing" ? "Swing-Based (MSS)" : "Unknown";

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 mb-1">
        {isBull
          ? <TrendingUp className="w-4 h-4 text-emerald-400" />
          : <TrendingDown className="w-4 h-4 text-red-400" />
        }
        <span className={`text-sm font-bold ${isBull ? "text-emerald-400" : "text-red-400"}`}>
          {isBull ? "Bullish" : "Bearish"} CHoCH
        </span>
        {choch.is_fake && (
          <span className="text-xs px-2 py-0.5 rounded bg-red-500/15 text-red-400 border border-red-500/25 font-medium">
            FAKE
          </span>
        )}
        {choch.confirmed && !choch.is_fake && (
          <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 font-medium">
            CONFIRMED
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-x-6">
        <Row label="Broken Price" value={formatPrice(choch.broken_price)} mono />
        <Row label="Confidence" value={`${(choch.confidence * 100).toFixed(0)}%`} mono />
        <Row label="Model" value={modelLabel} />
        <Row label="Break Candle" value={`#${choch.candle_index}`} mono />
        {breakCandle && <Row label="Time" value={formatTime(breakCandle.timestamp)} />}
        {brokenSwing && <Row label="Broken Swing" value={`${brokenSwing.classification} @ ${formatPrice(brokenSwing.price)}`} />}
      </div>

      <div className="flex flex-wrap gap-1.5">
        <Badge ok={!choch.is_fake} label={choch.is_fake ? "Fake CHoCH" : "Real CHoCH"} />
        <Badge ok={choch.confirmed} label={choch.confirmed ? "Confirmed" : "Unconfirmed"} />
        {choch.is_mss && <Badge ok={true} label="MSS (Liq Swept + Expansion)" />}
        <Badge ok={choch.has_climax_confluence} label={choch.has_climax_confluence ? "Climax Zone" : "No Climax"} />
      </div>

      {choch.is_fake && (
        <Explanation
          type="warn"
          text="This CHoCH has been identified as FAKE. The break likely represents a liquidity sweep or inducement trap rather than a genuine trend reversal. Avoid taking counter-trend entries based on this signal."
        />
      )}
      {choch.confirmed && !choch.is_fake && (
        <Explanation
          type="success"
          text="This CHoCH is CONFIRMED. Price created a follow-through structure after the break, validating the trend reversal. Look for entries on pullbacks in the new direction."
        />
      )}
      {!choch.confirmed && !choch.is_fake && (
        <Explanation
          type="info"
          text="This CHoCH is unconfirmed — the break occurred but follow-through structure has not yet developed. Wait for confirmation before entering."
        />
      )}

      {choch.model === "sweep" && (
        <Explanation
          type="info"
          text="Sweep-Based CHoCH (SBC): Only the wick went beyond the swing level. Confirmation requires the candle body to close beyond the valid pullback between the two swing points. No inducement needed."
        />
      )}
      {choch.model === "swing" && (
        <Explanation
          type="info"
          text="Swing-Based CHoCH (MSS): The candle body closed beyond the swing level. Confirmation requires price to create an inducement in the new direction and then break it (mini-BOS)."
        />
      )}
    </div>
  );
}

// ─── IDM Analysis ──────────────────────────────

function IDMAnalysis({ idm, data }: { idm: Inducement; data: AnalysisResult }) {
  const candle = data.candles[idm.candle_index];
  const parentSwing = data.swings.find(s => s.candle_index === idm.parent_swing_index);
  const takenCandle = idm.taken_at_candle != null ? data.candles[idm.taken_at_candle] : null;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 mb-1">
        <GitBranch className="w-4 h-4 text-blue-400" />
        <span className="text-sm font-bold text-blue-400">Inducement Level</span>
        <span className={`text-xs px-2 py-0.5 rounded font-medium ${
          idm.status === "taken" ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25" :
          idm.status === "active" ? "bg-blue-500/15 text-blue-400 border border-blue-500/25" :
          "bg-amber-500/15 text-amber-400 border border-amber-500/25"
        }`}>
          {idm.status.toUpperCase()}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-6">
        <Row label="Price" value={formatPrice(idm.price)} mono />
        <Row label="Candle" value={`#${idm.candle_index}`} mono />
        {candle && <Row label="Time" value={formatTime(candle.timestamp)} />}
        {parentSwing && <Row label="Parent Swing" value={`${parentSwing.classification} @ ${formatPrice(parentSwing.price)}`} />}
        {takenCandle && <Row label="Taken At" value={`#${idm.taken_at_candle} (${formatTime(takenCandle.timestamp)})`} />}
      </div>

      <div className="flex flex-wrap gap-1.5">
        <Badge ok={idm.body_closed} label={idm.body_closed ? "Body Closed" : "Wick Only"} />
        <Badge ok={idm.is_major} label={idm.is_major ? "Major IDM" : "Minor IDM"} />
      </div>

      {idm.is_major && (
        <Explanation
          type="success"
          text="Major IDM: This is the deepest pullback (first valid pullback on the left side of the swing). Major inducements carry 80-85% probability when properly swept."
        />
      )}
      {!idm.is_major && (
        <Explanation
          type="info"
          text="Minor IDM: This is a secondary pullback, not the deepest one. Minor inducements have lower probability than major ones."
        />
      )}
      {idm.body_closed && (
        <Explanation
          type="success"
          text="Body closed beyond the IDM level (not just wick sweep). This qualifies any subsequent BOS as a Swing HH/LL tier break — the highest confidence BOS type."
        />
      )}
      {idm.status === "transferred" && (
        <Explanation
          type="warn"
          text="This IDM was transferred from a previous swing — no pullback was found in the current range. The move to the parent swing was impulsive, which can indicate an inducement trap."
        />
      )}
    </div>
  );
}

// ─── FVG Analysis ──────────────────────────────

function FVGAnalysis({ fvg, data }: { fvg: FVG; data: AnalysisResult }) {
  const candle = data.candles[fvg.candle_index];
  const isBull = fvg.direction === "bullish";
  const gapSize = fvg.upper_price - fvg.lower_price;
  const midpoint = (fvg.upper_price + fvg.lower_price) / 2;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 mb-1">
        <Layers className="w-4 h-4 text-emerald-400" />
        <span className={`text-sm font-bold ${isBull ? "text-emerald-400" : "text-red-400"}`}>
          {isBull ? "Bullish" : "Bearish"} Fair Value Gap
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-6">
        <Row label="Upper Price" value={formatPrice(fvg.upper_price)} mono />
        <Row label="Lower Price" value={formatPrice(fvg.lower_price)} mono />
        <Row label="Midpoint" value={formatPrice(midpoint)} mono />
        <Row label="Gap Size" value={formatPrice(gapSize)} mono />
        <Row label="Candle" value={`#${fvg.candle_index}`} mono />
        {candle && <Row label="Time" value={formatTime(candle.timestamp)} />}
        {fvg.mitigated && fvg.mitigated_at_candle != null && (
          <Row label="Mitigated At" value={`#${fvg.mitigated_at_candle}`} mono />
        )}
      </div>

      <div className="flex flex-wrap gap-1.5">
        <Badge ok={fvg.valid} label={fvg.valid ? "Valid FVG" : "Invalid FVG"} />
        <Badge ok={fvg.from_extreme_candle} label={fvg.from_extreme_candle ? "Extreme Candle" : "Regular Candle"} />
        <Badge ok={!fvg.mitigated} label={fvg.mitigated ? "Mitigated" : "Active"} />
      </div>

      {fvg.valid && !fvg.mitigated && (
        <Explanation
          type="success"
          text={`Active FVG: This imbalance zone between ${formatPrice(fvg.lower_price)} and ${formatPrice(fvg.upper_price)} has not been filled. Price is likely to return to this zone to rebalance. The midpoint (${formatPrice(midpoint)}) is a key reaction level.`}
        />
      )}
      {fvg.mitigated && (
        <Explanation
          type="info"
          text="This FVG has been mitigated — price has returned and filled the imbalance. It is no longer an active zone for entries."
        />
      )}
      {fvg.from_extreme_candle && (
        <Explanation
          type="warn"
          text="V13 Rule: This FVG originates from an extreme candle. FVGs from extreme candles can be less reliable as they may represent exhaustion rather than continuation."
        />
      )}
    </div>
  );
}

// ─── OB Analysis ──────────────────────────────

function OBAnalysis({ ob, data }: { ob: OrderBlock; data: AnalysisResult }) {
  const startCandle = data.candles[ob.candle_index_start];
  const isBull = ob.direction === "bullish";
  const zoneSize = ob.upper_price - ob.lower_price;
  const midpoint = (ob.upper_price + ob.lower_price) / 2;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 mb-1">
        <Shield className="w-4 h-4 text-orange-400" />
        <span className={`text-sm font-bold ${isBull ? "text-blue-400" : "text-orange-400"}`}>
          {isBull ? "Bullish" : "Bearish"} Order Block
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-6">
        <Row label="Upper Price" value={formatPrice(ob.upper_price)} mono />
        <Row label="Lower Price" value={formatPrice(ob.lower_price)} mono />
        <Row label="Midpoint" value={formatPrice(midpoint)} mono />
        <Row label="Zone Size" value={formatPrice(zoneSize)} mono />
        <Row label="Candles" value={`#${ob.candle_index_start} - #${ob.candle_index_end}`} mono />
        {startCandle && <Row label="Time" value={formatTime(startCandle.timestamp)} />}
        {ob.mitigated && ob.mitigated_at_candle != null && (
          <Row label="Mitigated At" value={`#${ob.mitigated_at_candle}`} mono />
        )}
      </div>

      <div className="flex flex-wrap gap-1.5">
        <Badge ok={ob.valid} label={ob.valid ? "Valid OB" : "Invalid OB"} />
        <Badge ok={ob.has_fvg} label={ob.has_fvg ? "Has FVG" : "No FVG"} />
        <Badge ok={ob.swept_liquidity} label={ob.swept_liquidity ? "Swept Liquidity" : "No Sweep"} />
        <Badge ok={!ob.is_trap} label={ob.is_trap ? "Trap OB" : "Clean OB"} />
        <Badge ok={!ob.mitigated} label={ob.mitigated ? "Mitigated" : "Active"} />
      </div>

      {ob.valid && ob.has_fvg && ob.swept_liquidity && (
        <Explanation
          type="success"
          text="High-quality OB: Both validation rules are met. Rule 1: Previous candle liquidity was swept. Rule 2: An FVG exists adjacent to this block. This is the strongest form of order block."
        />
      )}
      {ob.valid && !ob.has_fvg && (
        <Explanation
          type="warn"
          text="Rule 2 violation: No FVG exists adjacent to this order block. While the OB is structurally valid, the absence of an imbalance gap reduces its reliability as a reaction zone."
        />
      )}
      {ob.valid && !ob.swept_liquidity && (
        <Explanation
          type="warn"
          text="Rule 1 violation: This OB did not sweep the previous candle's liquidity. Institutional order placement typically involves taking out liquidity first. Lower probability reaction zone."
        />
      )}
      {ob.is_trap && (
        <Explanation
          type="warn"
          text="TRAP: This order block is part of an inducement or engineered liquidity setup. It may attract traders before reversing against them."
        />
      )}
      {ob.mitigated && (
        <Explanation
          type="info"
          text="This order block has been mitigated — price has returned to this zone. It is no longer expected to provide fresh reactions."
        />
      )}
    </div>
  );
}

// ─── Main Component ──────────────────────────────

export default function AnalysisPanel({ element, data, onClose }: AnalysisPanelProps) {
  if (!element) return null;

  const config = TYPE_CONFIG[element.type];

  let content: React.ReactNode = null;

  switch (element.type) {
    case "swing": {
      const swing = data.swings[element.index];
      if (swing) content = <SwingAnalysis swing={swing} data={data} />;
      break;
    }
    case "bos": {
      const bos = data.bos_events[element.index];
      if (bos) content = <BOSAnalysis bos={bos} data={data} />;
      break;
    }
    case "choch": {
      const choch = data.choch_events[element.index];
      if (choch) content = <CHoCHAnalysis choch={choch} data={data} />;
      break;
    }
    case "idm": {
      const idm = data.inducements[element.index];
      if (idm) content = <IDMAnalysis idm={idm} data={data} />;
      break;
    }
    case "fvg": {
      const fvg = data.fvgs[element.index];
      if (fvg) content = <FVGAnalysis fvg={fvg} data={data} />;
      break;
    }
    case "ob": {
      const ob = data.order_blocks[element.index];
      if (ob) content = <OBAnalysis ob={ob} data={data} />;
      break;
    }
  }

  if (!content) return null;

  return (
    <div
      className="border-t-2 animate-fade-in max-h-[30vh] overflow-y-auto flex-shrink-0"
      style={{ background: "var(--bg-secondary)", borderColor: config.color }}
    >
      <div className="px-4 py-3">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div
              className="w-2.5 h-2.5 rounded-full animate-pulse"
              style={{ backgroundColor: config.color }}
            />
            <h3
              className="text-sm font-bold uppercase tracking-wider"
              style={{ color: config.color }}
            >
              {config.label} Analysis
            </h3>
            <span className="text-xs font-mono text-[var(--text-muted)] bg-[var(--bg-tertiary)] px-1.5 py-0.5 rounded">
              {data.timeframe}
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

        {/* Content */}
        {content}
      </div>
    </div>
  );
}
