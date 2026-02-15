"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Zap,
  Shield,
  BarChart3,
  AlertTriangle,
  Waves,
  Target,
  Gauge,
} from "lucide-react";
import { fetchQuantIntel, QuantIntelData } from "@/lib/api";

export default function QuantIntelPanel() {
  const [data, setData] = useState<QuantIntelData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const d = await fetchQuantIntel();
      setData(d);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 60_000); // Refresh every 60s
    return () => clearInterval(interval);
  }, [refresh]);

  if (loading) {
    return (
      <div className="glass-card rounded-xl p-4 animate-pulse">
        <div className="h-4 bg-[var(--bg-tertiary)] rounded w-1/3 mb-3"></div>
        <div className="h-20 bg-[var(--bg-tertiary)] rounded"></div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="glass-card rounded-xl p-4 text-red-400 text-xs">
        Quant Intel unavailable: {error || "no data"}
      </div>
    );
  }

  const opts = data.options_data || {};
  const cot = data.cot_data as any || {};
  const onchain = data.onchain_flow as any || {};
  const fng = data.fear_greed || {};
  const funding = data.cross_exchange_funding || {};
  const l2 = data.l2_depth || {};
  const liqWs = data.liquidation_ws || {};
  const whales = (data as any).whale_transactions || {};

  // Fear & Greed color
  const fngValue = fng.value ?? 50;
  const fngColor = fngValue < 25 ? "text-red-400" : fngValue < 45 ? "text-orange-400" : fngValue < 55 ? "text-yellow-400" : fngValue < 75 ? "text-lime-400" : "text-emerald-400";
  const fngBg = fngValue < 25 ? "bg-red-500/10" : fngValue < 45 ? "bg-orange-500/10" : fngValue < 55 ? "bg-yellow-500/10" : fngValue < 75 ? "bg-lime-500/10" : "bg-emerald-500/10";

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center gap-2 mb-1">
        <Zap className="w-4 h-4 text-violet-400" />
        <span className="text-sm font-bold text-violet-400">Quant Intelligence</span>
        <button onClick={refresh} className="ml-auto text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors">
          Refresh
        </button>
      </div>

      {/* Row 1: Fear & Greed + Volatility Regime */}
      <div className="grid grid-cols-2 gap-2">
        <div className={`rounded-lg p-2.5 border border-[var(--border-primary)] ${fngBg}`}>
          <div className="flex items-center gap-1 mb-1">
            <Gauge className="w-3 h-3 text-[var(--text-muted)]" />
            <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Fear & Greed</span>
          </div>
          <div className={`text-lg font-bold font-mono ${fngColor}`}>{fngValue}</div>
          <div className="text-[10px] text-[var(--text-muted)]">{fng.classification || "N/A"}</div>
        </div>

        <div className="rounded-lg p-2.5 border border-[var(--border-primary)] bg-[var(--bg-tertiary)]">
          <div className="flex items-center gap-1 mb-1">
            <Activity className="w-3 h-3 text-[var(--text-muted)]" />
            <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Liquidations</span>
          </div>
          <div className="text-lg font-bold font-mono text-[var(--text-primary)]">{liqWs.event_count || 0}</div>
          <div className="text-[10px] text-[var(--text-muted)]">
            {liqWs.running ? "WS active" : "WS offline"} | {liqWs.recent_30m?.length || 0} in 30m
          </div>
        </div>
      </div>

      {/* Row 2: Options Data */}
      {opts.pc_ratio != null && (
        <div className="rounded-lg p-2.5 border border-cyan-500/15 bg-cyan-500/5">
          <div className="flex items-center gap-1 mb-2">
            <Target className="w-3 h-3 text-cyan-400" />
            <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">Deribit Options</span>
          </div>
          <div className="grid grid-cols-4 gap-1.5 text-[10px]">
            <div className="text-center">
              <div className="text-[var(--text-muted)]">P/C</div>
              <div className="font-mono font-medium text-cyan-300">{opts.pc_ratio?.toFixed(2)}</div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Max Pain</div>
              <div className="font-mono font-medium text-cyan-300">
                {opts.max_pain ? `$${(opts.max_pain / 1000).toFixed(0)}K` : "—"}
              </div>
              {opts.max_pain_distance_pct != null && (
                <div className={`text-[9px] ${(opts.max_pain_distance_pct ?? 0) > 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {(opts.max_pain_distance_pct ?? 0) > 0 ? "+" : ""}{opts.max_pain_distance_pct?.toFixed(1)}%
                </div>
              )}
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">GEX</div>
              <div className={`font-mono font-medium ${(opts.net_gex ?? 0) > 0 ? "text-emerald-400" : "text-red-400"}`}>
                {opts.net_gex ? `${((opts.net_gex ?? 0) / 1e6).toFixed(1)}M` : "—"}
              </div>
              <div className="text-[9px] text-[var(--text-muted)]">
                {(opts.net_gex ?? 0) > 0 ? "Dampen" : "Amplify"}
              </div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">25Δ Skew</div>
              <div className={`font-mono font-medium ${
                (opts.skew_25d ?? 0) > 3 ? "text-red-400" :
                (opts.skew_25d ?? 0) < -3 ? "text-emerald-400" : "text-[var(--text-secondary)]"
              }`}>
                {opts.skew_25d != null ? `${opts.skew_25d > 0 ? "+" : ""}${opts.skew_25d.toFixed(1)}` : "—"}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Row 3: COT Positioning */}
      {cot.leveraged_net != null && (
        <div className="rounded-lg p-2.5 border border-amber-500/15 bg-amber-500/5">
          <div className="flex items-center gap-1 mb-2">
            <BarChart3 className="w-3 h-3 text-amber-400" />
            <span className="text-[10px] font-bold text-amber-400 uppercase tracking-wider">CME COT Positioning</span>
            <span className="text-[9px] text-[var(--text-muted)] ml-auto">{cot.report_date}</span>
          </div>
          <div className="grid grid-cols-3 gap-1.5 text-[10px]">
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Leveraged Net</div>
              <div className={`font-mono font-medium ${cot.leveraged_net > 0 ? "text-emerald-400" : "text-red-400"}`}>
                {cot.leveraged_net > 0 ? "+" : ""}{(cot.leveraged_net / 1000).toFixed(1)}K
              </div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Percentile</div>
              <div className={`font-mono font-medium ${
                cot.percentile < 20 ? "text-red-400" :
                cot.percentile > 80 ? "text-emerald-400" : "text-[var(--text-secondary)]"
              }`}>
                P{cot.percentile?.toFixed(0)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Signal</div>
              <div className={`font-mono font-medium ${
                cot.signal?.includes("extreme") ? "text-amber-400" : "text-[var(--text-secondary)]"
              }`}>
                {cot.signal?.replace("_", " ") || "—"}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Row 4: On-Chain Flow */}
      {onchain.net_flow != null && (
        <div className="rounded-lg p-2.5 border border-indigo-500/15 bg-indigo-500/5">
          <div className="flex items-center gap-1 mb-2">
            <Waves className="w-3 h-3 text-indigo-400" />
            <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider">On-Chain Flow</span>
          </div>
          <div className="grid grid-cols-3 gap-1.5 text-[10px]">
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Inflow</div>
              <div className="font-mono font-medium text-red-400">{(onchain.inflow_btc / 1000).toFixed(1)}K</div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Outflow</div>
              <div className="font-mono font-medium text-emerald-400">{(onchain.outflow_btc / 1000).toFixed(1)}K</div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Net</div>
              <div className={`font-mono font-medium ${onchain.net_flow > 0 ? "text-red-400" : "text-emerald-400"}`}>
                {onchain.net_flow > 0 ? "+" : ""}{(onchain.net_flow / 1000).toFixed(1)}K
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Row 5: Cross-Exchange Funding */}
      {funding.avg_rate != null && (
        <div className="rounded-lg p-2.5 border border-[var(--border-primary)] bg-[var(--bg-tertiary)]">
          <div className="flex items-center gap-1 mb-2">
            <Shield className="w-3 h-3 text-[var(--text-muted)]" />
            <span className="text-[10px] font-bold text-[var(--text-muted)] uppercase tracking-wider">Funding Rates</span>
          </div>
          <div className="grid grid-cols-3 gap-1.5 text-[10px]">
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Bybit</div>
              <div className={`font-mono font-medium ${(funding.bybit_rate ?? 0) > 0 ? "text-emerald-400" : "text-red-400"}`}>
                {((funding.bybit_rate ?? 0) * 100).toFixed(4)}%
              </div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">OKX</div>
              <div className={`font-mono font-medium ${(funding.okx_rate ?? 0) > 0 ? "text-emerald-400" : "text-red-400"}`}>
                {((funding.okx_rate ?? 0) * 100).toFixed(4)}%
              </div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Spread</div>
              <div className="font-mono font-medium text-[var(--text-secondary)]">
                {((funding.dispersion ?? 0) * 100).toFixed(4)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Row 6: L2 Depth */}
      {l2.best_bid != null && (
        <div className="rounded-lg p-2.5 border border-[var(--border-primary)] bg-[var(--bg-tertiary)]">
          <div className="flex items-center gap-1 mb-2">
            <BarChart3 className="w-3 h-3 text-[var(--text-muted)]" />
            <span className="text-[10px] font-bold text-[var(--text-muted)] uppercase tracking-wider">L2 Order Book</span>
          </div>
          <div className="grid grid-cols-3 gap-1.5 text-[10px]">
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Bid Wall</div>
              <div className="font-mono font-medium text-emerald-400">${((l2.bid_wall_usd ?? 0) / 1e6).toFixed(2)}M</div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Ask Wall</div>
              <div className="font-mono font-medium text-red-400">${((l2.ask_wall_usd ?? 0) / 1e6).toFixed(2)}M</div>
            </div>
            <div className="text-center">
              <div className="text-[var(--text-muted)]">Imbalance</div>
              <div className={`font-mono font-medium ${(l2.imbalance ?? 0) > 0 ? "text-emerald-400" : "text-red-400"}`}>
                {((l2.imbalance ?? 0) * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Row 7: Whale Activity */}
      {whales.whale_tx_count > 0 && (
        <div className="rounded-lg p-2.5 border border-purple-500/15 bg-purple-500/5">
          <div className="flex items-center gap-1 mb-1">
            <AlertTriangle className="w-3 h-3 text-purple-400" />
            <span className="text-[10px] font-bold text-purple-400 uppercase tracking-wider">Whale Activity</span>
          </div>
          <div className="text-[10px] text-[var(--text-secondary)]">
            {whales.whale_tx_count} large txs ({whales.total_whale_btc?.toFixed(0)} BTC)
          </div>
        </div>
      )}
    </div>
  );
}
