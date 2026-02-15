/** TypeScript types matching the backend Python models. */

export type Direction = "bullish" | "bearish";
export type SwingType = "swing_high" | "swing_low";
export type SwingClassification = "HH" | "HL" | "LH" | "LL" | "unclassified";
export type TrendState = "bullish" | "bearish" | "ranging";
export type IDMStatus = "active" | "taken" | "transferred";
export type LiquidityType = "buy_side" | "sell_side";
export type LiquiditySource = "equal_highs" | "equal_lows" | "swing_extreme" | "trendline" | "idm_level";
export type LiquidityEvent = "sweep" | "grab";
export type ZoneType = "premium" | "discount" | "equilibrium";
export type SignalGrade = "A" | "B" | "C" | "D" | "M";
export type MSSGrade = "none" | "standard" | "a_plus" | "a_plus_plus";
export type VolatilityRegime = "low" | "normal" | "high" | "extreme";

export interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  index: number;
}

export interface SwingPoint {
  candle_index: number;
  price: number;
  swing_type: SwingType;
  classification: SwingClassification;
  is_valid_smc: boolean;
  is_strong: boolean;
  idm_taken: boolean;
  candle_closed_properly: boolean;
}

export interface Inducement {
  candle_index: number;
  price: number;
  parent_swing_index: number;
  status: IDMStatus;
  taken_at_candle: number | null;
  body_closed: boolean;
  is_major: boolean;
}

export interface LiquidityPool {
  price_level: number;
  pool_type: LiquidityType;
  source: LiquiditySource;
  candle_indices: number[];
  swept: boolean;
  swept_at_candle: number | null;
  event_type: LiquidityEvent | null;
}

export interface BOS {
  candle_index: number;
  direction: Direction;
  broken_swing_index: number;
  broken_price: number;
  valid: boolean;
  invalidation_reason: string | null;
  idm_body_closed: boolean;
}

export interface CHoCH {
  candle_index: number;
  direction: Direction;
  broken_swing_index: number;
  broken_price: number;
  confidence: number;
  has_vsa_confluence: boolean;
  is_fake: boolean;
  confirmed: boolean;
  model: string;
  is_mss: boolean;
  mss_grade: MSSGrade;
}

export interface FVG {
  candle_index: number;
  upper_price: number;
  lower_price: number;
  direction: Direction;
  valid: boolean;
  from_extreme_candle: boolean;
  mitigated: boolean;
  mitigated_at_candle: number | null;
  is_inverted: boolean;
  inverted_at_candle: number | null;
}

export interface OrderBlock {
  candle_index_start: number;
  candle_index_end: number;
  upper_price: number;
  lower_price: number;
  direction: Direction;
  valid: boolean;
  has_fvg: boolean;
  swept_liquidity: boolean;
  is_trap: boolean;
  mitigated: boolean;
  mitigated_at_candle: number | null;
}

export interface PremiumDiscount {
  swing_high: number;
  swing_low: number;
  equilibrium: number;
  zone: ZoneType;
  depth_pct: number;
  fibonacci_levels: Record<string, number>;
  closest_fib: string;
  fib_distance_pct: number;
  is_fib_qualified: boolean;
  swing_high_index: number;
  swing_low_index: number;
}

export interface Session {
  name: string;
  is_kill_zone: boolean;
  volatility_expectation: string;
}

export interface TakeProfit {
  price: number;
  rr: number;
  label: string;  // "TP1", "TP2", "TP3"
}

// ── Quant Types ──

export interface QuantScore {
  alpha_score: number;       // -100 to +100 (55% weight)
  micro_score: number;       // -100 to +100 (20% weight)
  risk_tradability: number;  // 0 to 100 (15% weight)
  execution_score: number;   // -100 to +100 (10% weight)
  combined_score: number;    // -100 to +100 (weighted sum)
  confirmations: number;     // 0-3+ quant confirmations
  contradictions: number;    // 0-3+ quant contradictions
  grade_change: number;      // -2 to +2 (grade tiers shifted)
  components: Record<string, unknown>;
}

export interface LiquidationEvent {
  timestamp: number;
  side: string;
  price: number;
  qty: number;
  qty_usd: number;
}

export interface QuantContext {
  vpin: number;
  vpin_history: number[];
  atr_m15: number;
  atr_h4: number;
  atr_d1: number;
  dvol: number;
  volatility_regime: VolatilityRegime;
  recent_liquidations: LiquidationEvent[];
  liquidation_clusters: Record<string, unknown>[];
  cross_exchange_funding: {
    bybit_rate?: number;
    okx_rate?: number;
    avg_rate?: number;
    dispersion?: number;
  };
  options_data: {
    pc_ratio?: number;
    total_put_oi?: number;
    total_call_oi?: number;
    max_pain?: number;
    net_gex?: number;
    skew_25d?: number | null;
    max_pain_distance_pct?: number;
  };
  cot_data: Record<string, unknown>;
  onchain_flow: Record<string, unknown>;
  fear_greed: {
    value?: number;
    classification?: string;
    timestamp?: number;
  };
}

export interface TradingSignal {
  direction: Direction;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  take_profits?: TakeProfit[];
  risk_reward_ratio: number;
  confidence_score: number;
  grade: SignalGrade;
  confluences: string[];
  timeframe: string;
  entry_method: string | null;
  pattern_type: string;
  timestamp: number;
  vsa_absorption: boolean;
  is_counter_trend: boolean;
  mss_quality: string;
  trading_style: string;
  // Signal lifecycle fields (populated by SignalStore)
  signal_id?: string;
  trading_styles?: string[];
  status?: string;           // "active" | "sl_hit" | "tp_hit" | "expired"
  created_at?: number;
  bars_active?: number;
  trigger_candle_index?: number;
  // Quant fields (populated in quant mode only)
  quant_score?: QuantScore;
  atr_stop_loss?: number;
  atr_take_profit?: number;
  position_size_pct?: number;
  quant_confluences?: string[];
  suppressed?: boolean;
}

export interface DetectorVisibility {
  swings: boolean;
  idm: boolean;
  bos: boolean;
  choch: boolean;
  fvg: boolean;
  ob: boolean;
  pd: boolean;
  futures: boolean;
}

export interface AnalysisResult {
  timeframe: string;
  trend: TrendState;
  trade_bias: TrendState;  // HTF-derived trade direction (W1→D1→M15 fallback)
  swings: SwingPoint[];
  inducements: Inducement[];
  liquidity_pools: LiquidityPool[];
  bos_events: BOS[];
  choch_events: CHoCH[];
  fvgs: FVG[];
  order_blocks: OrderBlock[];
  premium_discount: PremiumDiscount | null;
  session: Session | null;
  vsa_active: boolean;
  vsa_absorptions: { candle_index: number; direction: Direction; volume_ratio: number; confirmation: boolean }[];
  signals: TradingSignal[];
  all_style_signals: TradingSignal[];
  candles: Candle[];
  futures_context?: FuturesContext | null;
  quant_context?: QuantContext | null;
}

export type SelectedElementType = "swing" | "bos" | "choch" | "idm" | "fvg" | "ob";

export interface SelectedElement {
  type: SelectedElementType;
  index: number;
  candle_index: number;
}

export interface ClickCandidate extends SelectedElement {
  dist: number;
  label: string;
}

export interface ChartClickResult {
  candidates: ClickCandidate[];
  clickX: number;
  clickY: number;
}

// ── Deep Analysis Types ──

export interface DeepAnalysisComponent {
  type: string;        // "swing" | "bos" | "choch" | "idm" | "fvg" | "ob" | "liquidity" | "pd" | "vsa"
  detail: string;      // "LH at $85,309 (weak swing)"
  price: number | null;
}

export interface DeepAnalysisLevel {
  tf: string;          // "W1", "D1", etc.
  role: string;        // "Bias" | "Setup" | "Entry"
  trend: TrendState;
  narrative: string;   // Human-readable paragraph
  components: DeepAnalysisComponent[];
}

export interface DeepAnalysisEntryZone {
  type: string;
  upper: number;
  lower: number;
  method: string;
  narrative: string;
}

export interface DeepAnalysisTP {
  label: string;
  price: number;
  target: string;
  rr: number;
}

export interface DeepAnalysisConfluence {
  name: string;
  detail: string;
  strength: string;   // "strong" | "medium" | "weak"
}

export interface DeepAnalysisTiming {
  activated_at: number;
  trigger_timestamp: number;
  trigger_candle_index: number;
  bars_active: number;
  entry_timeframe: string;
}

export interface DeepAnalysis {
  signal_id: string;
  signal_summary: {
    direction: Direction;
    entry_price: number;
    stop_loss: number;
    take_profits: TakeProfit[];
    grade: string;
    style: string;
  };
  levels: DeepAnalysisLevel[];
  entry_zone: DeepAnalysisEntryZone;
  tp_logic: DeepAnalysisTP[];
  confluences: DeepAnalysisConfluence[];
  timing: DeepAnalysisTiming;
}

// ── Phase 3: Backtest Types ──

export type TradeOutcome = "win" | "loss" | "timeout";
export type EngineMode = "smc" | "quant";
export type AppTab = "live" | "backtest" | "performance" | "signals" | "demo" | "market_intel" | "quant_analysis" | "algo_bias";

export interface TradeRecord {
  direction: Direction;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  risk_reward_ratio: number;
  grade: SignalGrade;
  confidence_score: number;
  confluences: string[];
  entry_method: string;
  pattern_type: string;
  is_counter_trend: boolean;
  vsa_absorption: boolean;
  entry_candle_idx: number;
  entry_timestamp: number;
  outcome: TradeOutcome;
  exit_price: number;
  exit_candle_idx: number;
  exit_timestamp: number;
  bars_held: number;
  pnl_pct: number;
  max_favorable_excursion: number;
  max_adverse_excursion: number;
}

export interface GradeBreakdown {
  grade: string;
  trades: number;
  wins: number;
  losses: number;
  timeouts: number;
  win_rate: number;
  avg_pnl: number;
  recommendation: string;
}

export interface ConfluenceEdge {
  name: string;
  present_wr: number;
  absent_wr: number;
  edge: number;
  present_count: number;
  absent_count: number;
}

// ── Phase 4: Signals Tab Types ──

export type ChecklistStatus = "passed" | "failed" | "pending";

export interface ChecklistItem {
  rule?: number;
  step?: number;
  name: string;
  status: ChecklistStatus;
  detail: string;
}

export interface MultiTFContext {
  w1_trend: TrendState;
  d1_trend: TrendState;
  h4_trend: TrendState;
  h1_trend: TrendState;
  m15_trend: TrendState;
  session: Session | null;
  vsa_active: boolean;
  current_phase: string;
  premium_discount: {
    swing_high: number;
    swing_low: number;
    equilibrium: number;
    zone: ZoneType;
    depth_pct: number;
    is_fib_qualified: boolean;
    closest_fib: string;
    swing_high_index: number;
    swing_low_index: number;
  } | null;
}

export interface DetailedSignals {
  symbol: string;
  timestamp: number;
  context: MultiTFContext;
  checklist_v24: ChecklistItem[];
  checklist_v23: ChecklistItem[];
  signals: TradingSignal[];
  all_style_signals: TradingSignal[];
}

// ── Signal Lifecycle Types ──

export interface ResolvedSignal {
  signal_id: string;
  direction: Direction;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  take_profits?: TakeProfit[];
  risk_reward_ratio: number;
  grade: SignalGrade;
  trading_styles: string[];
  status: string;
  created_at: number;
  resolved_at: number | null;
  resolved_price: number | null;
  bars_active: number;
  confluences: string[];
  entry_method: string | null;
  confidence_score: number;
  timeframe: string;
}

export interface SignalStoreStats {
  active_count: number;
  resolved_count: number;
  sl_hits: number;
  tp_hits: number;
  expired: number;
  win_rate: number;
}

// ── Phase 5: Demo Account Types ──

export type TradeStatus = "pending" | "open" | "skipped" | "closed";
export type ActionSource = "telegram" | "web" | "timeout" | "monitor";
export type TradeSource = "signal" | "manual";

export interface DemoAccount {
  balance: number;
  initial_balance: number;
  risk_per_trade_pct: number;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  pnl_total: number;
  open_positions: number;
}

export interface DemoTrade {
  id: number;
  signal_id: number;
  status: TradeStatus;
  direction: Direction;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  risk_reward_ratio: number;
  grade: SignalGrade;
  confidence_score: number;
  confluences: string[];
  entry_method: string | null;
  pattern_type: string | null;
  position_size_usd: number | null;
  position_size_btc: number | null;
  exit_price: number | null;
  pnl_usd: number | null;
  pnl_pct: number | null;
  outcome: string | null;
  created_at: string;
  opened_at: string | null;
  closed_at: string | null;
  action_source: ActionSource;
  trade_source: TradeSource;
}

export interface DemoEquityPoint {
  balance: number;
  timestamp: string;
}

export interface TelegramStatus {
  connected: boolean;
  chat_id: string | null;
  bot_username: string | null;
}

// ── Futures Context (chart overlays from signal engine) ──

export interface OIDelta {
  timestamp: number;
  delta_pct: number;
  oi_usd: number;
}

export interface FuturesContext {
  oi_deltas: OIDelta[];
  funding_history: { timestamp: number; rate: number; mark_price: number }[];
  current_funding: number;
  taker_volume: { timestamp: number; buy_vol: number; sell_vol: number; ratio: number }[];
}

// ── Phase 6: Market Intel Types ──

export interface MarketIntelData {
  open_interest: {
    current: number;
    current_usd: number;
    history: { timestamp: number; oi: number; oi_usd: number }[];
  };
  funding_rate: {
    current: number;
    next_funding_time: number;
    mark_price: number;
    index_price: number;
    history: { timestamp: number; rate: number; mark_price: number }[];
  };
  top_trader_ratio: { timestamp: number; long_pct: number; short_pct: number; ratio: number }[];
  global_ratio: { timestamp: number; long_pct: number; short_pct: number; ratio: number }[];
  taker_volume: { timestamp: number; buy_vol: number; sell_vol: number; ratio: number }[];
  premium_index: {
    mark_price: number;
    index_price: number;
    premium_pct: number;
  };
}

export interface BacktestResult {
  symbol: string;
  start_date: string;
  end_date: string;
  total_signals: number;
  total_trades: number;
  wins: number;
  losses: number;
  timeouts: number;
  win_rate: number;
  avg_rr: number;
  profit_factor: number;
  total_pnl_pct: number;
  max_drawdown_pct: number;
  by_grade: GradeBreakdown[];
  by_confluence: ConfluenceEdge[];
  by_entry_method: Record<string, { trades: number; win_rate: number; avg_pnl: number }>;
  by_session: Record<string, { trades: number; win_rate: number }>;
  trades: TradeRecord[];
  equity_curve: { timestamp: number; pnl: number }[];
}

// ── Algo Bias Types ──

export type AlgoBiasDirection = "bullish" | "bearish" | "neutral";
export type CompositeBiasDirection = "strong_bullish" | "bullish" | "neutral" | "bearish" | "strong_bearish";
export type MarketRegime = "trending" | "mean_reverting" | "random";
export type VolRegime = "low" | "normal" | "high" | "extreme";

export interface AlgoBiasResult {
  algo_id: string;
  algo_name: string;
  direction: AlgoBiasDirection;
  score: number;        // -100 to +100
  confidence: number;   // 0 to 100
  components: Record<string, unknown>;
  explanation: string;
  data_age_seconds: number;
}

export interface AlgoWeightDetail {
  base_weight: number;      // Original weight (%)
  adjusted_weight: number;  // After Hurst + vol adjustments (%)
  weight_change: number;    // Delta from base (%)
  contribution: number;     // Score contribution to composite
  category: "momentum" | "reversion" | "regime";
}

export interface MetaAlgoDetails {
  hurst_value: number;
  hurst_effect: string;
  vol_effect: string;
  entropy_adj: number;
  vol_confidence_penalty: number;
  base_confidence: number;
  algo_weights: Record<string, AlgoWeightDetail>;
  momentum_algos: string[];
  reversion_algos: string[];
}

export interface CompositeBias {
  direction: CompositeBiasDirection;
  score: number;        // -100 to +100
  confidence: number;   // 0 to 100
  algo_count: number;
  agreement_count: number;
  regime: MarketRegime;
  vol_regime: VolRegime;
  entropy: number;
  algos: AlgoBiasResult[];
  timestamp: number;
  meta: MetaAlgoDetails;
}
