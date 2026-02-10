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
export type SignalGrade = "A" | "B" | "C" | "D";

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
  idm_taken: boolean;
}

export interface Inducement {
  candle_index: number;
  price: number;
  parent_swing_index: number;
  status: IDMStatus;
  taken_at_candle: number | null;
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
}

export interface CHoCH {
  candle_index: number;
  direction: Direction;
  broken_swing_index: number;
  broken_price: number;
  confidence: number;
  has_climax_confluence: boolean;
  is_fake: boolean;
  confirmed: boolean;
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
}

export interface Session {
  name: string;
  is_kill_zone: boolean;
  volatility_expectation: string;
}

export interface TradingSignal {
  direction: Direction;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  risk_reward_ratio: number;
  confidence_score: number;
  grade: SignalGrade;
  confluences: string[];
  timeframe: string;
  entry_method: string | null;
  pattern_type: string;
  timestamp: number;
  climax_warning: boolean;
  is_counter_trend: boolean;
}

export interface AnalysisResult {
  timeframe: string;
  trend: TrendState;
  swings: SwingPoint[];
  inducements: Inducement[];
  liquidity_pools: LiquidityPool[];
  bos_events: BOS[];
  choch_events: CHoCH[];
  fvgs: FVG[];
  order_blocks: OrderBlock[];
  premium_discount: PremiumDiscount | null;
  session: Session | null;
  climax_warning: boolean;
  climax_ratio: number;
  signals: TradingSignal[];
  candles: Candle[];
}
