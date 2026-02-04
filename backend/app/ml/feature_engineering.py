"""
Feature Engineering - Extract ML Features from OHLCV Data (Tier 3)

Extracts 42 features from price data for ML model training.
Features cover price action, volatility, momentum, volume,
structure, time, and context dimensions.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature names for reference
FEATURE_NAMES = [
    # Price Action (10)
    'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'range_pct',
    'gap_from_prev', 'close_position_in_range', 'is_doji', 'is_hammer',
    'is_engulfing', 'body_direction',
    # Volatility (6)
    'atr_14', 'atr_ratio', 'bollinger_width', 'range_percentile_20',
    'true_range', 'volatility_regime',
    # Momentum (8)
    'rsi_14', 'rsi_7', 'macd_histogram', 'stochastic_k',
    'stochastic_d', 'roc_5', 'roc_10', 'momentum_12',
    # Volume (4)
    'volume_ratio_20', 'volume_trend', 'obv_slope', 'relative_volume',
    # Structure (6)
    'dist_to_swing_high', 'dist_to_swing_low', 'higher_high_count',
    'lower_low_count', 'structure_trend', 'swing_range',
    # Time (4)
    'hour_of_day', 'day_of_week', 'is_kill_zone', 'session',
    # Context (4)
    'candles_since_last_swing', 'trend_strength_20', 'ma_20_distance',
    'ma_50_distance',
]

NUM_FEATURES = len(FEATURE_NAMES)  # 42

# Video-derived feature names (from video training knowledge)
VIDEO_FEATURE_NAMES = [
    # Pattern Context (6)
    'ob_proximity', 'fvg_proximity', 'liquidity_proximity',
    'premium_discount_zone', 'active_pattern_count', 'pattern_confluence_score',
    # Video Teaching Depth (5)
    'max_pattern_teaching_depth', 'avg_pattern_teaching_depth',
    'video_confidence_score', 'teaching_word_density', 'visual_frame_density',
    # Co-occurrence Context (4)
    'co_occurrence_score', 'concept_pair_strength',
    'unexpected_pattern_flag', 'synergy_score',
    # ICT Rule Alignment (4)
    'displacement_present', 'structure_break_recent',
    'kill_zone_alignment', 'fibonacci_alignment',
    # Learned Directional Bias (3)
    'video_bullish_bias', 'video_bearish_bias', 'video_bias_confidence',
]

EXTENDED_FEATURE_NAMES = FEATURE_NAMES + VIDEO_FEATURE_NAMES
NUM_EXTENDED_FEATURES = len(EXTENDED_FEATURE_NAMES)  # 64
NUM_VIDEO_FEATURES = len(VIDEO_FEATURE_NAMES)  # 22


class FeatureEngineer:
    """Extract ML features from OHLCV data."""

    def __init__(self):
        self._feature_cache = {}

    def extract_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """
        Extract feature vector for a single bar.

        Args:
            data: OHLCV DataFrame
            index: Bar index to extract features for (needs lookback context)

        Returns:
            np.ndarray of shape (42,)
        """
        if index < 50 or index >= len(data):
            return np.zeros(NUM_FEATURES)

        features = np.zeros(NUM_FEATURES)
        fi = 0  # feature index

        o = float(data['open'].iloc[index])
        h = float(data['high'].iloc[index])
        l = float(data['low'].iloc[index])
        c = float(data['close'].iloc[index])
        v = float(data['volume'].iloc[index]) if 'volume' in data.columns else 0.0

        total_range = h - l
        body = abs(c - o)

        # === PRICE ACTION (10) ===
        # body_ratio
        features[fi] = body / total_range if total_range > 0 else 0
        fi += 1
        # upper_wick_ratio
        upper_wick = h - max(o, c)
        features[fi] = upper_wick / total_range if total_range > 0 else 0
        fi += 1
        # lower_wick_ratio
        lower_wick = min(o, c) - l
        features[fi] = lower_wick / total_range if total_range > 0 else 0
        fi += 1
        # range_pct
        features[fi] = total_range / c * 100 if c > 0 else 0
        fi += 1
        # gap_from_prev
        prev_c = float(data['close'].iloc[index - 1])
        features[fi] = (o - prev_c) / prev_c * 100 if prev_c > 0 else 0
        fi += 1
        # close_position_in_range (0 = at low, 1 = at high)
        features[fi] = (c - l) / total_range if total_range > 0 else 0.5
        fi += 1
        # is_doji (body < 10% of range)
        features[fi] = 1.0 if (total_range > 0 and body / total_range < 0.1) else 0.0
        fi += 1
        # is_hammer (lower wick > 2x body, small upper wick)
        features[fi] = 1.0 if (lower_wick > body * 2 and upper_wick < body * 0.5 and body > 0) else 0.0
        fi += 1
        # is_engulfing
        prev_o = float(data['open'].iloc[index - 1])
        prev_body = abs(prev_c - prev_o)
        features[fi] = 1.0 if (body > prev_body * 1.1 and total_range > 0) else 0.0
        fi += 1
        # body_direction (1 = bullish, -1 = bearish, 0 = doji)
        features[fi] = 1.0 if c > o else (-1.0 if c < o else 0.0)
        fi += 1

        # === VOLATILITY (6) ===
        highs = data['high'].iloc[max(0, index-14):index+1].values.astype(float)
        lows = data['low'].iloc[max(0, index-14):index+1].values.astype(float)
        closes = data['close'].iloc[max(0, index-14):index+1].values.astype(float)

        # ATR 14
        if len(closes) >= 2:
            tr_values = []
            for i in range(1, len(closes)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                tr_values.append(tr)
            atr_14 = np.mean(tr_values) if tr_values else total_range
        else:
            atr_14 = total_range
        features[fi] = atr_14 / c * 100 if c > 0 else 0  # Normalized
        fi += 1

        # atr_ratio (current range vs ATR)
        features[fi] = total_range / atr_14 if atr_14 > 0 else 1.0
        fi += 1

        # bollinger_width
        closes_20 = data['close'].iloc[max(0, index-20):index+1].values.astype(float)
        if len(closes_20) >= 5:
            ma_20 = np.mean(closes_20)
            std_20 = np.std(closes_20)
            features[fi] = (std_20 * 4) / ma_20 * 100 if ma_20 > 0 else 0
        fi += 1

        # range_percentile_20 (where current range sits vs last 20 bars)
        ranges_20 = (data['high'].iloc[max(0, index-20):index+1] - data['low'].iloc[max(0, index-20):index+1]).values.astype(float)
        if len(ranges_20) > 1:
            features[fi] = np.searchsorted(np.sort(ranges_20), total_range) / len(ranges_20)
        fi += 1

        # true_range
        features[fi] = total_range / c * 100 if c > 0 else 0
        fi += 1

        # volatility_regime (ATR expanding = 1, contracting = -1)
        if len(closes) >= 7:
            atr_recent = np.mean([highs[i] - lows[i] for i in range(-3, 0)] if len(highs) >= 3 else [total_range])
            atr_older = np.mean([highs[i] - lows[i] for i in range(-7, -3)] if len(highs) >= 7 else [total_range])
            features[fi] = 1.0 if atr_recent > atr_older else -1.0
        fi += 1

        # === MOMENTUM (8) ===
        closes_all = data['close'].iloc[max(0, index-50):index+1].values.astype(float)

        # RSI 14
        features[fi] = self._compute_rsi(closes_all, 14)
        fi += 1
        # RSI 7
        features[fi] = self._compute_rsi(closes_all, 7)
        fi += 1

        # MACD histogram (12, 26, 9)
        if len(closes_all) >= 26:
            ema_12 = self._ema(closes_all, 12)
            ema_26 = self._ema(closes_all, 26)
            macd_line = ema_12 - ema_26
            features[fi] = macd_line / c * 100 if c > 0 else 0
        fi += 1

        # Stochastic K
        if len(highs) >= 14:
            highest = np.max(data['high'].iloc[max(0, index-14):index+1].values.astype(float))
            lowest = np.min(data['low'].iloc[max(0, index-14):index+1].values.astype(float))
            features[fi] = (c - lowest) / (highest - lowest) * 100 if highest != lowest else 50
        fi += 1

        # Stochastic D (3-period SMA of K) - simplified
        features[fi] = features[fi - 1]  # Approximation
        fi += 1

        # ROC 5
        if len(closes_all) > 5:
            features[fi] = (c - closes_all[-6]) / closes_all[-6] * 100 if closes_all[-6] > 0 else 0
        fi += 1
        # ROC 10
        if len(closes_all) > 10:
            features[fi] = (c - closes_all[-11]) / closes_all[-11] * 100 if closes_all[-11] > 0 else 0
        fi += 1

        # Momentum 12
        if len(closes_all) > 12:
            features[fi] = (c - closes_all[-13]) / closes_all[-13] * 100 if closes_all[-13] > 0 else 0
        fi += 1

        # === VOLUME (4) ===
        if 'volume' in data.columns and v > 0:
            vol_20 = data['volume'].iloc[max(0, index-20):index].values.astype(float)
            avg_vol = np.mean(vol_20) if len(vol_20) > 0 else v

            # volume_ratio_20
            features[fi] = v / avg_vol if avg_vol > 0 else 1.0
            fi += 1
            # volume_trend (rising = 1, falling = -1)
            if len(vol_20) >= 5:
                recent_vol = np.mean(vol_20[-3:]) if len(vol_20) >= 3 else v
                older_vol = np.mean(vol_20[:3]) if len(vol_20) >= 3 else v
                features[fi] = 1.0 if recent_vol > older_vol else -1.0
            fi += 1
            # obv_slope (simplified - direction of volume vs price)
            if c > prev_c:
                features[fi] = 1.0
            elif c < prev_c:
                features[fi] = -1.0
            fi += 1
            # relative_volume
            features[fi] = min(v / avg_vol, 5.0) if avg_vol > 0 else 1.0
            fi += 1
        else:
            fi += 4  # Skip volume features if not available

        # === STRUCTURE (6) ===
        # Find recent swing highs/lows for structure features
        lookback = min(50, index)
        recent_highs = data['high'].iloc[max(0, index-lookback):index+1].values.astype(float)
        recent_lows = data['low'].iloc[max(0, index-lookback):index+1].values.astype(float)

        swing_high = np.max(recent_highs) if len(recent_highs) > 0 else h
        swing_low = np.min(recent_lows) if len(recent_lows) > 0 else l

        # dist_to_swing_high
        features[fi] = (swing_high - c) / c * 100 if c > 0 else 0
        fi += 1
        # dist_to_swing_low
        features[fi] = (c - swing_low) / c * 100 if c > 0 else 0
        fi += 1

        # higher_high_count (last 20 bars)
        hh_count = 0
        for i in range(max(1, len(recent_highs) - 20), len(recent_highs)):
            if recent_highs[i] > recent_highs[i-1]:
                hh_count += 1
        features[fi] = hh_count
        fi += 1

        # lower_low_count (last 20 bars)
        ll_count = 0
        for i in range(max(1, len(recent_lows) - 20), len(recent_lows)):
            if recent_lows[i] < recent_lows[i-1]:
                ll_count += 1
        features[fi] = ll_count
        fi += 1

        # structure_trend (1 = bullish HH+HL, -1 = bearish LH+LL)
        features[fi] = 1.0 if hh_count > ll_count else (-1.0 if ll_count > hh_count else 0.0)
        fi += 1

        # swing_range (as % of price)
        features[fi] = (swing_high - swing_low) / c * 100 if c > 0 else 0
        fi += 1

        # === TIME (4) ===
        idx = data.index[index]
        if hasattr(idx, 'hour'):
            features[fi] = idx.hour / 24.0  # Normalized
            fi += 1
            features[fi] = idx.weekday() / 6.0  # Normalized
            fi += 1
            # is_kill_zone
            hour = idx.hour
            features[fi] = 1.0 if hour in range(1, 5) or hour in range(7, 10) or hour in range(12, 15) else 0.0
            fi += 1
            # session (0=asian, 0.33=london, 0.66=ny, 1.0=off)
            if 1 <= hour < 5:
                features[fi] = 0.0  # Asian
            elif 7 <= hour < 12:
                features[fi] = 0.33  # London
            elif 12 <= hour < 17:
                features[fi] = 0.66  # NY
            else:
                features[fi] = 1.0  # Off-hours
            fi += 1
        else:
            fi += 4  # Skip time features for daily+ data

        # === CONTEXT (4) ===
        # candles_since_last_swing (approximate)
        peak_idx = np.argmax(recent_highs) if len(recent_highs) > 0 else 0
        trough_idx = np.argmin(recent_lows) if len(recent_lows) > 0 else 0
        last_swing = max(peak_idx, trough_idx)
        features[fi] = (len(recent_highs) - 1 - last_swing) / max(lookback, 1)
        fi += 1

        # trend_strength_20 (slope of closes over 20 bars)
        if len(closes_20) >= 5:
            x = np.arange(len(closes_20))
            slope = np.polyfit(x, closes_20, 1)[0]
            features[fi] = slope / c * 1000 if c > 0 else 0  # Normalized
        fi += 1

        # ma_20_distance
        if len(closes_20) >= 5:
            ma20 = np.mean(closes_20)
            features[fi] = (c - ma20) / ma20 * 100 if ma20 > 0 else 0
        fi += 1

        # ma_50_distance
        closes_50 = data['close'].iloc[max(0, index-50):index+1].values.astype(float)
        if len(closes_50) >= 10:
            ma50 = np.mean(closes_50)
            features[fi] = (c - ma50) / ma50 * 100 if ma50 > 0 else 0
        fi += 1

        return features

    def extract_features_batch(self, data: pd.DataFrame, start: int = 50) -> np.ndarray:
        """
        Extract features for all bars from start to end.

        Returns:
            np.ndarray of shape (n_bars, 42)
        """
        n = len(data) - start
        if n <= 0:
            return np.zeros((0, NUM_FEATURES))

        X = np.zeros((n, NUM_FEATURES))
        for i in range(n):
            X[i] = self.extract_features(data, start + i)
        return X

    def build_training_dataset(
        self,
        symbol: str,
        timeframe: str = 'D1',
        lookback_days: int = 365,
        forward_bars: int = 20,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Build labeled dataset for ML training.

        Uses backtester outcomes as labels:
        - For each bar, extract features
        - Label = 1 if pattern at that bar was profitable, 0 otherwise

        Returns:
            (X, y_dict) where y_dict maps pattern_type -> binary labels
        """
        from .data_cache import get_data_cache
        from .backtester import get_backtester

        cache = get_data_cache()
        data = cache.get_ohlcv(symbol, timeframe, lookback_days)

        if len(data) < 100:
            return np.zeros((0, NUM_FEATURES)), {}

        # Run backtest to get outcomes
        backtester = get_backtester()
        bt_result = backtester.backtest_patterns(symbol, timeframe, lookback_days)

        # Extract features for all bars
        start_idx = 50
        X = self.extract_features_batch(data, start_idx)

        # Build label arrays per pattern type
        y_dict: Dict[str, np.ndarray] = {}
        n_bars = X.shape[0]

        # Map outcomes to bar indices
        for outcome in bt_result.outcomes:
            ptype = outcome.pattern_type
            if ptype not in y_dict:
                y_dict[ptype] = np.full(n_bars, -1.0)  # -1 = no signal

            # Map detection index to feature array index
            feat_idx = outcome.detection_index - start_idx
            if 0 <= feat_idx < n_bars:
                y_dict[ptype][feat_idx] = 1.0 if outcome.return_20bar > 0 else 0.0

        return X, y_dict

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
        """Compute RSI."""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Compute EMA (returns last value)."""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0

        multiplier = 2.0 / (period + 1)
        ema = data[0]
        for val in data[1:]:
            ema = (val - ema) * multiplier + ema
        return ema

    def get_feature_names(self) -> List[str]:
        """Get ordered feature names."""
        return list(FEATURE_NAMES)

    def get_extended_feature_names(self) -> List[str]:
        """Get ordered extended feature names (42 OHLCV + 22 video)."""
        return list(EXTENDED_FEATURE_NAMES)

    def extract_extended_features(self, data: pd.DataFrame, index: int,
                                   analysis: Dict = None) -> np.ndarray:
        """
        Extract 64-dim feature vector: 42 OHLCV + 22 video-derived.

        Args:
            data: OHLCV DataFrame
            index: Bar index
            analysis: Smart Money analysis result dict (from SmartMoneyAnalyzer.analyze())
                      Keys used: order_blocks, fvgs, liquidity_levels, structure,
                      premium_discount, bias, displacement patterns

        Returns:
            np.ndarray of shape (64,)
        """
        # Get base 42 OHLCV features
        base_features = self.extract_features(data, index)

        # Get 22 video features
        video_features = self._extract_video_features(data, index, analysis)

        return np.concatenate([base_features, video_features])

    def extract_extended_features_batch(self, data: pd.DataFrame, start: int = 50,
                                         analysis: Dict = None) -> np.ndarray:
        """
        Extract extended features for all bars from start to end.

        Args:
            data: OHLCV DataFrame
            start: Starting bar index
            analysis: Smart Money analysis result (applied to all bars for context)

        Returns:
            np.ndarray of shape (n_bars, 64)
        """
        n = len(data) - start
        if n <= 0:
            return np.zeros((0, NUM_EXTENDED_FEATURES))

        X = np.zeros((n, NUM_EXTENDED_FEATURES))
        for i in range(n):
            X[i] = self.extract_extended_features(data, start + i, analysis)
        return X

    def _extract_video_features(self, data: pd.DataFrame, index: int,
                                 analysis: Dict = None) -> np.ndarray:
        """
        Extract 22 video-derived features from analysis + video knowledge.

        If no video knowledge is loaded or no analysis provided,
        returns zeros (graceful degradation).
        """
        features = np.zeros(NUM_VIDEO_FEATURES)

        # Try to get video knowledge index (playlist-aware via context variable)
        try:
            from .playlist_registry import PlaylistRegistry, playlist_context
            vk = PlaylistRegistry.get_video_knowledge(playlist_context.get())
            if not vk.is_loaded():
                return features
        except Exception:
            return features

        if analysis is None:
            return features

        c = float(data['close'].iloc[index]) if index < len(data) else 0.0
        if c == 0:
            return features

        # Helper: analysis can be dict or SmartMoneyAnalysisResult dataclass
        def _aget(key, default=None):
            if isinstance(analysis, dict):
                return analysis.get(key, default)
            return getattr(analysis, key, default)

        # Helper: get value from pattern (dict or dataclass)
        def _pget(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        fi = 0  # feature index

        # === Pattern Context (6) ===
        order_blocks = _aget('order_blocks', []) or []
        fvgs = _aget('fair_value_gaps', []) or _aget('fvgs', []) or []
        # liquidity_levels can be a dict of lists; flatten to a list
        raw_liq = _aget('liquidity_levels', []) or []
        if isinstance(raw_liq, dict):
            liquidity_levels = []
            for v in raw_liq.values():
                if isinstance(v, list):
                    liquidity_levels.extend(v)
        else:
            liquidity_levels = raw_liq

        # ob_proximity: normalized distance to nearest order block
        if order_blocks:
            min_dist = min(abs(c - (_pget(ob, 'high', None) or c)) / c for ob in order_blocks) if order_blocks else 1.0
            features[fi] = min(min_dist * 100, 10.0) / 10.0  # 0 = at OB, 1 = far
        else:
            features[fi] = 1.0  # No OBs detected
        fi += 1

        # fvg_proximity: normalized distance to nearest FVG
        if fvgs:
            min_dist = min(abs(c - (_pget(fvg, 'high', None) or c)) / c for fvg in fvgs) if fvgs else 1.0
            features[fi] = min(min_dist * 100, 10.0) / 10.0
        else:
            features[fi] = 1.0
        fi += 1

        # liquidity_proximity
        if liquidity_levels:
            min_dist = min(abs(c - (_pget(ll, 'price', None) or _pget(ll, 'level', None) or c)) / c for ll in liquidity_levels) if liquidity_levels else 1.0
            features[fi] = min(min_dist * 100, 10.0) / 10.0
        else:
            features[fi] = 1.0
        fi += 1

        # premium_discount_zone: -1 = deep discount, 0 = equilibrium, +1 = deep premium
        pd_zone = _aget('premium_discount', {})
        if isinstance(pd_zone, dict) and 'zone' in pd_zone:
            zone = pd_zone['zone']
            if isinstance(zone, str):
                zone_lower = zone.lower()
                if 'premium' in zone_lower:
                    features[fi] = 1.0
                elif 'discount' in zone_lower:
                    features[fi] = -1.0
                else:
                    features[fi] = 0.0
            elif isinstance(zone, (int, float)):
                features[fi] = max(-1.0, min(1.0, float(zone)))
        fi += 1

        # active_pattern_count
        active_patterns = []
        detected_pattern_types = []
        if order_blocks:
            active_patterns.extend(order_blocks)
            detected_pattern_types.append('order_block')
        if fvgs:
            active_patterns.extend(fvgs)
            detected_pattern_types.append('fvg')
        if liquidity_levels:
            active_patterns.extend(liquidity_levels)
            detected_pattern_types.append('liquidity')

        # Check for other pattern types in analysis
        if _aget('displacements') or _aget('displacement_patterns'):
            detected_pattern_types.append('displacement')
        if _aget('ote_zones'):
            detected_pattern_types.append('optimal_trade_entry')
        if _aget('breaker_blocks'):
            detected_pattern_types.append('breaker')
        structure = _aget('structure_events', []) or _aget('structure', []) or []
        if structure:
            detected_pattern_types.append('market_structure')

        features[fi] = min(len(active_patterns), 20) / 20.0  # Normalized 0-1
        fi += 1

        # pattern_confluence_score: weighted by teaching depth
        confluence = 0.0
        for pt in detected_pattern_types:
            confluence += vk.get_teaching_depth_score(pt)
        features[fi] = min(confluence / max(len(detected_pattern_types), 1), 1.0) if detected_pattern_types else 0.0
        fi += 1

        # === Video Teaching Depth (5) ===
        depth_scores = [vk.get_teaching_depth_score(pt) for pt in detected_pattern_types]

        # max_pattern_teaching_depth
        features[fi] = max(depth_scores) if depth_scores else 0.0
        fi += 1

        # avg_pattern_teaching_depth
        features[fi] = np.mean(depth_scores) if depth_scores else 0.0
        fi += 1

        # video_confidence_score (max confidence among detected patterns)
        confidence_scores = []
        for pt in detected_pattern_types:
            profile = vk.get_concept_profile(pt)
            if profile:
                # Confidence based on teaching depth + frame count
                conf = min(profile.teaching_depth / 20.0, 0.5) + min(profile.total_frames / 50.0, 0.3) + min(profile.video_count / 10.0, 0.2)
                confidence_scores.append(min(conf, 1.0))
        features[fi] = max(confidence_scores) if confidence_scores else 0.0
        fi += 1

        # teaching_word_density
        word_scores = [vk.get_word_density_score(pt) for pt in detected_pattern_types]
        features[fi] = np.mean(word_scores) if word_scores else 0.0
        fi += 1

        # visual_frame_density
        frame_scores = [vk.get_frame_density_score(pt) for pt in detected_pattern_types]
        features[fi] = np.mean(frame_scores) if frame_scores else 0.0
        fi += 1

        # === Co-occurrence Context (4) ===
        # co_occurrence_score
        features[fi] = vk.get_context_similarity(detected_pattern_types)
        fi += 1

        # concept_pair_strength
        features[fi] = vk.get_strongest_pair(detected_pattern_types)
        fi += 1

        # unexpected_pattern_flag
        features[fi] = 1.0 if vk.has_unexpected_combination(detected_pattern_types) else 0.0
        fi += 1

        # synergy_score
        features[fi] = vk.get_synergy_score(detected_pattern_types)
        fi += 1

        # === ICT Rule Alignment (4) ===
        # displacement_present (large body candle nearby)
        has_displacement = False
        if _aget('displacements') or _aget('displacement_patterns'):
            has_displacement = True
        else:
            # Check if recent bar has large body (>2x ATR approximation)
            lookback = min(5, index)
            for offset in range(lookback):
                bar_idx = index - offset
                if bar_idx >= 0 and bar_idx < len(data):
                    bar_body = abs(float(data['close'].iloc[bar_idx]) - float(data['open'].iloc[bar_idx]))
                    bar_range = float(data['high'].iloc[bar_idx]) - float(data['low'].iloc[bar_idx])
                    if bar_range > 0 and bar_body / bar_range > 0.7:
                        has_displacement = True
                        break
        features[fi] = 1.0 if has_displacement else 0.0
        fi += 1

        # structure_break_recent
        has_bos = False
        if structure:
            for s in structure[-5:]:  # Last 5 structure events
                s_type = _pget(s, 'type', '')
                s_type = str(s_type) if s_type else ''
                if 'bos' in s_type.lower() or 'choch' in s_type.lower():
                    has_bos = True
                    break
        features[fi] = 1.0 if has_bos else 0.0
        fi += 1

        # kill_zone_alignment
        idx = data.index[index] if index < len(data) else None
        in_kill_zone = False
        if idx is not None and hasattr(idx, 'hour'):
            hour = idx.hour
            in_kill_zone = hour in range(1, 5) or hour in range(7, 10) or hour in range(12, 15)
        # Boost if kill_zone was learned from videos
        kz_depth = vk.get_teaching_depth_score('kill_zone')
        features[fi] = (1.0 if in_kill_zone else 0.0) * max(kz_depth, 0.5)
        fi += 1

        # fibonacci_alignment (price near 62-79% retracement)
        has_fib = False
        if _aget('ote_zones'):
            has_fib = True
        fib_depth = vk.get_teaching_depth_score('fibonacci')
        features[fi] = (1.0 if has_fib else 0.0) * max(fib_depth, 0.5)
        fi += 1

        # === Learned Directional Bias (3) ===
        bullish, bearish, bias_conf = vk.get_directional_bias(detected_pattern_types, analysis)
        features[fi] = bullish
        fi += 1
        features[fi] = bearish
        fi += 1
        features[fi] = bias_conf
        fi += 1

        return features


# Singleton
_engineer_instance = None

def get_feature_engineer() -> FeatureEngineer:
    global _engineer_instance
    if _engineer_instance is None:
        _engineer_instance = FeatureEngineer()
    return _engineer_instance
