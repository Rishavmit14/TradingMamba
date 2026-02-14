"""Test the SMC engine with real BTCUSDT data from Binance.

Fetches H4 candles and runs the full detection pipeline to verify
all 10 detectors produce meaningful output on real market data.
"""
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from app.services.data_fetcher import fetch_klines
from app.core.engine import analyze_timeframe
from app.models import TrendState, SwingType, SwingClassification, Direction


async def test_single_timeframe(tf: str, limit: int):
    """Fetch real data and run analysis for one timeframe."""
    print(f"\n{'='*60}")
    print(f"  TESTING {tf} — Fetching {limit} BTCUSDT candles from Binance")
    print(f"{'='*60}")

    candles = await fetch_klines("BTCUSDT", tf, limit)
    print(f"\nFetched {len(candles)} candles")
    if candles:
        print(f"  Date range: {candles[0].timestamp} → {candles[-1].timestamp}")
        print(f"  Price range: ${candles[-1].low:,.2f} — ${candles[-1].high:,.2f}")
        print(f"  Latest close: ${candles[-1].close:,.2f}")

    result = analyze_timeframe(candles, tf)

    # --- Report ---
    print(f"\n--- TREND ---")
    print(f"  {result.trend.value}")

    print(f"\n--- 1.1 SWINGS ({len(result.swings)} detected) ---")
    if result.swings:
        highs = [s for s in result.swings if s.swing_type == SwingType.SWING_HIGH]
        lows = [s for s in result.swings if s.swing_type == SwingType.SWING_LOW]
        print(f"  Swing Highs: {len(highs)}  |  Swing Lows: {len(lows)}")
        # Show classifications
        for cls in [SwingClassification.HH, SwingClassification.HL,
                    SwingClassification.LH, SwingClassification.LL]:
            count = sum(1 for s in result.swings if s.classification == cls)
            if count:
                print(f"  {cls.value}: {count}")
        # Show last few swings
        for s in result.swings[-6:]:
            valid_tag = "SMC" if s.is_valid_smc else "X"
            print(f"    idx={s.candle_index:4d}  ${s.price:>10,.2f}  "
                  f"{s.swing_type.value:<11}  {s.classification.value:<5}  [{valid_tag}]")

    print(f"\n--- 1.2 INDUCEMENTS ({len(result.inducements)} detected) ---")
    if result.inducements:
        from app.models import IDMStatus
        active = sum(1 for i in result.inducements if i.status == IDMStatus.ACTIVE)
        taken = sum(1 for i in result.inducements if i.status == IDMStatus.TAKEN)
        transferred = sum(1 for i in result.inducements if i.status == IDMStatus.TRANSFERRED)
        print(f"  Active: {active}  |  Taken: {taken}  |  Transferred: {transferred}")

    print(f"\n--- 1.3 LIQUIDITY POOLS ({len(result.liquidity_pools)} detected) ---")
    if result.liquidity_pools:
        from app.models import LiquiditySource, LiquidityEvent
        by_source = {}
        for lp in result.liquidity_pools:
            by_source[lp.source.value] = by_source.get(lp.source.value, 0) + 1
        for src, cnt in by_source.items():
            print(f"  {src}: {cnt}")
        swept = sum(1 for lp in result.liquidity_pools if lp.swept)
        print(f"  Swept: {swept}")
        sweeps = sum(1 for lp in result.liquidity_pools
                     if lp.event_type == LiquidityEvent.SWEEP)
        grabs = sum(1 for lp in result.liquidity_pools
                    if lp.event_type == LiquidityEvent.GRAB)
        print(f"  Events — Sweeps: {sweeps}  |  Grabs: {grabs}")

    print(f"\n--- 1.4 BOS ({len(result.bos_events)} detected) ---")
    if result.bos_events:
        valid = sum(1 for b in result.bos_events if b.valid)
        invalid = len(result.bos_events) - valid
        bullish = sum(1 for b in result.bos_events if b.direction == Direction.BULLISH)
        bearish = sum(1 for b in result.bos_events if b.direction == Direction.BEARISH)
        print(f"  Valid: {valid}  |  Invalid: {invalid}")
        print(f"  Bullish: {bullish}  |  Bearish: {bearish}")
        # Show last few
        for b in result.bos_events[-4:]:
            v = "VALID" if b.valid else f"INVALID ({b.invalidation_reason})"
            print(f"    idx={b.candle_index:4d}  {b.direction.value:<8}  "
                  f"broke ${b.broken_price:>10,.2f}  [{v}]")

    print(f"\n--- 1.5 CHoCH ({len(result.choch_events)} detected) ---")
    if result.choch_events:
        for ch in result.choch_events[-4:]:
            fake = "FAKE" if ch.is_fake else ("CONFIRMED" if ch.confirmed else "UNCONFIRMED")
            print(f"    idx={ch.candle_index:4d}  {ch.direction.value:<8}  "
                  f"conf={ch.confidence:.2f}  [{fake}]"
                  f"{'  VSA!' if ch.has_vsa_confluence else ''}")
    print(f"  VSA active: {result.vsa_active} ({len(result.vsa_absorptions)} absorptions)")

    print(f"\n--- 1.6 FVGs ({len(result.fvgs)} detected) ---")
    if result.fvgs:
        valid_fvgs = sum(1 for f in result.fvgs if f.valid)
        mitigated = sum(1 for f in result.fvgs if f.mitigated)
        bullish_fvgs = sum(1 for f in result.fvgs if f.direction == Direction.BULLISH)
        bearish_fvgs = sum(1 for f in result.fvgs if f.direction == Direction.BEARISH)
        extreme = sum(1 for f in result.fvgs if f.from_extreme_candle)
        print(f"  Valid: {valid_fvgs}  |  Mitigated: {mitigated}")
        print(f"  Bullish: {bullish_fvgs}  |  Bearish: {bearish_fvgs}")
        print(f"  From extreme candle: {extreme}")

    print(f"\n--- 1.7 ORDER BLOCKS ({len(result.order_blocks)} detected) ---")
    if result.order_blocks:
        valid_obs = sum(1 for o in result.order_blocks if o.valid)
        mitigated_obs = sum(1 for o in result.order_blocks if o.mitigated)
        with_fvg = sum(1 for o in result.order_blocks if o.has_fvg)
        swept_liq = sum(1 for o in result.order_blocks if o.swept_liquidity)
        traps = sum(1 for o in result.order_blocks if o.is_trap)
        print(f"  Valid: {valid_obs}  |  Mitigated: {mitigated_obs}")
        print(f"  Has FVG (Rule 2): {with_fvg}  |  Swept liq (Rule 1): {swept_liq}")
        print(f"  Traps: {traps}")

    print(f"\n--- 1.8 PREMIUM/DISCOUNT ---")
    if result.premium_discount:
        pd = result.premium_discount
        print(f"  Zone: {pd.zone.value}")
        print(f"  Range: ${pd.swing_low:,.2f} — ${pd.swing_high:,.2f}")
        print(f"  Equilibrium (50%): ${pd.equilibrium:,.2f}")
        print(f"  Depth: {pd.depth_pct:.1f}%")

    print(f"\n--- 1.9 SESSION ---")
    if result.session:
        s = result.session
        print(f"  Session: {s.name}")
        print(f"  Kill zone: {s.is_kill_zone}")
        print(f"  Volatility: {s.volatility_expectation}")

    print(f"\n--- 1.10 SIGNALS ({len(result.signals)} generated) ---")
    if result.signals:
        for sig in result.signals:
            print(f"    {sig.direction.value} {sig.grade.value}-grade  "
                  f"R:R={sig.risk_reward_ratio}  conf={sig.confidence_score}%")
            print(f"      Entry: ${sig.entry_price:,.2f}  SL: ${sig.stop_loss:,.2f}  "
                  f"TP: ${sig.take_profit:,.2f}")
            print(f"      Confluences: {', '.join(sig.confluences)}")

    return result


async def main():
    print("TradingMamba SMC Engine — Real Data Test")
    print("=" * 60)

    # Test H4 with 500 candles (main analysis timeframe)
    h4_result = await test_single_timeframe("H4", 500)

    # Test D1 with 200 candles
    d1_result = await test_single_timeframe("D1", 200)

    # Summary
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    detectors = {
        "1.1 Swings": len(h4_result.swings),
        "1.2 IDMs": len(h4_result.inducements),
        "1.3 Liquidity": len(h4_result.liquidity_pools),
        "1.4 BOS": len(h4_result.bos_events),
        "1.5 CHoCH": len(h4_result.choch_events),
        "1.6 FVGs": len(h4_result.fvgs),
        "1.7 OBs": len(h4_result.order_blocks),
        "1.8 P/D": 1 if h4_result.premium_discount else 0,
        "1.9 Session": 1 if h4_result.session else 0,
        "1.10 Signals": len(h4_result.signals),
    }

    all_producing = True
    for name, count in detectors.items():
        status = "PASS" if count > 0 else "ZERO"
        if count == 0 and name not in ("1.10 Signals", "1.5 CHoCH"):
            all_producing = False
        print(f"  {name:<15} {count:>5}  [{status}]")

    print(f"\n  Overall: {'ALL DETECTORS PRODUCING OUTPUT' if all_producing else 'SOME DETECTORS NEED ATTENTION'}")


if __name__ == "__main__":
    asyncio.run(main())
