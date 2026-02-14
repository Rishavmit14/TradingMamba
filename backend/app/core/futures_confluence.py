"""Futures Confluence Evaluator — scores Binance Futures data against signal direction.

Pure function: takes raw futures_data dict (from fetch_market_intel) + signal direction,
returns (confluences: list[str], confidence_modifier: int).

Designed to be called AFTER _grade_signal() so futures data affects confidence
but NOT the A/B/C/D structural grade.
"""

from __future__ import annotations

from app.models import Direction


def evaluate_futures_confluences(
    futures_data: dict | None,
    signal_direction: Direction,
) -> tuple[list[str], int]:
    """Evaluate Binance Futures data for confluence with a signal direction.

    Returns:
        (confluences, confidence_modifier)
        - confluences: list of human-readable strings (e.g. "Futures: OI rising +3.2%")
        - confidence_modifier: integer to add to confidence score (+5 per supporting, -5 per contradicting)
    """
    if not futures_data:
        return [], 0

    is_bull = (
        signal_direction == Direction.BULLISH
        if isinstance(signal_direction, Direction)
        else signal_direction == "bullish"
    )

    confluences: list[str] = []
    modifier = 0

    # ── 1. OI Delta ──
    # Compare recent 6h OI vs previous 6h
    oi_hist = futures_data.get("open_interest", {}).get("history", [])
    if len(oi_hist) >= 12:
        recent_oi = oi_hist[-1]["oi_usd"]
        earlier_oi = oi_hist[-12]["oi_usd"]
        if earlier_oi > 0:
            oi_change_pct = (recent_oi - earlier_oi) / earlier_oi * 100
            if abs(oi_change_pct) >= 2.0:
                if oi_change_pct > 0:
                    confluences.append(f"Futures: OI rising +{oi_change_pct:.1f}%")
                    modifier += 5  # Rising OI = conviction, supports any direction
                else:
                    confluences.append(f"Futures: OI falling {oi_change_pct:.1f}%")
                    modifier -= 5  # Falling OI = unwinding, weaker conviction

    # ── 2. Funding Rate ──
    # Positive funding = longs pay shorts = crowded long (bearish pressure)
    # Negative funding = shorts pay longs = crowded short (bullish pressure)
    current_funding = futures_data.get("funding_rate", {}).get("current", 0)
    if abs(current_funding) > 0.0001:  # > 0.01% is meaningful
        rate_pct = current_funding * 100
        if is_bull:
            if current_funding <= 0:
                # Bullish signal + negative funding = longs not crowded = supportive
                confluences.append(f"Futures: Funding {rate_pct:+.3f}% favors longs")
                modifier += 5
            elif current_funding > 0.0005:
                # Bullish signal + highly positive funding = crowded longs = contrarian warning
                confluences.append(f"Futures: Funding {rate_pct:+.3f}% crowded longs")
                modifier -= 5
        else:
            if current_funding > 0:
                # Bearish signal + positive funding = longs crowded = supports shorts
                confluences.append(f"Futures: Funding {rate_pct:+.3f}% favors shorts")
                modifier += 5
            elif current_funding < -0.0005:
                # Bearish signal + highly negative funding = crowded shorts = contrarian warning
                confluences.append(f"Futures: Funding {rate_pct:+.3f}% crowded shorts")
                modifier -= 5

    # ── 3. Taker Buy/Sell Volume ──
    # Aggregate last 4 hourly entries
    taker = futures_data.get("taker_volume", [])
    if len(taker) >= 4:
        recent_taker = taker[-4:]
        total_buy = sum(t["buy_vol"] for t in recent_taker)
        total_sell = sum(t["sell_vol"] for t in recent_taker)
        if total_sell > 0:
            taker_ratio = total_buy / total_sell
            if is_bull and taker_ratio > 1.1:
                confluences.append(f"Futures: Taker buy dominant ({taker_ratio:.2f}x)")
                modifier += 5
            elif not is_bull and taker_ratio < 0.9:
                confluences.append(f"Futures: Taker sell dominant ({1/taker_ratio:.2f}x)")
                modifier += 5
            elif is_bull and taker_ratio < 0.85:
                confluences.append(f"Futures: Taker sell dominant ({1/taker_ratio:.2f}x) \u26a0")
                modifier -= 5
            elif not is_bull and taker_ratio > 1.15:
                confluences.append(f"Futures: Taker buy dominant ({taker_ratio:.2f}x) \u26a0")
                modifier -= 5

    # ── 4. Top Trader Long/Short Ratio ──
    top_ratio = futures_data.get("top_trader_ratio", [])
    if top_ratio:
        latest = top_ratio[-1]
        long_pct = latest["long_pct"]
        short_pct = latest["short_pct"]
        if is_bull and long_pct > 0.55:
            confluences.append(f"Futures: Top traders {long_pct*100:.0f}% long")
            modifier += 5
        elif not is_bull and short_pct > 0.55:
            confluences.append(f"Futures: Top traders {short_pct*100:.0f}% short")
            modifier += 5
        elif is_bull and short_pct > 0.55:
            confluences.append(f"Futures: Top traders {short_pct*100:.0f}% short \u26a0")
            modifier -= 5
        elif not is_bull and long_pct > 0.55:
            confluences.append(f"Futures: Top traders {long_pct*100:.0f}% long \u26a0")
            modifier -= 5

    return confluences, modifier
