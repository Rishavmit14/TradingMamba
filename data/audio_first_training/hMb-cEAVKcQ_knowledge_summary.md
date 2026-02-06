# Knowledge Summary: 13-Fair Value Gap

**Video ID:** hMb-cEAVKcQ
**Title:** 13-Fair Value Gap (Forex Minions)
**Generated:** 2026-02-05
**Method:** Claude Code expert analysis
**Duration:** ~1377 seconds (~23 minutes)
**Total Words:** 3480

---

## Processing Statistics

| Metric | Value |
|---|---|
| Teaching Units | 46 |
| Vision Analyses | 72 |
| Concepts Extracted | 5 |
| Total Audio Duration | 1229.02s |
| Total Words | 3480 |

---

## Concepts Overview

This video is part of the Forex Minions ICT entry techniques series. It covers the foundational concepts needed to identify and validate high-probability buy and sell zones using liquidity events and Fair Value Gaps. The lecture progresses from prerequisite liquidity concepts to the core FVG definition, and culminates with detailed validation rules for determining which FVGs are worth trading.

---

## 1. Liquidity Sweep

**Teaching Duration:** 148s | **Word Count:** 460 | **Teaching Units:** 10

A liquidity sweep occurs when price briefly exceeds a previous swing high or low using only a candle wick, triggering resting stop-loss and pending orders, before reversing direction. The market maker uses this mechanism to collect liquidity needed for the next directional move.

**Two types:**
- **Buy Side Liquidity Sweep** (bearish context): Wick sweeps above a prior high, trapping breakout buyers, then reverses downward.
- **Sell Side Liquidity Sweep** (bullish context): Wick sweeps below a prior low, trapping breakout sellers, then reverses upward.

**Key takeaway:** Liquidity sweeps are a prerequisite confirmation for valid trade entries. Always look for a sweep before entering at an FVG or order block.

---

## 2. Liquidity Grab

**Teaching Duration:** 165s | **Word Count:** 490 | **Teaching Units:** 10

A liquidity grab differs from a sweep in that it involves a full candle body close beyond a consolidation range (not just a wick). Price breaks out of a ranging phase, trapping breakout traders, then reverses sharply in the opposite direction.

**Two types:**
- **Buy Side Liquidity Grab** (bearish context): Candle body closes above consolidation, then market reverses sharply downward.
- **Sell Side Liquidity Grab** (bullish context): Candle body closes below consolidation, then market reverses sharply upward.

**Critical requirement:** The reversal must occur in speed/expansion form (fast, sharp movement) to qualify as a valid liquidity grab.

**Key takeaway:** Both sweeps and grabs share the same objective -- gaining liquidity as the market's driving force for the next move.

---

## 3. Price Imbalance / Liquidity Void

**Teaching Duration:** 310s | **Word Count:** 870 | **Teaching Units:** 14

A price imbalance (or liquidity void) is an area where price moves sharply in one direction, creating an incomplete auction between buyers and sellers. Pending orders left unfilled during the sharp move create liquidity within the zone.

**Identification rules:**
- Measured from wick to wick of the sharp one-sided candle sequence.
- Number of candles does not matter -- only the one-sided speed of movement.
- Voids aligned with the prevailing trend are high probability; counter-trend voids are low probability.

**Trading rules:**
- Market will always return to fill the void (no time limit).
- Target the 50% level of the void for trade entry using the Fibonacci tool.
- Price can fill 100% of the void, but 50% is the minimum target for entries.

**Real chart examples** shown in the video demonstrate price returning to bearish and bullish liquidity voids, filling them, and continuing in the trend direction.

---

## 4. Fair Value Gap (FVG)

**Teaching Duration:** 200s | **Word Count:** 650 | **Teaching Units:** 12

The FVG is the refined form of a liquidity void. While a liquidity void marks the entire sharp move zone, an FVG precisely measures the wick-to-wick gap between non-adjacent candles in the sequence.

**Identification:**
- Measured as the gap between the wick of Candle 1 and the wick of Candle 3 (the middle candle's body spans through without the wicks overlapping).
- Multiple FVGs can exist within a single liquidity void.
- Bearish FVG: gap between the low wick of a higher candle and the high wick of a lower candle.
- Bullish FVG: gap between the high wick of a lower candle and the low wick of a higher candle.

**Trading rules:**
1. Trade FVGs only in the direction of the prevailing trend.
2. Always trade after a confirmed liquidity sweep or liquidity grab.
3. FVGs coupled with order blocks or supply/demand zones are highest probability.
4. Use the 50% level of the FVG zone for entry.
5. Preferably trade after an inducement or FTR (Failed To Reach) level.

---

## 5. Valid FVG Confirmation Rules

**Teaching Duration:** 406s | **Word Count:** 1010 | **Teaching Units:** 18

The largest section of the video, dedicated to teaching how to distinguish valid from invalid FVG zones. Not every FVG is tradable.

**Bearish (sell trend) rules:**
1. **Highest candle rule:** Draw FVG from the highest candle's wick in the swing. If the selected candle is not the highest, the FVG is invalid.
2. **Valid pullback rule:** An FVG in a pullback is only valid if the pullback itself swept liquidity from a previous high.

**Bullish (buy trend) rules (inverted):**
1. **Lowest candle rule:** Draw FVG from the lowest candle's wick in the swing.
2. **Valid pullback rule:** The pullback must have swept liquidity from a previous low.

**Video examples evaluated (A through K):**
- Valid examples: zones drawn from the highest/lowest candle wick (green checkmarks).
- Invalid examples (C, D, E, G, I): zones drawn from a non-highest/non-lowest candle (red X marks).

**Key takeaway:** Use horizontal lines to verify which candle is truly the highest/lowest when it is not visually obvious. Invalid FVG zones are why traders experience stop-loss hits -- the market does not respect zones drawn from the wrong candle.

---

## Concept Relationship Flow

```
Liquidity Sweep / Liquidity Grab
        |
        v
   (liquidity collected)
        |
        v
Price Imbalance / Liquidity Void  (broad zone of sharp one-sided movement)
        |
        v
Fair Value Gap (FVG)  (refined wick-to-wick measurement within void)
        |
        v
Valid FVG Confirmation Rules  (highest/lowest candle + valid pullback)
        |
        v
Trade Entry at 50% of validated FVG, in trend direction,
coupled with order block or supply/demand confluence
```

---

## Key Principles Taught

1. **Liquidity is the answer** to all questions about which zones are valid for trading.
2. **FVG is a refinement** of the broader liquidity void / price imbalance concept.
3. **Not all FVGs are valid** -- the highest/lowest candle wick rule is critical.
4. **Trend alignment is mandatory** -- only trade FVGs in the direction of the prevailing trend.
5. **Liquidity events precede valid entries** -- always confirm a sweep or grab before entering at an FVG.
6. **50% retracement level** of the FVG zone is the target entry point.
7. **Confluence increases probability** -- FVG + order block + supply/demand = highest probability setup.
