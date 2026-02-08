# TradingMamba - Known Issues

**Last Updated:** 2026-02-08

---

## Issue #1: IDM Detection Inaccuracies on Higher Timeframes (W1)

**Status:** Open — Deferred to Valid Pullback (Component #6) enhancement
**Component:** Inducement (IDM) — `find_inducement()` in `smart_money_analyzer.py`
**Discovered:** 2026-02-08 via manual W1 BTCUSDT analysis (May 2022 - Dec 2023)

### Problem Summary

4 out of 10 MS events in the W1 bear market period have incorrect or missing IDM associations. The 6 that work correctly happen to have clean single-candle pullbacks nearby.

### Specific Cases (W1 BTCUSDT)

| # | MS Event | Current IDM | Problem | Correct IDM |
|---|----------|-------------|---------|-------------|
| 1 | Jun 13 LL $17,709 | $42,894 (Apr 18) | **Wrong** — 142% away, different price cycle | **$32,250** (May 30 week high) |
| 2 | Nov 21 LL $15,599 | NONE | **Missing** — FTX crash impulse | **~$21,053** (Nov 7 high, impulse per V4 Rule 3) |
| 3 | Jun 12 HL $24,797 | NONE | **Missing** — gradual decline, no single-candle pullback | **~$30,470** (Apr 17 pullback high) |
| 4 | Jul 10 HH $31,815 | NONE | **Missing** — sharp rally, minor pullback not detected | **~$29,600** (Jun 26 pullback low) |

### Root Causes (3 bugs in `find_inducement()`)

**1. Search boundary too wide (lines 1827-1837 / 1925-1935)**
- Code searches between `prev_swing_high_idx` (from sparse `find_swing_points(lookback=5)`) and the current swing
- On W1, the previous swing high can be months/years away in a different price regime
- Caused: Issue #1 ($42,894 IDM for a $17,709 swing low)

**2. Minor pullback detection too strict (lines 1844 / 1941)**
- Requires `highs[j] > highs[j-1] AND highs[j] > highs[j+1]` — single-candle local extremum
- On W1, pullbacks often span 2-3 weeks where no single candle is strictly higher than both neighbors
- Caused: Issues #3 and #4 (missing IDMs in gradual moves)

**3. Depth-first sorting picks wrong candidate (line 1878 / 1974)**
- `pullback_candidates.sort(key=lambda x: (-x['depth'], -x['index']))` — deepest first
- Combined with wide search window, selects distant-cycle pullbacks over nearby relevant ones
- Per Video 1: IDM = "first pullback on left side" (nearest, not deepest)

### Proposed Resolution

These issues will likely be resolved when enhancing the **Valid Pullback** component (#6), since:
- Video 3: "Valid pullback = valid inducement" — fixing pullback detection fixes IDM
- Video 3: "LIQUIDITY SWEEP is THE ONLY confirmation factor" for pullbacks
- Proper pullback detection naturally scopes to the correct price move
- The Valid Pullback component needs its own "move origin" detection (where the move toward the swing started), which is exactly what IDM search needs for its boundary

**If fixed independently, the approach would be:**

1. **Move-origin boundary**: For each swing, scan left from raw candle data to find where the move started (highest high for descent / lowest low for rally). Search for pullbacks only within this move.

2. **Multi-candle pullback detection**: As fallback when no single-candle extremum found, check 2-3 candle clusters.

3. **Nearest-first sorting**: `sort(key=lambda x: (-x['index'], -x['depth']))` — proximity first, depth as tiebreaker.

### ICT Rules Reference (from ML Training)

- **Video 1**: IDM = first pullback on left side of recent high/low
- **Video 2**: HH/HL ONLY valid if inducement swept; no sweep = single impulse
- **Video 3**: Valid pullback requires liquidity sweep; valid pullback = valid inducement
- **Video 4 Rule 3**: Impulse swing (no pullback) — the low/high itself IS inducement
- **Video 8**: Multiple IDMs — prioritize LARGEST, then MOST RECENT (same vicinity)

### Correct IDM Concept (confirmed)

- **Bullish IDM** (for swing HIGH): pullback LOW on the left side (swept on way up)
- **Bearish IDM** (for swing LOW): pullback HIGH on the left side (swept on way down)

---

## Issue #2: (reserved for future)

---

**File maintained by:** Claude Code analysis sessions
