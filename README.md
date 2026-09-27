# BTC V9 + V10 Regime Shadow

This package keeps the original V9 paper-trading logic unchanged and adds a **shadow-only V10 regime classifier**.

The shadow classifier does NOT block, open, close, resize, or otherwise change any V9 trade. It only records what V10 *would* have allowed at the moment each real V9 paper entry is opened.

## Regimes
- BULL_EXPANSION
- BEAR_EXPANSION
- CHOP_LOWVOL
- TRANSITION
- WARMUP

The classifier uses only closed 1h/4h candles and evaluates:
- 1h EMA20 vs EMA50 and 3h EMA20 slope
- 4h EMA20 vs EMA50 and 8h EMA20 slope
- 1h ADX
- 24h efficiency ratio
- 1h ATR% relative to its 30-day median

## Shadow mapping
- BULL_EXPANSION: LONG Breakout / Trend Pullback / SMC are marked allowed
- BEAR_EXPANSION: SHORT Breakout / Trend Pullback / SMC are marked allowed
- CHOP_LOWVOL: Mean Reversion is marked allowed
- TRANSITION: no trade is marked allowed

Again: this is research logging only. V9 continues trading exactly as before.

## New dashboard items
- Current V10 shadow regime
- ADX and volatility ratio
- Download button for `v10_regime_shadow.csv`

## Files on persistent Railway volume
- `/data/v9_state.json` — unchanged
- `/data/v9_trades.csv` — unchanged
- `/data/v9_candles_5m.csv` — unchanged
- `/data/v10_regime_shadow.csv` — new

The shadow CSV can later be joined with V9 trades using `entry_time + strategy + side` to measure whether the regime filter would have improved forward performance.
