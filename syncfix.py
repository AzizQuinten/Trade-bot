import os
import pandas as pd
import numpy as np

import bot as core

# -------------------------------------------------------------------
# BTC V1.8.1 EXIT-SYNC RESEARCH
# Clean forward dataset after V1.8.0's stale position-management issue.
# The old v180 files are kept untouched for post-mortem analysis.
# -------------------------------------------------------------------
core.STATE_FILE = os.path.join(core.DATA_DIR, "btc_v181_state.json")
core.TRADES_FILE = os.path.join(core.DATA_DIR, "btc_v181_trades.csv")
core.FEATURES_FILE = os.path.join(core.DATA_DIR, "btc_v181_entries.csv")
core.DECISIONS_FILE = os.path.join(core.DATA_DIR, "btc_v181_decisions.csv")
core.EVENTS_FILE = os.path.join(core.DATA_DIR, "btc_v181_management_events.csv")
core.SHADOW_FILE = os.path.join(core.DATA_DIR, "btc_v181_shadow_outcomes.csv")
# Keep the existing candle cache. Candles were not the corrupted research output.

# -------------------------------------------------------------------
# Exit Engine 4.0 — remove the aggressive "almost everything scratches
# at +0.08/+0.10R" behavior.
# -------------------------------------------------------------------
core.CONFIRMED_PROFIT_CLOSE_R = 999.0
core.BE_TRIGGER_R = 999.0
core.DYNAMIC_PROTECT_R = 999.0
core.STRUCTURE_PROTECT_MFE_R = 1.00
core.STRUCTURE_PROTECT_LOCK_R = 0.40

import dashboard_patch  # installs the upgraded dashboard route on core.app

_original_check_position = core.check_position
_original_open_position = core.open_position
_original_save_state = core.save_state
_original_open_positions_dashboard = core.open_positions_dashboard

_enriched_cache_ts = None
_enriched_cache_row = None
_reconciling = False


def _enriched_management_bar(candle):
    """Return the exact confirmed 5m bar enriched with EMA/ATR fields."""
    global _enriched_cache_ts, _enriched_cache_row

    try:
        ts = pd.Timestamp(candle.timestamp)

        if (
            _enriched_cache_ts is not None
            and ts == _enriched_cache_ts
            and _enriched_cache_row is not None
        ):
            return _enriched_cache_row

        candles = core.load_candles()
        d5 = core.enrich(candles.set_index("timestamp")).reset_index()
        row = d5[d5.timestamp == ts]

        if len(row):
            _enriched_cache_ts = ts
            _enriched_cache_row = row.iloc[-1]
            return _enriched_cache_row

    except Exception as e:
        print("[WARN][EXIT-SYNC] management enrichment failed:", repr(e), flush=True)

    try:
        safe = candle.copy()
        safe["ema20"] = np.nan
        return safe
    except Exception:
        return candle


def synced_open_position(key, state, account, sig):
    """Open normally, then attach a per-position management cursor."""
    result = _original_open_position(key, state, account, sig)

    p = account.get("position")
    if p:
        p["last_checked_bar"] = pd.Timestamp(sig["time"]).isoformat()

    return result


core.open_position = synced_open_position


def synced_check_position(key, state, account, candle):
    """Process each confirmed 5m candle at most once for each position."""
    p = account.get("position")
    if not p:
        return None

    ts = pd.Timestamp(candle.timestamp)
    last_checked = p.get("last_checked_bar")

    if last_checked:
        last_ts = pd.Timestamp(last_checked)
        if ts <= last_ts:
            return None

    result = _original_check_position(
        key,
        state,
        account,
        _enriched_management_bar(candle),
    )

    surviving = account.get("position")
    if surviving is p:
        surviving["last_checked_bar"] = ts.isoformat()

    return result


core.check_position = synced_check_position


def reconcile_open_positions(state, candles=None):
    """Replay only missing confirmed 5m bars for every open position."""
    global _reconciling

    if _reconciling:
        return 0

    if not any(
        st.get("position")
        for st in state.get("strategies", {}).values()
    ):
        return 0

    _reconciling = True
    try:
        if candles is None:
            candles = core.load_candles()

        if candles is None or not len(candles):
            return 0

        d5 = core.enrich(candles.set_index("timestamp")).reset_index()
        processed = 0

        for key, account in state["strategies"].items():
            p = account.get("position")
            if not p:
                continue

            cursor = pd.Timestamp(
                p.get("last_checked_bar") or p.get("entry_time")
            )

            pending = d5[d5.timestamp > cursor]

            for _, bar in pending.iterrows():
                if not account.get("position"):
                    break

                core.check_position(key, state, account, bar)
                processed += 1

        if processed:
            print(
                f"[RECONCILE] repaired {processed} missing position-management bars",
                flush=True,
            )

        return processed

    finally:
        _reconciling = False


def save_state_with_reconciliation(state):
    """Never persist a state that is behind the confirmed candle cache."""
    if any(
        st.get("position")
        for st in state.get("strategies", {}).values()
    ):
        try:
            reconcile_open_positions(state)
        except Exception as e:
            print("[WARN][EXIT-SYNC] reconcile-before-save:", repr(e), flush=True)

    return _original_save_state(state)


core.save_state = save_state_with_reconciliation


def _safe_dashboard_price(state, fallback):
    """Dashboard can never show a price newer than the slowest open position manager."""
    try:
        cursors = []

        global_cursor = state.get("last_processed_5m")
        if global_cursor:
            cursors.append(pd.Timestamp(global_cursor))

        for account in state.get("strategies", {}).values():
            p = account.get("position")
            if p:
                c = p.get("last_checked_bar")
                if c:
                    cursors.append(pd.Timestamp(c))

        if not cursors or not os.path.exists(core.CANDLES_FILE):
            return fallback

        safe_cursor = min(cursors)

        d = pd.read_csv(
            core.CANDLES_FILE,
            usecols=["timestamp", "close"],
        )
        d["timestamp"] = pd.to_datetime(
            d["timestamp"],
            utc=True,
            errors="coerce",
        )

        if safe_cursor.tzinfo is None:
            safe_cursor = safe_cursor.tz_localize("UTC")

        rows = d[d.timestamp <= safe_cursor]
        return float(rows.iloc[-1].close) if len(rows) else fallback

    except Exception:
        return fallback


def synced_open_positions_dashboard(state, last_price):
    return _original_open_positions_dashboard(
        state,
        _safe_dashboard_price(state, last_price),
    )


core.open_positions_dashboard = synced_open_positions_dashboard


def startup_reconcile():
    """Repair a persisted position before the main loop/dashboard starts."""
    try:
        state = core.load_state()

        if not any(
            st.get("position")
            for st in state.get("strategies", {}).values()
        ):
            return

        candles = core.update_candles(core.load_candles())

        if reconcile_open_positions(state, candles):
            _original_save_state(state)

    except Exception as e:
        print("[WARN][EXIT-SYNC] startup reconciliation:", repr(e), flush=True)


if __name__ == "__main__":
    print(
        "[EXIT-SYNC V1.8.1] per-position candle cursor ON · "
        "TP/SL reconciliation ON · early +0.08/+0.10R scratch rules OFF · "
        "clean v181 dataset ON",
        flush=True,
    )

    startup_reconcile()
    core.main()
