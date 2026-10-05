import os
import pandas as pd
import numpy as np

import bot as core

# -------------------------------------------------------------------
# BTC V1.8.2 PROFIT-RATCHET RESEARCH
# New clean forward dataset because exit policy materially changed.
# V1.8.1 files remain untouched for comparison.
# -------------------------------------------------------------------
core.STATE_FILE = os.path.join(core.DATA_DIR, "btc_v182_state.json")
core.TRADES_FILE = os.path.join(core.DATA_DIR, "btc_v182_trades.csv")
core.FEATURES_FILE = os.path.join(core.DATA_DIR, "btc_v182_entries.csv")
core.DECISIONS_FILE = os.path.join(core.DATA_DIR, "btc_v182_decisions.csv")
core.EVENTS_FILE = os.path.join(core.DATA_DIR, "btc_v182_management_events.csv")
core.SHADOW_FILE = os.path.join(core.DATA_DIR, "btc_v182_shadow_outcomes.csv")

# Keep the proven V1.8.1 anti-stale protections.
core.CONFIRMED_PROFIT_CLOSE_R = 999.0
core.BE_TRIGGER_R = 999.0
core.DYNAMIC_PROTECT_R = 999.0

# Disable the old fixed +0.60R lock and close-based 0.85R trailing gap.
# V1.8.2 replaces both with one monotonic MFE ratchet below.
core.LOCK_TRIGGER_R = 999.0
core.TRAIL_TRIGGER_R = 999.0

# Structure protection remains a separate "thesis failed" protection.
core.STRUCTURE_PROTECT_MFE_R = 1.00
core.STRUCTURE_PROTECT_LOCK_R = 0.40

import dashboard_patch  # installs upgraded dashboard on core.app

_original_check_position = core.check_position
_original_open_position = core.open_position
_original_save_state = core.save_state
_original_open_positions_dashboard = core.open_positions_dashboard

_enriched_cache_ts = None
_enriched_cache_row = None
_reconciling = False


# -------------------------------------------------------------------
# Profit Ratchet 1.0
#
# Designed as a broad, monotonic hypothesis — not fitted per strategy.
# It protects progressively more of an excursion without choking a trade
# before 1R. A stop is changed only AFTER a confirmed 5m candle, so the
# system never assumes an impossible retroactive intrabar fill.
#
# MFE reached     Minimum locked R for NEXT bar
# 1.00R           +0.30R
# 1.25R           +0.75R
# 1.50R           +1.00R
# 1.75R           +1.20R
# 2.00R           +1.45R
# 2.50R           +1.90R
# 3.00R+          max(+2.35R, MFE - 0.65R)
# -------------------------------------------------------------------
def ratchet_floor_r(mfe):
    mfe = float(mfe or 0.0)

    if mfe >= 3.00:
        return max(2.35, mfe - 0.65)
    if mfe >= 2.50:
        return 1.90
    if mfe >= 2.00:
        return 1.45
    if mfe >= 1.75:
        return 1.20
    if mfe >= 1.50:
        return 1.00
    if mfe >= 1.25:
        return 0.75
    if mfe >= 1.00:
        return 0.30
    return None


def apply_profit_ratchet(key, p, t, close_r):
    floor_r = ratchet_floor_r(p.get("mfe_r", 0.0))
    if floor_r is None:
        return False

    entry = float(p["entry_price"])
    dist = float(p["stop_distance"])

    new_stop = (
        entry + floor_r * dist
        if p["side"] == "LONG"
        else entry - floor_r * dist
    )

    old_stop = float(p["stop"])
    core.move_stop(
        key,
        p,
        t,
        new_stop,
        "PROFIT_RATCHET",
        close_r,
    )

    changed = abs(float(p["stop"]) - old_stop) > 1e-9
    if changed:
        p["protected_stage"] = max(int(p.get("protected_stage", 0)), 4)
        p["exit_state"] = "RATCHET"
        p["ratchet_floor_r"] = floor_r

    return changed


def _enriched_management_bar(candle):
    """Return the exact confirmed 5m candle with EMA/ATR structure fields."""
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
        print("[WARN][V182] management enrichment failed:", repr(e), flush=True)

    # Fail safe: price-based TP/SL/time management stays alive.
    try:
        safe = candle.copy()
        safe["ema20"] = np.nan
        return safe
    except Exception:
        return candle


def synced_open_position(key, state, account, sig):
    result = _original_open_position(key, state, account, sig)

    p = account.get("position")
    if p:
        # Entry candle is known at entry and may not be replayed as management.
        p["last_checked_bar"] = pd.Timestamp(sig["time"]).isoformat()
        p["ratchet_floor_r"] = None

    return result


core.open_position = synced_open_position


def synced_check_position(key, state, account, candle):
    """Exactly-once management per confirmed 5m candle + progressive profit ratchet."""
    p = account.get("position")
    if not p:
        return None

    ts = pd.Timestamp(candle.timestamp)
    last_checked = p.get("last_checked_bar")

    if last_checked and ts <= pd.Timestamp(last_checked):
        return None

    managed_bar = _enriched_management_bar(candle)

    # The original manager first evaluates any stop/TP that existed BEFORE
    # this completed candle. This preserves causal execution semantics.
    result = _original_check_position(
        key,
        state,
        account,
        managed_bar,
    )

    surviving = account.get("position")
    if surviving is p:
        surviving["last_checked_bar"] = ts.isoformat()

        cl = float(managed_bar.close)
        close_r = (
            (cl - surviving["entry_price"]) / surviving["stop_distance"]
            if surviving["side"] == "LONG"
            else (surviving["entry_price"] - cl) / surviving["stop_distance"]
        )

        # Ratchet is armed only after this bar closes. It is therefore active
        # from the NEXT bar forward and cannot create a look-ahead fill.
        apply_profit_ratchet(
            key,
            surviving,
            ts,
            close_r,
        )

    return result


core.check_position = synced_check_position


def reconcile_open_positions(state, candles=None):
    """Replay only missing confirmed 5m bars for each open position."""
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

                core.check_position(
                    key,
                    state,
                    account,
                    bar,
                )
                processed += 1

        if processed:
            print(
                f"[RECONCILE][V182] managed {processed} missing position-bars",
                flush=True,
            )

        return processed

    finally:
        _reconciling = False


def save_state_with_reconciliation(state):
    """Never persist a position behind the confirmed-candle cache."""
    if any(
        st.get("position")
        for st in state.get("strategies", {}).values()
    ):
        try:
            reconcile_open_positions(state)
        except Exception as e:
            print("[WARN][V182] reconcile-before-save:", repr(e), flush=True)

    return _original_save_state(state)


core.save_state = save_state_with_reconciliation


def _safe_dashboard_price(state, fallback):
    """Do not show a market price newer than the slowest open position manager."""
    try:
        cursors = []

        global_cursor = state.get("last_processed_5m")
        if global_cursor:
            cursors.append(pd.Timestamp(global_cursor))

        for account in state.get("strategies", {}).values():
            p = account.get("position")
            if p and p.get("last_checked_bar"):
                cursors.append(pd.Timestamp(p["last_checked_bar"]))

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
        print("[WARN][V182] startup reconciliation:", repr(e), flush=True)


if __name__ == "__main__":
    print(
        "[BTC V1.8.2 PROFIT-RATCHET] "
        "clean v182 dataset ON · per-position sync ON · "
        "old fixed +0.60R lock OFF · MFE ratchet ON",
        flush=True,
    )

    startup_reconcile()
    core.main()
