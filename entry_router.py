import os
import threading
import pandas as pd

import syncfix as v182

core = v182.core
dashboard_patch = v182.dashboard_patch

# -------------------------------------------------------------------
# BTC V1.8.3 ENTRY-ROUTER RESEARCH
#
# Overlay on top of the proven V1.8.2 exit-sync + profit-ratchet engine.
# Only the candidate/routing layer changes here.
# -------------------------------------------------------------------
core.STATE_FILE = os.path.join(core.DATA_DIR, "btc_v183_state.json")
core.TRADES_FILE = os.path.join(core.DATA_DIR, "btc_v183_trades.csv")
core.FEATURES_FILE = os.path.join(core.DATA_DIR, "btc_v183_entries.csv")
core.DECISIONS_FILE = os.path.join(core.DATA_DIR, "btc_v183_decisions.csv")
core.EVENTS_FILE = os.path.join(core.DATA_DIR, "btc_v183_management_events.csv")
core.SHADOW_FILE = os.path.join(core.DATA_DIR, "btc_v183_shadow_outcomes.csv")

CONTINUATION_ENGINES = {
    "EMA_SCALP",
    "MOMENTUM",
    "BREAKOUT",
    "TREND_PULLBACK",
}

_thread_ctx = threading.local()

_base_load_state = core.load_state
_base_signal_for = core.signal_for
_base_open_position = core.open_position
_base_open_positions_dashboard = core.open_positions_dashboard
_base_ui_meta = dashboard_patch.ui_meta


def load_state_v183():
    """Attach persistent routing metadata to each thread's state object."""
    state = _base_load_state()
    meta = state.setdefault("routing_meta", {})

    for key in core.STRATEGIES:
        row = meta.setdefault(key, {})
        row.setdefault("latched_signature", None)
        row.setdefault("suppressed_rearm", 0)
        row.setdefault("shadow_countertrend", 0)
        row.setdefault("shadow_tier_b", 0)
        row.setdefault("last_candidate_time", None)
        row.setdefault("last_candidate_signature", None)

    _thread_ctx.state = state
    return state


core.load_state = load_state_v183


def _routing_row(key):
    state = getattr(_thread_ctx, "state", None)
    if state is None:
        return None

    meta = state.setdefault("routing_meta", {})
    return meta.setdefault(
        key,
        {
            "latched_signature": None,
            "suppressed_rearm": 0,
            "shadow_countertrend": 0,
            "shadow_tier_b": 0,
            "last_candidate_time": None,
            "last_candidate_signature": None,
        },
    )


def _wait_for_rearm(sig):
    """Same uninterrupted setup is not a new trade opportunity."""
    out = dict(sig)
    feat = dict(sig.get("features") or {})

    feat["rearm_blocked"] = True
    feat["router_allowed"] = False
    feat["router_reason"] = "WAIT_SETUP_REARM"
    feat["execution_tier"] = "WAIT_REARM"

    out["signal"] = None
    out["reason"] = "WAIT_SETUP_REARM"
    out["allowed"] = False
    out["router_reason"] = "WAIT_SETUP_REARM"
    out["features"] = feat
    return out


def routed_signal_for(key, candles, snap):
    """A-only live execution, HTF alignment and event-based re-arming."""
    sig = _base_signal_for(key, candles, snap)
    row = _routing_row(key)

    if not sig:
        if row is not None:
            row["latched_signature"] = None
        return sig

    side = sig.get("signal")

    # A real NO_SETUP evaluation resets the event latch.
    if not side:
        if row is not None:
            row["latched_signature"] = None
        return sig

    feat = sig.setdefault("features", {})
    setup = str(sig.get("reason") or "")
    direction = str(snap.get("direction") or "NEUTRAL")
    raw_tier = str(feat.get("execution_tier") or "")

    # Direction/tier are included so a material context change can create a
    # new candidate even when the low-level detector remains continuously true.
    signature = f"{side}|{setup}|{direction}|{raw_tier}"

    if row is not None and row.get("latched_signature") == signature:
        row["suppressed_rearm"] = int(row.get("suppressed_rearm", 0)) + 1
        return _wait_for_rearm(sig)

    if row is not None:
        row["latched_signature"] = signature
        row["last_candidate_time"] = pd.Timestamp(sig["time"]).isoformat()
        row["last_candidate_signature"] = signature

    aligned = (
        (side == "LONG" and direction == "BULL")
        or (side == "SHORT" and direction == "BEAR")
    )

    feat["htf_direction_aligned"] = bool(aligned)
    feat["routing_policy"] = "V183_A_ONLY_HTF_ALIGNED"

    # Continuation engines may carry risk only with the HTF direction.
    # Reversal engines retain their own context logic.
    if key in CONTINUATION_ENGINES and not aligned:
        sig["allowed"] = False
        sig["router_reason"] = "COUNTERTREND_CONTINUATION"
        feat["router_allowed"] = False
        feat["router_reason"] = "COUNTERTREND_CONTINUATION"
        feat["execution_tier"] = "SHADOW_COUNTERTREND"

        if row is not None:
            row["shadow_countertrend"] = int(row.get("shadow_countertrend", 0)) + 1

        return sig

    # Lower-quality Tier B stays fully measured in Shadow Lab but risks €0.
    if raw_tier == "B":
        sig["allowed"] = False
        sig["router_reason"] = "TIER_B_SHADOW"
        feat["router_allowed"] = False
        feat["router_reason"] = "TIER_B_SHADOW"
        feat["execution_tier"] = "SHADOW_TIER_B"

        if row is not None:
            row["shadow_tier_b"] = int(row.get("shadow_tier_b", 0)) + 1

        return sig

    return sig


core.signal_for = routed_signal_for


def _cluster_id(sig):
    ts = pd.Timestamp(sig["time"])
    return f"{ts.strftime('%Y%m%dT%H%M')}|{sig['signal']}"


def open_position_v183(key, state, account, sig):
    """Keep accounts independent, but tag simultaneous same-side BTC exposure."""
    result = _base_open_position(key, state, account, sig)

    p = account.get("position")
    if not p:
        return result

    cid = _cluster_id(sig)
    p["correlation_cluster"] = cid

    members = []

    for other_key, other_account in state.get("strategies", {}).items():
        op = other_account.get("position")
        if not op:
            continue

        same_time = pd.Timestamp(op.get("entry_time")) == pd.Timestamp(p["entry_time"])
        same_side = op.get("side") == p.get("side")

        if same_time and same_side:
            op["correlation_cluster"] = cid
            members.append(other_key)

    if len(members) >= 2:
        member_text = ",".join(sorted(members))

        for member in members:
            op = state["strategies"][member].get("position")
            if not op:
                continue

            core.log_event({
                "schema_version": "7.0",
                "candidate_id": op["candidate_id"],
                "time": p["entry_time"],
                "strategy": member,
                "event": "CORRELATION_CLUSTER",
                "price": op["entry_price"],
                "r_value": None,
                "old_stop": op["stop"],
                "new_stop": op["stop"],
                "detail": f"cluster={cid};members={member_text}",
            })

    return result


core.open_position = open_position_v183


def open_positions_dashboard_v183(state, last_price):
    rows = _base_open_positions_dashboard(state, last_price)

    positions = {
        key: account.get("position")
        for key, account in state.get("strategies", {}).items()
    }

    counts = {}
    for p in positions.values():
        if p and p.get("correlation_cluster"):
            cid = p["correlation_cluster"]
            counts[cid] = counts.get(cid, 0) + 1

    for row in rows:
        p = positions.get(row.get("strategy")) or {}
        cid = p.get("correlation_cluster")
        n = counts.get(cid, 0) if cid else 0

        if n >= 2:
            row["correlation_cluster"] = cid
            row["setup"] = f"{row.get('setup', '')} · CORR ×{n}"

    return rows


core.open_positions_dashboard = open_positions_dashboard_v183


def ui_meta_v183(candles, state):
    out = _base_ui_meta(candles, state)
    out["core"] = "V1.8.3"
    out["ui"] = "Control Center 1.1"
    return out


dashboard_patch.ui_meta = ui_meta_v183

# Expose the active live-routing policy directly in the header.
dashboard_patch.DASH = dashboard_patch.DASH.replace(
    '<span class="chip">Risk <b>1.00%</b> / strategy</span>',
    '<span class="chip">Risk <b>1.00%</b> / strategy</span>'
    '<span class="chip">Routing <b>A-only · HTF aligned</b></span>'
)


if __name__ == "__main__":
    print(
        "[BTC V1.8.3 ENTRY-ROUTER] "
        "clean v183 dataset ON · HTF direction gate ON · "
        "Tier-B shadow-only ON · event re-arm ON · "
        "correlation intelligence ON · V1.8.2 profit ratchet preserved",
        flush=True,
    )

    # V1.8.2's anti-stale reconciliation now operates on the v183 paths
    # because the core file globals above were changed before this call.
    v182.startup_reconcile()
    core.main()
