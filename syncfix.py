import os
import pandas as pd
import numpy as np

import bot as core
import dashboard_patch  # installs the upgraded UI route on core.app

# Keep the original strategy/entry/exit logic. This patch only fixes position-management
# synchronization and guarantees that management receives the 5m structure fields it expects.
_original_check_position = core.check_position
_original_open_positions_dashboard = core.open_positions_dashboard
_enriched_cache_ts = None
_enriched_cache_row = None


def _enriched_management_bar(candle):
    global _enriched_cache_ts, _enriched_cache_row
    try:
        ts = pd.Timestamp(candle.timestamp)
        if _enriched_cache_ts is not None and ts == _enriched_cache_ts and _enriched_cache_row is not None:
            return _enriched_cache_row

        candles = core.load_candles()
        d5 = core.enrich(candles.set_index('timestamp')).reset_index()
        row = d5[d5.timestamp == ts]
        if len(row):
            _enriched_cache_ts = ts
            _enriched_cache_row = row.iloc[-1]
            return _enriched_cache_row
    except Exception as e:
        print('[WARN][SYNCFIX] could not enrich management bar:', repr(e), flush=True)

    # Fail safe: the old V1.8 manager accessed candle.ema20 directly. Supplying NaN keeps
    # TP/SL/time-exit management alive even if structure enrichment is temporarily unavailable.
    try:
        safe = candle.copy()
        safe['ema20'] = np.nan
        return safe
    except Exception:
        return candle


def synced_check_position(key, state, account, candle):
    return _original_check_position(key, state, account, _enriched_management_bar(candle))


core.check_position = synced_check_position


def _processed_price(state, fallback):
    """Never show a dashboard price newer than the trade-management cursor."""
    try:
        cursor = state.get('last_processed_5m')
        if not cursor or not os.path.exists(core.CANDLES_FILE):
            return fallback
        d = pd.read_csv(core.CANDLES_FILE, usecols=['timestamp', 'close'])
        d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True, errors='coerce')
        target = pd.Timestamp(cursor)
        if target.tzinfo is None:
            target = target.tz_localize('UTC')
        rows = d[d.timestamp <= target]
        return float(rows.iloc[-1].close) if len(rows) else fallback
    except Exception:
        return fallback


def synced_open_positions_dashboard(state, last_price):
    return _original_open_positions_dashboard(state, _processed_price(state, last_price))


core.open_positions_dashboard = synced_open_positions_dashboard


if __name__ == '__main__':
    print('[SYNCFIX] position manager enrichment ON · dashboard/state price sync ON', flush=True)
    core.main()
