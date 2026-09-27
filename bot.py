import os
import json
import time
import threading
from datetime import datetime, timezone, timedelta

from flask import Flask, jsonify, render_template_string, send_file
import requests
import pandas as pd
import numpy as np

# BTC V9 Strategy Lab — PAPER trading only.
# One Railway service runs all strategies and one dashboard.

INST_ID = os.getenv("INST_ID", "BTC-USDT")
START_BALANCE = float(os.getenv("START_BALANCE", "500"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
ROUND_TRIP_COST = float(os.getenv("ROUND_TRIP_COST", "0.0007"))
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))
BOOTSTRAP_DAYS = int(os.getenv("BOOTSTRAP_DAYS", "90"))
COOLDOWN_AFTER_LOSSES = 3
COOLDOWN_HOURS = 8

OKX_URL = os.getenv("OKX_PUBLIC_BASE", "https://www.okx.com")
HISTORY_ENDPOINT = "/api/v5/market/history-candles"
LIVE_ENDPOINT = "/api/v5/market/candles"

DATA_DIR = os.getenv("DATA_DIR", "/data")
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except PermissionError:
    DATA_DIR = "./data"
    os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE = os.path.join(DATA_DIR, "v9_state.json")
TRADES_FILE = os.path.join(DATA_DIR, "v9_trades.csv")
CANDLES_FILE = os.path.join(DATA_DIR, "v9_candles_5m.csv")
SHADOW_FILE = os.path.join(DATA_DIR, "v10_regime_shadow.csv")

STRATEGIES = {
    "SMC_SWEEP": {"label": "SMC Liquidity Sweep", "description": "Sweep van recente high/low + reclaim + momentumbevestiging.", "rr": 2.5, "stop_atr": 1.35, "max_hours": 6},
    "TREND_PULLBACK": {"label": "Trend Pullback", "description": "15m trend, pullback naar EMA20 en hervatting in trendrichting.", "rr": 2.5, "stop_atr": 1.6, "max_hours": 8},
    "BREAKOUT": {"label": "Breakout", "description": "15m range-breakout met volume en 1h trendfilter.", "rr": 3.0, "stop_atr": 1.8, "max_hours": 8},
    "MOMENTUM": {"label": "Momentum", "description": "Snelle 5m impuls met RSI/volume en 15m trendalignment.", "rr": 2.0, "stop_atr": 1.25, "max_hours": 4},
    "MEAN_REVERSION": {"label": "Mean Reversion", "description": "Extreme EMA-afstand + RSI-extreme in rustige 1h markt.", "rr": 1.7, "stop_atr": 1.4, "max_hours": 4},
    "EMA_SCALP": {"label": "EMA Scalp", "description": "5m EMA9/21 continuation met korte stop en target.", "rr": 1.8, "stop_atr": 1.15, "max_hours": 3},
}

session = requests.Session()
session.headers.update({"User-Agent": "BTC-V9-Strategy-Lab/1.0"})
state_lock = threading.RLock()


def utc_now():
    return datetime.now(timezone.utc)


def strategy_default():
    return {
        "balance": START_BALANCE, "peak_balance": START_BALANCE, "max_drawdown": 0.0,
        "position": None, "loss_streak": 0, "cooldown_until": None,
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "gross_profit": 0.0, "gross_loss": 0.0, "scans": 0, "signals": 0,
        "last_signal": "—", "last_signal_time": None,
    }


def default_state():
    return {"version": "V9.0", "created": utc_now().isoformat(), "last_processed_5m": None,
            "strategies": {k: strategy_default() for k in STRATEGIES}}


def normalize_state(state):
    state.setdefault("version", "V9.0")
    state.setdefault("created", utc_now().isoformat())
    state.setdefault("last_processed_5m", None)
    state.setdefault("strategies", {})
    for key in STRATEGIES:
        old = state["strategies"].get(key, {})
        fresh = strategy_default(); fresh.update(old)
        state["strategies"][key] = fresh
    return state


def save_state(state):
    with state_lock:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)


def load_state():
    with state_lock:
        if not os.path.exists(STATE_FILE):
            s = default_state(); save_state(s); return s
        try:
            with open(STATE_FILE, "r") as f:
                return normalize_state(json.load(f))
        except Exception:
            return default_state()


def okx_get(endpoint, params, retries=8):
    for attempt in range(retries):
        try:
            r = session.get(OKX_URL + endpoint, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 + attempt); continue
            r.raise_for_status()
            result = r.json()
            if result.get("code") != "0":
                raise RuntimeError(result)
            return result["data"]
        except Exception as e:
            wait = 2 + attempt
            print(f"[WARN] OKX error: {e!r}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("OKX request failed repeatedly")


def rows_to_df(rows):
    if not rows: return pd.DataFrame()
    cols = ["timestamp_ms","open","high","low","close","volume","volume_ccy","volume_quote","confirm"]
    d = pd.DataFrame(rows, columns=cols)
    d["timestamp_ms"] = pd.to_numeric(d["timestamp_ms"], errors="coerce")
    d["timestamp"] = pd.to_datetime(d["timestamp_ms"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["confirm"].astype(str) == "1"]
    return d[["timestamp","open","high","low","close","volume"]].dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def download_bootstrap():
    print("[BOOT] Downloading initial history...", flush=True)
    target = utc_now() - timedelta(days=BOOTSTRAP_DAYS)
    target_ms = int(target.timestamp() * 1000)
    cursor, all_rows = None, []
    while True:
        p = {"instId": INST_ID, "bar": "5m", "limit": "300"}
        if cursor is not None: p["after"] = str(cursor)
        rows = okx_get(HISTORY_ENDPOINT, p)
        if not rows: break
        all_rows.extend(rows)
        oldest = min(int(r[0]) for r in rows)
        cursor = oldest - 1
        if oldest <= target_ms: break
        time.sleep(0.13)
    d = rows_to_df(all_rows)
    d = d[d["timestamp"] >= target]
    d.to_csv(CANDLES_FILE, index=False)
    print(f"[BOOT] Ready: {len(d):,} candles", flush=True)
    return d


def load_candles():
    if not os.path.exists(CANDLES_FILE): return download_bootstrap()
    d = pd.read_csv(CANDLES_FILE); d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    return d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def update_candles(candles):
    latest = rows_to_df(okx_get(LIVE_ENDPOINT, {"instId": INST_ID, "bar": "5m", "limit": "300"}))
    d = pd.concat([candles, latest], ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    cutoff = d["timestamp"].max() - pd.Timedelta(days=100)
    d = d[d["timestamp"] >= cutoff].reset_index(drop=True)
    d.to_csv(CANDLES_FILE, index=False)
    return d


def resample_ohlcv(data, rule):
    x = data.set_index("timestamp")
    return x.resample(rule, label="right", closed="right").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()


def atr(data, n=14):
    pc = data["close"].shift()
    tr = pd.concat([data["high"]-data["low"], (data["high"]-pc).abs(), (data["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def rsi(series, n=14):
    delta = series.diff(); up = delta.clip(lower=0); dn = -delta.clip(upper=0)
    ag = up.ewm(alpha=1/n, adjust=False).mean(); al = dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + ag/al.replace(0, np.nan))


def adx(data, n=14):
    up = data["high"].diff(); down = -data["low"].diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=data.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=data.index)
    a = atr(data, n).replace(0, np.nan)
    pdi = 100 * plus.ewm(alpha=1/n, adjust=False).mean()/a
    mdi = 100 * minus.ewm(alpha=1/n, adjust=False).mean()/a
    dx = 100 * (pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def prepared_frames(candles):
    d5 = candles.copy().set_index("timestamp")
    d15 = resample_ohlcv(candles, "15min")
    d1 = resample_ohlcv(candles, "1h")
    for d in (d5, d15, d1):
        d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
        d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
        d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
        d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
        d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()
        d["atr"] = atr(d); d["rsi"] = rsi(d["close"], 14); d["vol_ma"] = d["volume"].rolling(20).mean()
    d1["adx"] = adx(d1)
    return d5, d15, d1



def closed_resample_ohlcv(data, rule):
    """Resample using only fully closed higher-timeframe candles."""
    if data is None or len(data) == 0:
        return pd.DataFrame()
    latest = pd.Timestamp(data["timestamp"].iloc[-1])
    out = resample_ohlcv(data, rule)
    return out[out.index <= latest].copy()


def v10_regime_snapshot(candles):
    """Shadow-only V10 regime classifier. It NEVER changes V9 entries."""
    empty = {
        "regime": "WARMUP", "direction": "MIXED", "adx": None,
        "er24": None, "vol_ratio": None, "price": None,
        "one_h_ema20_slope3": None, "four_h_ema20_slope2": None,
    }
    try:
        if candles is None or len(candles) < 10000:
            return empty
        d1 = closed_resample_ohlcv(candles, "1h")
        d4 = closed_resample_ohlcv(candles, "4h")
        if len(d1) < 800 or len(d4) < 220:
            return empty

        for d in (d1, d4):
            d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
            d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
            d["atr"] = atr(d, 14)
        d1["adx"] = adx(d1, 14)
        d1["ema20_slope3"] = d1["ema20"] / d1["ema20"].shift(3) - 1
        d4["ema20_slope2"] = d4["ema20"] / d4["ema20"].shift(2) - 1
        d1["er24"] = (d1["close"] - d1["close"].shift(24)).abs() / d1["close"].diff().abs().rolling(24).sum().replace(0, np.nan)
        d1["atr_pct"] = d1["atr"] / d1["close"]
        d1["atr_med30d"] = d1["atr_pct"].rolling(24 * 30, min_periods=24 * 20).median()
        d1["vol_ratio"] = d1["atr_pct"] / d1["atr_med30d"].replace(0, np.nan)

        c1, c4 = d1.iloc[-1], d4.iloc[-1]
        bull_dir = bool(c1["ema20"] > c1["ema50"] and c1["ema20_slope3"] > 0 and c4["ema20"] > c4["ema50"] and c4["ema20_slope2"] > 0)
        bear_dir = bool(c1["ema20"] < c1["ema50"] and c1["ema20_slope3"] < 0 and c4["ema20"] < c4["ema50"] and c4["ema20_slope2"] < 0)
        direction = "BULL" if bull_dir else "BEAR" if bear_dir else "MIXED"

        strong = bool(float(c1["adx"]) >= 25)
        high_vol = bool(np.isfinite(c1["vol_ratio"]) and float(c1["vol_ratio"]) >= 1.20)
        expansion = bool(np.isfinite(c1["er24"]) and float(c1["er24"]) >= 0.25)
        low_vol_chop = bool(np.isfinite(c1["vol_ratio"]) and float(c1["vol_ratio"]) < 0.90 and float(c1["adx"]) < 20 and np.isfinite(c1["er24"]) and float(c1["er24"]) < 0.18)

        if bull_dir and strong and high_vol and expansion:
            regime = "BULL_EXPANSION"
        elif bear_dir and strong and high_vol and expansion:
            regime = "BEAR_EXPANSION"
        elif low_vol_chop:
            regime = "CHOP_LOWVOL"
        else:
            regime = "TRANSITION"

        return {
            "regime": regime,
            "direction": direction,
            "adx": float(c1["adx"]) if np.isfinite(c1["adx"]) else None,
            "er24": float(c1["er24"]) if np.isfinite(c1["er24"]) else None,
            "vol_ratio": float(c1["vol_ratio"]) if np.isfinite(c1["vol_ratio"]) else None,
            "price": float(candles["close"].iloc[-1]),
            "one_h_ema20_slope3": float(c1["ema20_slope3"]) if np.isfinite(c1["ema20_slope3"]) else None,
            "four_h_ema20_slope2": float(c4["ema20_slope2"]) if np.isfinite(c4["ema20_slope2"]) else None,
        }
    except Exception as e:
        print(f"[WARN] V10 shadow regime error: {e!r}", flush=True)
        return empty


def v10_shadow_allowed(key, side, regime):
    """Research mapping only; V9 continues to trade regardless of this result."""
    if regime == "BULL_EXPANSION":
        return side == "LONG" and key in {"BREAKOUT", "TREND_PULLBACK", "SMC_SWEEP"}
    if regime == "BEAR_EXPANSION":
        return side == "SHORT" and key in {"BREAKOUT", "TREND_PULLBACK", "SMC_SWEEP"}
    if regime == "CHOP_LOWVOL":
        return key == "MEAN_REVERSION"
    return False


def log_shadow_entry(key, sig, snap):
    row = {
        "entry_time": sig["time"].isoformat(),
        "strategy": key,
        "side": sig["signal"],
        "setup": sig.get("reason", ""),
        "regime": snap.get("regime", "WARMUP"),
        "direction": snap.get("direction", "MIXED"),
        "shadow_allowed": bool(v10_shadow_allowed(key, sig["signal"], snap.get("regime", "WARMUP"))),
        "btc_price": snap.get("price"),
        "adx_1h": snap.get("adx"),
        "er24_1h": snap.get("er24"),
        "vol_ratio_1h": snap.get("vol_ratio"),
        "ema20_slope3_1h": snap.get("one_h_ema20_slope3"),
        "ema20_slope2_4h": snap.get("four_h_ema20_slope2"),
    }
    exists = os.path.exists(SHADOW_FILE)
    pd.DataFrame([row]).to_csv(SHADOW_FILE, mode="a" if exists else "w", header=not exists, index=False)

def signal_for(key, candles):
    if len(candles) < 600: return None
    d5, d15, d1 = prepared_frames(candles)
    if len(d15) < 220 or len(d1) < 220: return None
    t = d5.index[-1]; c5 = d5.iloc[-1]; p5 = d5.iloc[-2]; c15 = d15.iloc[-1]; p15 = d15.iloc[-2]; c1 = d1.iloc[-1]
    side, note, stop_distance = None, "NO_SETUP", None

    if key == "SMC_SWEEP":
        prior_hi = d5["high"].shift(1).rolling(24).max().iloc[-1]
        prior_lo = d5["low"].shift(1).rolling(24).min().iloc[-1]
        bull = c5["low"] < prior_lo and c5["close"] > prior_lo and c5["close"] > c5["open"] and c5["rsi"] > p5["rsi"]
        bear = c5["high"] > prior_hi and c5["close"] < prior_hi and c5["close"] < c5["open"] and c5["rsi"] < p5["rsi"]
        if bull: side, note = "LONG", "LOW_SWEEP_RECLAIM"
        elif bear: side, note = "SHORT", "HIGH_SWEEP_REJECT"
        wick = abs(float(c5["close"]-c5["low"])) if side == "LONG" else abs(float(c5["high"]-c5["close"]))
        stop_distance = max(float(c5["atr"]*STRATEGIES[key]["stop_atr"]), wick)

    elif key == "TREND_PULLBACK":
        bulltrend = c1["ema20"] > c1["ema50"] and c1["close"] > c1["ema50"]
        beartrend = c1["ema20"] < c1["ema50"] and c1["close"] < c1["ema50"]
        bull = bulltrend and p15["low"] <= p15["ema20"] and c15["close"] > c15["ema20"] and c15["close"] > p15["high"]
        bear = beartrend and p15["high"] >= p15["ema20"] and c15["close"] < c15["ema20"] and c15["close"] < p15["low"]
        if bull: side, note = "LONG", "BULL_PULLBACK_RESUME"
        elif bear: side, note = "SHORT", "BEAR_PULLBACK_RESUME"
        stop_distance = float(c15["atr"]*STRATEGIES[key]["stop_atr"])

    elif key == "BREAKOUT":
        hi = d15["high"].shift(1).rolling(12).max().iloc[-1]; lo = d15["low"].shift(1).rolling(12).min().iloc[-1]
        vol_ok = c15["volume"] > c15["vol_ma"]*1.1
        bull = c1["ema20"] > c1["ema50"] and c15["close"] > hi and vol_ok
        bear = c1["ema20"] < c1["ema50"] and c15["close"] < lo and vol_ok
        if bull: side, note = "LONG", "15M_RANGE_BREAK"
        elif bear: side, note = "SHORT", "15M_RANGE_BREAK"
        stop_distance = float(c15["atr"]*STRATEGIES[key]["stop_atr"])

    elif key == "MOMENTUM":
        body_ok = abs(float(c5["close"]-c5["open"])) > float(c5["atr"])*0.45
        vol_ok = c5["volume"] > c5["vol_ma"]*1.25
        bull = c15["ema20"] > c15["ema50"] and c5["close"] > c5["ema20"] and c5["rsi"] >= 58 and body_ok and vol_ok
        bear = c15["ema20"] < c15["ema50"] and c5["close"] < c5["ema20"] and c5["rsi"] <= 42 and body_ok and vol_ok
        if bull: side, note = "LONG", "5M_BULL_IMPULSE"
        elif bear: side, note = "SHORT", "5M_BEAR_IMPULSE"
        stop_distance = float(c5["atr"]*STRATEGIES[key]["stop_atr"])

    elif key == "MEAN_REVERSION":
        dist = (c5["close"]-c5["ema20"])/c5["atr"] if c5["atr"] else 0
        calm = float(c1["adx"]) < 24
        bull = calm and dist < -1.7 and c5["rsi"] < 30 and c5["close"] > c5["open"]
        bear = calm and dist > 1.7 and c5["rsi"] > 70 and c5["close"] < c5["open"]
        if bull: side, note = "LONG", "OVERSOLD_SNAPBACK"
        elif bear: side, note = "SHORT", "OVERBOUGHT_SNAPBACK"
        stop_distance = float(c5["atr"]*STRATEGIES[key]["stop_atr"])

    elif key == "EMA_SCALP":
        bull_cross = p5["ema9"] <= p5["ema21"] and c5["ema9"] > c5["ema21"]
        bear_cross = p5["ema9"] >= p5["ema21"] and c5["ema9"] < c5["ema21"]
        bull = bull_cross and c15["ema20"] > c15["ema50"] and c5["rsi"] > 52
        bear = bear_cross and c15["ema20"] < c15["ema50"] and c5["rsi"] < 48
        if bull: side, note = "LONG", "EMA9_21_CROSS"
        elif bear: side, note = "SHORT", "EMA9_21_CROSS"
        stop_distance = float(c5["atr"]*STRATEGIES[key]["stop_atr"])

    if side is None or stop_distance is None or not np.isfinite(stop_distance) or stop_distance <= 0:
        return {"signal": None, "time": t, "reason": note}
    return {"signal": side, "time": t, "entry": float(c5["close"]), "stop_distance": float(stop_distance), "reason": note}


def is_cooldown(s, now=None):
    val = s.get("cooldown_until")
    if not val: return False
    now = now or pd.Timestamp.now(tz="UTC"); until = pd.Timestamp(val)
    if now >= until:
        s["cooldown_until"] = None; return False
    return True


def open_position(key, s, sig):
    entry, dist, side = sig["entry"], sig["stop_distance"], sig["signal"]
    cfg = STRATEGIES[key]
    stop = entry-dist if side == "LONG" else entry+dist
    target = entry+cfg["rr"]*dist if side == "LONG" else entry-cfg["rr"]*dist
    s["position"] = {"side": side, "entry_time": sig["time"].isoformat(), "entry_price": entry, "stop": stop,
                     "target": target, "stop_distance": dist, "risk_pct": RISK_PER_TRADE, "setup": sig["reason"]}
    s["signals"] += 1; s["last_signal"] = f"{side} · {sig['reason']}"; s["last_signal_time"] = sig["time"].isoformat()
    print(f"[ENTRY][{key}] {side} entry={entry:.2f} stop={stop:.2f} tp={target:.2f}", flush=True)


def log_trade(key, row):
    row = dict(row); row["strategy"] = key
    cols = ["strategy","entry_time","exit_time","side","setup","entry","exit","stop","target","R","reason","pnl_eur","balance"]
    exists = os.path.exists(TRADES_FILE)
    pd.DataFrame([row], columns=cols).to_csv(TRADES_FILE, mode="a" if exists else "w", header=not exists, index=False)


def close_position(key, s, exit_price, exit_time, reason):
    p = s["position"]; entry = p["entry_price"]; dist = p["stop_distance"]
    R = (exit_price-entry)/dist if p["side"] == "LONG" else (entry-exit_price)/dist
    before = s["balance"]; pnl = before*(p["risk_pct"]*R - ROUND_TRIP_COST); after = before+pnl
    s["balance"] = after; s["peak_balance"] = max(s["peak_balance"], after)
    s["max_drawdown"] = min(s["max_drawdown"], after/s["peak_balance"]-1); s["total_trades"] += 1
    if pnl > 0:
        s["winning_trades"] += 1; s["gross_profit"] += pnl; s["loss_streak"] = 0
    else:
        s["losing_trades"] += 1; s["gross_loss"] += abs(pnl); s["loss_streak"] += 1
        if s["loss_streak"] >= COOLDOWN_AFTER_LOSSES:
            s["cooldown_until"] = (exit_time + timedelta(hours=COOLDOWN_HOURS)).isoformat()
    log_trade(key, {"entry_time":p["entry_time"],"exit_time":exit_time.isoformat(),"side":p["side"],"setup":p.get("setup",""),
                    "entry":entry,"exit":exit_price,"stop":p["stop"],"target":p["target"],"R":R,"reason":reason,"pnl_eur":pnl,"balance":after})
    s["position"] = None
    print(f"[EXIT][{key}] {reason} R={R:.2f} pnl=€{pnl:+.2f} bal=€{after:.2f}", flush=True)


def check_position(key, s, candle):
    p = s.get("position")
    if not p: return
    high, low, close, t = float(candle["high"]), float(candle["low"]), float(candle["close"]), candle["timestamp"]
    if p["side"] == "LONG":
        if low <= p["stop"]: close_position(key, s, p["stop"], t, "STOP"); return
        if high >= p["target"]: close_position(key, s, p["target"], t, "TAKE_PROFIT"); return
    else:
        if high >= p["stop"]: close_position(key, s, p["stop"], t, "STOP"); return
        if low <= p["target"]: close_position(key, s, p["target"], t, "TAKE_PROFIT"); return
    if t - pd.Timestamp(p["entry_time"]) >= pd.Timedelta(hours=STRATEGIES[key]["max_hours"]):
        close_position(key, s, close, t, "TIME_EXIT")


def stats_for(key, s):
    trades, wins = int(s["total_trades"]), int(s["winning_trades"])
    pf = s["gross_profit"]/s["gross_loss"] if s["gross_loss"] > 0 else (999.0 if s["gross_profit"] > 0 else 0.0)
    return {"key":key,"label":STRATEGIES[key]["label"],"description":STRATEGIES[key]["description"],"balance":float(s["balance"]),
            "return_pct":(float(s["balance"])/START_BALANCE-1)*100,"trades":trades,"winrate":wins/trades*100 if trades else 0.0,
            "pf":pf,"max_dd":float(s["max_drawdown"])*100,"position":s["position"]["side"] if s.get("position") else "—",
            "cooldown":is_cooldown(s),"signals":int(s.get("signals",0)),"scans":int(s.get("scans",0)),"last_signal":s.get("last_signal","—")}


DASHBOARD_HTML = '''
<!doctype html><html lang="nl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta http-equiv="refresh" content="20"><title>BTC V9 Strategy Lab</title>
<style>body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0d0f14;color:#f4f5f7;margin:0;padding:16px}.wrap{max-width:1100px;margin:auto}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}.ok{color:#59d18b}.muted{color:#9ca3af}.hero{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:12px}.grid{display:grid;grid-template-columns:1fr;gap:10px}.card{background:#171a22;border:1px solid #272b36;border-radius:14px;padding:14px}.big{font-size:24px;font-weight:700;margin-top:5px}.strategy h3{margin:0 0 4px}.mini{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}.mini div{background:#11141b;border-radius:10px;padding:9px}.v{font-size:17px;font-weight:700}.pos{color:#59d18b}.neg{color:#ff6b6b}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:8px 5px;border-bottom:1px solid #272b36}@media(min-width:760px){.hero{grid-template-columns:repeat(4,1fr)}.grid{grid-template-columns:repeat(2,1fr)}}</style></head>
<body><div class="wrap"><div class="top"><div><h1 style="margin:0">BTC V9 Strategy Lab</h1><div class="muted">6 algoritmes · 1 Railway server · paper only</div></div><div class="ok">● ONLINE</div></div><div style="margin:0 0 12px;display:flex;gap:8px;flex-wrap:wrap"><a href="/download/trades" style="display:inline-block;background:#171a22;border:1px solid #272b36;border-radius:10px;padding:10px 14px;color:#f4f5f7;text-decoration:none;font-weight:700">Download trades CSV</a><a href="/download/shadow" style="display:inline-block;background:#171a22;border:1px solid #272b36;border-radius:10px;padding:10px 14px;color:#f4f5f7;text-decoration:none;font-weight:700">Download V10 shadow CSV</a></div>
<div class="hero"><div class="card"><div class="muted">BTC</div><div class="big">${{ "{:,.0f}".format(s.btc_price) if s.btc_price else "—" }}</div></div><div class="card"><div class="muted">V10 shadow regime</div><div class="big">{{ s.shadow.regime }}</div><div class="muted">ADX {{ "%.1f"|format(s.shadow.adx) if s.shadow.adx is not none else "—" }} · Vol {{ "%.2f"|format(s.shadow.vol_ratio) if s.shadow.vol_ratio is not none else "—" }}</div></div><div class="card"><div class="muted">Lab portfolio</div><div class="big {{ 'pos' if s.portfolio_return >= 0 else 'neg' }}">{{ "%+.2f"|format(s.portfolio_return) }}%</div></div><div class="card"><div class="muted">Totaal trades</div><div class="big">{{ s.total_trades }}</div></div><div class="card"><div class="muted">Open posities</div><div class="big">{{ s.open_positions }}</div></div></div>
<div class="grid">{% for x in s.strategies %}<div class="card strategy"><h3>{{ x.label }}</h3><div class="muted">{{ x.description }}</div><div class="mini"><div><span class="muted">Return</span><br><span class="v {{ 'pos' if x.return_pct >= 0 else 'neg' }}">{{ "%+.2f"|format(x.return_pct) }}%</span></div><div><span class="muted">Trades</span><br><span class="v">{{ x.trades }}</span></div><div><span class="muted">Winrate</span><br><span class="v">{{ "%.0f"|format(x.winrate) }}%</span></div><div><span class="muted">PF</span><br><span class="v">{{ "%.2f"|format(x.pf) if x.pf < 900 else "∞" }}</span></div><div><span class="muted">Max DD</span><br><span class="v">{{ "%.2f"|format(x.max_dd) }}%</span></div><div><span class="muted">Positie</span><br><span class="v">{{ x.position }}</span></div></div><div class="muted" style="margin-top:10px">Signals: {{ x.signals }} · Scans: {{ x.scans }} · Laatste: {{ x.last_signal }}</div></div>{% endfor %}</div>
<div class="card" style="margin-top:10px"><div class="muted">Laatste trades</div><table><thead><tr><th>Strategie</th><th>Side</th><th>Exit</th><th>R</th><th>P/L</th></tr></thead><tbody>{% for t in trades %}<tr><td>{{ t.strategy }}</td><td>{{ t.side }}</td><td>{{ t.reason }}</td><td>{{ "%.2f"|format(t.R) }}</td><td class="{{ 'pos' if t.pnl_eur >= 0 else 'neg' }}">€{{ "%+.2f"|format(t.pnl_eur) }}</td></tr>{% else %}<tr><td colspan="5" class="muted">Nog geen afgeronde trades.</td></tr>{% endfor %}</tbody></table></div>
<div class="muted" style="margin-top:14px">Auto-refresh 20s · Iedere strategie heeft eigen virtueel €{{ "%.0f"|format(s.start_balance) }} account · Portfolio = gelijkgewogen gemiddelde · {{ s.updated }}</div></div></body></html>
'''

app = Flask(__name__)


def dashboard_snapshot():
    state = load_state(); rows = [stats_for(k, state["strategies"][k]) for k in STRATEGIES]
    btc_price = None
    try:
        if os.path.exists(CANDLES_FILE):
            c = pd.read_csv(CANDLES_FILE, usecols=["close"])
            if len(c): btc_price = float(c["close"].iloc[-1])
    except Exception: pass
    shadow = {"regime":"WARMUP","adx":None,"vol_ratio":None}
    try:
        candles = load_candles()
        shadow = v10_regime_snapshot(candles)
    except Exception:
        pass
    return {"btc_price":btc_price,"portfolio_return":float(np.mean([x["return_pct"] for x in rows])) if rows else 0.0,
            "total_trades":sum(x["trades"] for x in rows),"open_positions":sum(1 for x in rows if x["position"] != "—"),
            "strategies":rows,"shadow":shadow,"start_balance":START_BALANCE,"updated":utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")}


def recent_trades(limit=20):
    if not os.path.exists(TRADES_FILE): return []
    try:
        df = pd.read_csv(TRADES_FILE).tail(limit).iloc[::-1]
        return df[["strategy","side","reason","R","pnl_eur"]].to_dict("records")
    except Exception: return []


@app.get("/")
def dashboard(): return render_template_string(DASHBOARD_HTML, s=dashboard_snapshot(), trades=recent_trades())

@app.get("/api/status")
def api_status(): return jsonify(dashboard_snapshot())

@app.get("/download/trades")
def download_trades():
    if not os.path.exists(TRADES_FILE):
        return {"error": "Nog geen tradebestand beschikbaar."}, 404
    return send_file(
        TRADES_FILE,
        mimetype="text/csv",
        as_attachment=True,
        download_name="v9_trades.csv"
    )


@app.get("/download/shadow")
def download_shadow():
    if not os.path.exists(SHADOW_FILE):
        return {"error": "Nog geen V10 shadow entries beschikbaar."}, 404
    return send_file(
        SHADOW_FILE,
        mimetype="text/csv",
        as_attachment=True,
        download_name="v10_regime_shadow.csv"
    )

@app.get("/health")
def health(): return {"status":"ok","version":"V9.1-shadow","strategies":len(STRATEGIES)}, 200


def run_dashboard():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8080")), threaded=True, use_reloader=False)


def main():
    threading.Thread(target=run_dashboard, daemon=True).start()
    print("="*72, flush=True); print("BTC V9 STRATEGY LAB — PAPER ONLY", flush=True)
    print(f"Strategies: {len(STRATEGIES)} | One Railway service", flush=True)
    print(f"Instrument: {INST_ID} | Start balance per strategy: €{START_BALANCE:.2f}", flush=True)
    print(f"Persistent directory: {DATA_DIR}", flush=True); print("="*72, flush=True)
    state = load_state(); candles = update_candles(load_candles())
    print(f"[BOOT] Candle cache: {len(candles):,} rows", flush=True)
    while True:
        try:
            candles = update_candles(candles); newest = candles["timestamp"].iloc[-1]; prev = state.get("last_processed_5m")
            if prev is None or newest > pd.Timestamp(prev):
                new_rows = candles if prev is None else candles[candles["timestamp"] > pd.Timestamp(prev)]
                for _, candle in new_rows.iterrows():
                    for key in STRATEGIES: check_position(key, state["strategies"][key], candle)
                    state["last_processed_5m"] = candle["timestamp"].isoformat()
                shadow_snap = v10_regime_snapshot(candles)
                for key in STRATEGIES:
                    s = state["strategies"][key]; s["scans"] += 1; sig = signal_for(key, candles)
                    if sig and sig.get("signal"):
                        s["last_signal"] = f"{sig['signal']} · {sig['reason']}"; s["last_signal_time"] = sig["time"].isoformat()
                        if s.get("position") is None and not is_cooldown(s):
                            log_shadow_entry(key, sig, shadow_snap)
                            open_position(key, s, sig)
                save_state(state)
                total = sum(state["strategies"][k]["total_trades"] for k in STRATEGIES)
                opens = sum(1 for k in STRATEGIES if state["strategies"][k]["position"])
                print(f"[STATUS] {utc_now().strftime('%Y-%m-%d %H:%M UTC')} BTC=${candles['close'].iloc[-1]:,.0f} total_trades={total} open={opens} regime={shadow_snap.get('regime')}", flush=True)
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            save_state(state); break
        except Exception as e:
            print(f"[ERROR] {e!r}", flush=True); save_state(state); time.sleep(60)


if __name__ == "__main__":
    main()
