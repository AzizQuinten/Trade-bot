import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from flask import render_template_string

import bot as core


def _num(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _records(path):
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def ui_stats(state):
    s = dict(core.stats(state))
    trades = _records(core.TRADES_FILE)
    if len(trades) and "net_R" in trades:
        nr = pd.to_numeric(trades["net_R"], errors="coerce").fillna(0.0)
        s["expectancy_r"] = float(nr.mean()) if len(nr) else 0.0
    else:
        s["expectancy_r"] = 0.0
    peak = _num(s.get("peak_balance"), s.get("balance", 0.0))
    bal = _num(s.get("balance"))
    s["current_dd"] = ((bal / peak) - 1.0) * 100 if peak else 0.0
    return s


def strategy_cards(state):
    base = core.strategy_dashboard(state)
    out = []
    for x in base:
        item = dict(x)
        st = state["strategies"].get(item["key"], {})
        n = int(item.get("trades", 0) or 0)
        item["expectancy_r"] = _num(item.get("net_r")) / n if n else 0.0
        item["scans"] = int(st.get("scans", 0) or 0)
        item["has_position"] = bool(st.get("position"))
        item["position_side"] = (st.get("position") or {}).get("side")
        out.append(item)
    return out


def period_metrics():
    out = {k: {"trades": 0, "r": 0.0, "pnl": 0.0} for k in ("today", "week", "month")}
    t = _records(core.TRADES_FILE)
    if not len(t) or "exit_time" not in t:
        return out
    dt = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
    local = dt.dt.tz_convert(ZoneInfo("Europe/Amsterdam"))
    now = datetime.now(ZoneInfo("Europe/Amsterdam"))
    netr = pd.to_numeric(t.get("net_R", 0), errors="coerce").fillna(0.0)
    pnl = pd.to_numeric(t.get("pnl_eur", 0), errors="coerce").fillna(0.0)
    masks = {
        "today": local.dt.date == now.date(),
        "week": (local.dt.isocalendar().year == now.isocalendar().year) & (local.dt.isocalendar().week == now.isocalendar().week),
        "month": (local.dt.year == now.year) & (local.dt.month == now.month),
    }
    for name, mask in masks.items():
        mask = mask.fillna(False)
        out[name] = {"trades": int(mask.sum()), "r": float(netr[mask].sum()), "pnl": float(pnl[mask].sum())}
    return out


def research_funnel():
    out = {"candidates": 0, "executed": 0, "blocked": 0, "no_setup": 0, "top_blocks": []}
    d = _records(core.DECISIONS_FILE)
    if not len(d) or "decision" not in d:
        return out
    dec = d["decision"].fillna("UNKNOWN").astype(str)
    side = d["side"].fillna("NONE").astype(str) if "side" in d else pd.Series(["NONE"] * len(d))
    vc = dec.value_counts()
    out["candidates"] = int((side != "NONE").sum())
    out["executed"] = int(vc.get("OPEN", 0))
    out["blocked"] = int(sum(int(v) for k, v in vc.items() if k.startswith("BLOCK_")))
    out["no_setup"] = int(vc.get("NO_SETUP", 0))
    out["top_blocks"] = [{"name": str(k).replace("BLOCK_", "", 1), "count": int(v)} for k, v in vc.items() if str(k).startswith("BLOCK_")][:6]
    return out


def shadow_lab():
    out = {"total": 0, "target": 0, "stop": 0, "other": 0, "mfe": 0.0, "mae": 0.0}
    sh = _records(core.SHADOW_FILE)
    if not len(sh):
        return out
    out["total"] = int(len(sh))
    if "first_touch" in sh:
        ft = sh["first_touch"].fillna("OTHER").astype(str)
        out["target"] = int((ft == "TARGET").sum())
        out["stop"] = int((ft == "STOP").sum())
        out["other"] = int(len(sh) - out["target"] - out["stop"])
    if "MFE_R" in sh:
        out["mfe"] = _num(pd.to_numeric(sh["MFE_R"], errors="coerce").mean())
    if "MAE_R" in sh:
        out["mae"] = _num(pd.to_numeric(sh["MAE_R"], errors="coerce").mean())
    return out


def performance_lab():
    out = {"avg_mfe": 0.0, "avg_mae": 0.0, "avg_giveback": 0.0, "sides": [], "exits": [], "scores": []}
    t = _records(core.TRADES_FILE)
    if not len(t):
        return out
    for col in ["net_R", "pnl_eur", "MFE_R", "MAE_R", "giveback_R", "score_total"]:
        if col in t:
            t[col] = pd.to_numeric(t[col], errors="coerce")
    if "MFE_R" in t:
        out["avg_mfe"] = _num(t["MFE_R"].mean())
    if "MAE_R" in t:
        out["avg_mae"] = _num(t["MAE_R"].mean())
    if "giveback_R" in t:
        out["avg_giveback"] = _num(t["giveback_R"].mean())

    if "side" in t:
        for side, d in t.groupby("side"):
            nr = d["net_R"].fillna(0.0) if "net_R" in d else pd.Series(dtype=float)
            pnl = d["pnl_eur"].fillna(0.0) if "pnl_eur" in d else pd.Series(dtype=float)
            out["sides"].append({"name": str(side), "trades": len(d), "wr": 100.0 * float((pnl > 0).mean()), "net_r": float(nr.sum()), "exp": float(nr.mean())})

    if "reason" in t:
        for reason, d in t.groupby("reason"):
            nr = d["net_R"].fillna(0.0) if "net_R" in d else pd.Series(dtype=float)
            out["exits"].append({"name": str(reason), "trades": len(d), "net_r": float(nr.sum()), "exp": float(nr.mean())})
        out["exits"] = sorted(out["exits"], key=lambda x: x["trades"], reverse=True)[:8]

    if "score_total" in t and t["score_total"].notna().any():
        buckets = pd.cut(t["score_total"], bins=[-1, 69, 79, 89, 101], labels=["≤69", "70–79", "80–89", "90+"])
        for label in ["≤69", "70–79", "80–89", "90+"]:
            d = t[buckets == label]
            if len(d):
                nr = d["net_R"].fillna(0.0)
                pnl = d["pnl_eur"].fillna(0.0)
                out["scores"].append({"name": label, "trades": len(d), "wr": 100.0 * float((pnl > 0).mean()), "net_r": float(nr.sum()), "exp": float(nr.mean())})
    return out


def important_decisions(n=30):
    d = _records(core.DECISIONS_FILE)
    if not len(d):
        return []
    if "decision" in d:
        relevant = d[d["decision"].fillna("").astype(str) != "NO_SETUP"]
        if len(relevant):
            d = relevant
    return d.replace({np.nan: None}).tail(n).iloc[::-1].to_dict("records")


def ui_meta(candles, state):
    out = {"core": "V1.8.0", "ui": "Control Center 1.0", "fresh": "UNKNOWN", "last_candle": "—", "processed": "—"}
    try:
        if len(candles):
            ts = pd.Timestamp(candles.timestamp.iloc[-1])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60.0
            out["fresh"] = "FRESH" if age <= 15 else "STALE"
            out["last_candle"] = ts.tz_convert("Europe/Amsterdam").strftime("%H:%M")
        if state.get("last_processed_5m"):
            pt = pd.Timestamp(state["last_processed_5m"])
            if pt.tzinfo is None:
                pt = pt.tz_localize("UTC")
            out["processed"] = pt.tz_convert("Europe/Amsterdam").strftime("%H:%M")
    except Exception:
        pass
    return out


DASH = r'''<!doctype html>
<html lang="nl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta http-equiv="refresh" content="20"><meta name="theme-color" content="#080b10"><title>BTC V1 Quant Control Center</title>
<style>
:root{--bg:#080b10;--panel:#11161f;--panel2:#0d121a;--line:#263041;--text:#f4f7fb;--muted:#8d99ac;--green:#53db8f;--red:#ff6b75;--amber:#f2c75c;--blue:#6e9cff;--radius:18px}*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;font-variant-numeric:tabular-nums}.shell{max-width:1280px;margin:auto;padding:16px 16px 42px}.top{position:sticky;top:0;z-index:20;margin:-16px -16px 14px;padding:13px 16px 11px;background:rgba(8,11,16,.94);backdrop-filter:blur(18px);border-bottom:1px solid rgba(38,48,65,.7)}.head{display:flex;justify-content:space-between;align-items:center;gap:12px}.brand{font-size:34px;font-weight:950;letter-spacing:-1.3px}.sub{font-size:11px;color:var(--muted);margin-top:5px}.status{display:flex;align-items:center;gap:7px;padding:9px 12px;border-radius:999px;background:#113520;color:var(--green);font-size:11px;font-weight:900}.dot{width:8px;height:8px;border-radius:50%;background:currentColor;box-shadow:0 0 12px currentColor}.chips{display:flex;gap:7px;overflow:auto;margin-top:10px;scrollbar-width:none}.chips::-webkit-scrollbar{display:none}.chip{white-space:nowrap;background:#111720;border:1px solid var(--line);border-radius:999px;padding:6px 9px;color:var(--muted);font-size:9px}.chip b{color:var(--text)}.card{background:linear-gradient(180deg,var(--panel),#0f141c);border:1px solid var(--line);border-radius:var(--radius)}.grid{display:grid;gap:10px}.kpis{grid-template-columns:repeat(6,1fr)}.kpi{padding:14px}.label{font-size:9px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);font-weight:800}.big{font-size:24px;font-weight:900;margin-top:6px}.small{font-size:10px;color:var(--muted);margin-top:6px}.good{color:var(--green)!important}.bad{color:var(--red)!important}.warn{color:var(--amber)!important}.info{color:var(--blue)!important}.periods{grid-template-columns:repeat(3,1fr);margin-top:10px}.period{padding:12px 14px;display:flex;justify-content:space-between;align-items:center}.period b{font-size:16px}.section{margin-top:10px;padding:15px}.section-title{display:flex;justify-content:space-between;align-items:flex-end;gap:10px;margin-bottom:11px}.section-title h2{margin:0;font-size:17px}.section-title p{margin:0;color:var(--muted);font-size:9px}.badge{padding:4px 7px;border-radius:999px;font-size:9px;font-weight:850;background:#202735;color:#c4cfde}.badge.green{background:#113520;color:var(--green)}.badge.blue{background:#16243d;color:var(--blue)}.market-head{display:flex;justify-content:space-between;gap:12px}.regime{font-size:24px;font-weight:950;overflow-wrap:anywhere}.price{font-size:18px;font-weight:900;text-align:right}.market-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:7px;margin-top:13px}.mini{background:var(--panel2);border:1px solid #202939;border-radius:12px;padding:10px}.mini b{display:block;font-size:17px;margin-top:4px}.strategy-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.strategy{background:var(--panel2);border:1px solid #222c3c;border-radius:14px;padding:12px}.strategy-head{display:flex;justify-content:space-between;gap:8px}.strategy-title{font-size:12px;font-weight:900}.key{font-size:8px;color:var(--muted);margin-top:2px}.balance{font-size:20px;font-weight:900;margin-top:10px}.sstats{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:9px}.sstat{background:#101720;border-radius:8px;padding:7px}.sstat span{display:block;font-size:7px;color:var(--muted);text-transform:uppercase}.sstat b{font-size:11px}.strategy-foot{display:flex;justify-content:space-between;gap:5px;border-top:1px solid #202939;margin-top:9px;padding-top:8px;color:var(--muted);font-size:8px}.last{font-size:9px;color:#c4cfde;margin-top:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.open-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:9px}.trade{background:var(--panel2);border:1px solid #263244;border-radius:14px;padding:13px}.trade-head{display:flex;justify-content:space-between;gap:10px}.trade-r{font-size:22px;font-weight:950}.levels{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-top:9px}.level{background:#111925;border-radius:8px;padding:7px}.level span{display:block;font-size:7px;color:var(--muted);text-transform:uppercase}.level b{font-size:10px}.empty{background:var(--panel2);border:1px dashed #2b3546;border-radius:13px;padding:22px;text-align:center;color:var(--muted);font-size:11px}.two{display:grid;grid-template-columns:1.2fr 1fr;gap:10px}.funnel{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}.fbox{background:var(--panel2);border:1px solid #202939;border-radius:11px;padding:10px}.fbox b{display:block;font-size:18px}.fbox span{font-size:7px;color:var(--muted);text-transform:uppercase}.block{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #202939;font-size:9px}.block:last-child{border:0}.shadow{background:var(--panel2);border:1px solid #202939;border-radius:13px;padding:12px}.shadow-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:9px}.shadow-grid div{background:#111925;border-radius:8px;padding:8px;text-align:center}.shadow-grid b{display:block;font-size:15px}.shadow-grid span{font-size:7px;color:var(--muted)}.analytics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.abox{background:var(--panel2);border:1px solid #202939;border-radius:13px;padding:11px}.abox h3{font-size:11px;margin:0 0 7px}.arow{display:grid;grid-template-columns:1fr auto auto auto;gap:7px;border-bottom:1px solid #202939;padding:6px 0;font-size:8px}.arow:last-child{border:0}.tablewrap{overflow:auto;-webkit-overflow-scrolling:touch;border:1px solid #222c3c;border-radius:11px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:9px;background:var(--panel2)}th,td{padding:9px 8px;border-bottom:1px solid #222c3c;text-align:left}th{color:var(--muted);font-size:7px;text-transform:uppercase;position:sticky;top:0;background:#121923}tr:last-child td{border:0}.strong{font-weight:850}.reason{max-width:235px;overflow:hidden;text-overflow:ellipsis}.buttons{display:flex;gap:7px;flex-wrap:wrap}.btn{text-decoration:none;color:var(--text);padding:9px 10px;background:#1b2431;border:1px solid #303b4d;border-radius:9px;font-size:9px;font-weight:800}.footer{text-align:center;color:#5f6b7e;font-size:8px;padding-top:16px}@media(max-width:900px){.kpis{grid-template-columns:repeat(3,1fr)}.strategy-grid{grid-template-columns:repeat(2,1fr)}.analytics{grid-template-columns:1fr}.two{grid-template-columns:1fr}.market-grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:620px){.shell{padding:11px 11px 32px}.top{margin:-11px -11px 11px;padding:11px}.brand{font-size:28px}.kpis{grid-template-columns:repeat(2,1fr);gap:7px}.kpi{padding:12px}.big{font-size:22px}.periods{gap:7px}.period{padding:10px}.period b{font-size:14px}.section{padding:13px}.regime{font-size:19px}.market-grid{grid-template-columns:repeat(2,1fr)}.market-grid .mini:first-child{grid-column:span 2}.strategy-grid,.open-grid{grid-template-columns:1fr}.levels{grid-template-columns:repeat(2,1fr)}.funnel{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="shell"><div class="top"><div class="head"><div><div class="brand">BTC V1</div><div class="sub">Quant Control Center · independent strategy accounts · paper research</div></div><div class="status"><span class="dot"></span>RUNNING</div></div><div class="chips"><span class="chip">Core <b>{{meta.core}}</b></span><span class="chip">UI <b>{{meta.ui}}</b></span><span class="chip">Risk <b>1.00%</b> / strategy</span><span class="chip">Costs <b>OFF</b></span><span class="chip">Data <b class="{{'good' if meta.fresh=='FRESH' else 'warn'}}">{{meta.fresh}}</b></span><span class="chip">Last candle <b>{{meta.last_candle}}</b></span><span class="chip">Refresh <b>20s</b></span></div></div>
<div class="grid kpis"><div class="card kpi"><div class="label">Combined equity</div><div class="big {{'good' if s.return_pct>=0 else 'bad'}}">€{{'%.2f'|format(s.balance)}}</div><div class="small">{{'%+.2f'|format(s.return_pct)}}%</div></div><div class="card kpi"><div class="label">Net P/L</div><div class="big {{'good' if s.net_pnl>=0 else 'bad'}}">€{{'%+.2f'|format(s.net_pnl)}}</div><div class="small">{{'%+.2f'|format(s.net_r)}}R total</div></div><div class="card kpi"><div class="label">Trades</div><div class="big">{{s.trades}}</div><div class="small">{{s.wins}}W · {{s.losses}}L</div></div><div class="card kpi"><div class="label">Winrate / PF</div><div class="big">{{'%.1f'|format(s.winrate)}}%</div><div class="small">PF {{'%.2f'|format(s.pf) if s.pf<900 else '∞'}}</div></div><div class="card kpi"><div class="label">Expectancy</div><div class="big {{'good' if s.expectancy_r>0 else 'bad' if s.expectancy_r<0 else ''}}">{{'%+.2f'|format(s.expectancy_r)}}R</div><div class="small">per trade</div></div><div class="card kpi"><div class="label">Open risk / DD</div><div class="big">{{'%.2f'|format(s.open_risk_pct)}}%</div><div class="small">{{'%.2f'|format(s.max_dd)}}% worst DD</div></div></div>
<div class="grid periods">{% for key,title in [('today','Vandaag'),('week','Deze week'),('month','Deze maand')] %}<div class="card period"><div><div class="label">{{title}}</div><div class="small">{{periods[key].trades}} trades · €{{'%+.2f'|format(periods[key].pnl)}}</div></div><b class="{{'good' if periods[key].r>=0 else 'bad'}}">{{'%+.2f'|format(periods[key].r)}}R</b></div>{% endfor %}</div>
<div class="card section"><div class="section-title"><div><h2>Market Intelligence</h2><p>HTF context en live marktconditie</p></div><span class="badge blue">{{r.market_state}}</span></div><div class="market-head"><div><div class="regime">{{r.regime}}</div><div class="small">{{r.direction}} · {{r.strength}} · {{r.volatility}}</div></div><div><div class="price">${{'{:,.0f}'.format(r.price) if r.price else '—'}}</div><div class="small">BTC-USDT</div></div></div><div class="market-grid"><div class="mini"><div class="label">Direction</div><b class="{{'good' if r.direction=='BULL' else 'bad' if r.direction=='BEAR' else ''}}">{{r.direction}}</b></div><div class="mini"><div class="label">ADX 1H</div><b>{{'%.1f'|format(r.adx) if r.adx is not none else '—'}}</b></div><div class="mini"><div class="label">ER24 1H</div><b>{{'%.3f'|format(r.er24) if r.er24 is not none else '—'}}</b></div><div class="mini"><div class="label">Vol ratio</div><b>{{'%.2f'|format(r.vol_ratio) if r.vol_ratio is not none else '—'}}</b></div><div class="mini"><div class="label">Open positions</div><b>{{openpos|length}} / 6</b></div></div></div>
<div class="card section"><div class="section-title"><div><h2>Strategy Accounts</h2><p>Eigen equity, expectancy en execution per engine</p></div></div><div class="strategy-grid">{% for x in strat %}<div class="strategy"><div class="strategy-head"><div><div class="strategy-title">{{x.label}}</div><div class="key">{{x.key}}</div></div>{% if x.has_position %}<span class="badge blue">{{x.position_side}} OPEN</span>{% else %}<span class="badge green">LIVE</span>{% endif %}</div><div class="balance">€{{'%.2f'|format(x.account_balance)}}</div><div class="small {{'good' if x.account_return>0 else 'bad' if x.account_return<0 else ''}}">{{'%+.2f'|format(x.account_return)}}% · €{{'%+.2f'|format(x.pnl)}}</div><div class="sstats"><div class="sstat"><span>Trades</span><b>{{x.trades}}</b></div><div class="sstat"><span>WR</span><b>{{'%.0f'|format(x.winrate)}}%</b></div><div class="sstat"><span>Exp</span><b class="{{'good' if x.expectancy_r>0 else 'bad' if x.expectancy_r<0 else ''}}">{{'%+.2f'|format(x.expectancy_r)}}R</b></div><div class="sstat"><span>PF</span><b>{{'%.2f'|format(x.pf) if x.pf<900 else '∞'}}</b></div></div><div class="last">{{x.last_signal}}</div><div class="strategy-foot"><span>Signals {{x.signals}}</span><span>Blocked {{x.blocked}}</span><span>Scans {{x.scans}}</span></div></div>{% endfor %}</div></div>
<div class="card section"><div class="section-title"><div><h2>Open Trades</h2><p>Current R, levels en management state</p></div><span class="badge {{'green' if openpos else ''}}">{{openpos|length}} OPEN</span></div>{% if openpos %}<div class="open-grid">{% for p in openpos %}<div class="trade"><div class="trade-head"><div><div class="strategy-title">{{p.strategy}} · {{p.side}}</div><div class="small">{{p.setup}}</div></div><div class="trade-r {{'good' if p.current_r>=0 else 'bad'}}">{{'%+.2f'|format(p.current_r)}}R</div></div><div class="levels"><div class="level"><span>Entry</span><b>${{'{:,.0f}'.format(p.entry_price)}}</b></div><div class="level"><span>Current</span><b>${{'{:,.0f}'.format(p.current_price)}}</b></div><div class="level"><span>Stop</span><b>${{'{:,.0f}'.format(p.stop)}}</b></div><div class="level"><span>Target</span><b>${{'{:,.0f}'.format(p.target)}}</b></div></div><div class="chips"><span class="chip">MFE <b class="good">{{'%.2f'|format(p.mfe_r)}}R</b></span><span class="chip">MAE <b class="bad">{{'%.2f'|format(p.mae_r)}}R</b></span><span class="chip">Risk <b>€{{'%.2f'|format(p.risk_eur)}}</b></span><span class="chip">State <b>{{p.exit_state}}</b></span></div></div>{% endfor %}</div>{% else %}<div class="empty">Geen open trades — engines scannen door op nieuwe edge-quality setups.</div>{% endif %}</div>
<div class="two"><div class="card section"><div class="section-title"><div><h2>Research Funnel</h2><p>Candidate → block → execution</p></div></div><div class="funnel"><div class="fbox"><b>{{f.candidates}}</b><span>Candidates</span></div><div class="fbox"><b class="good">{{f.executed}}</b><span>Executed</span></div><div class="fbox"><b class="warn">{{f.blocked}}</b><span>Blocked</span></div><div class="fbox"><b>{{'%.1f'|format((100*f.executed/f.candidates) if f.candidates else 0)}}%</b><span>Exec rate</span></div></div><div class="label" style="margin-top:10px">Top block reasons</div>{% if f.top_blocks %}{% for b in f.top_blocks %}<div class="block"><span>{{b.name}}</span><b>{{b.count}}</b></div>{% endfor %}{% else %}<div class="small">Nog geen blocked candidates.</div>{% endif %}</div><div class="card section"><div class="section-title"><div><h2>Shadow Lab</h2><p>Afgewezen setups achteraf gemeten</p></div></div><div class="shadow"><div class="big">{{shadow.total}}</div><div class="small">afgeronde shadow outcomes</div><div class="shadow-grid"><div><b class="good">{{shadow.target}}</b><span>Target first</span></div><div><b class="bad">{{shadow.stop}}</b><span>Stop first</span></div><div><b>{{shadow.other}}</b><span>Other</span></div></div><div class="chips"><span class="chip">Avg MFE <b class="good">{{'%.2f'|format(shadow.mfe)}}R</b></span><span class="chip">Avg MAE <b class="bad">{{'%.2f'|format(shadow.mae)}}R</b></span></div></div></div></div>
<div class="card section"><div class="section-title"><div><h2>Performance Lab</h2><p>Waar edge zit en waar R weglekt</p></div></div><div class="analytics"><div class="abox"><h3>Long vs Short</h3>{% if perf.sides %}{% for x in perf.sides %}<div class="arow"><span>{{x.name}}</span><span>{{x.trades}}T</span><span>{{'%.0f'|format(x.wr)}}%</span><b class="{{'good' if x.exp>0 else 'bad' if x.exp<0 else ''}}">{{'%+.2f'|format(x.exp)}}R</b></div>{% endfor %}{% else %}<div class="small">Nog geen data.</div>{% endif %}</div><div class="abox"><h3>Exit Reasons</h3>{% if perf.exits %}{% for x in perf.exits %}<div class="arow"><span>{{x.name}}</span><span>{{x.trades}}T</span><span>{{'%+.2f'|format(x.net_r)}}R</span><b class="{{'good' if x.exp>0 else 'bad' if x.exp<0 else ''}}">{{'%+.2f'|format(x.exp)}}R</b></div>{% endfor %}{% else %}<div class="small">Nog geen data.</div>{% endif %}</div><div class="abox"><h3>Score Buckets</h3>{% if perf.scores %}{% for x in perf.scores %}<div class="arow"><span>{{x.name}}</span><span>{{x.trades}}T</span><span>{{'%.0f'|format(x.wr)}}%</span><b class="{{'good' if x.exp>0 else 'bad' if x.exp<0 else ''}}">{{'%+.2f'|format(x.exp)}}R</b></div>{% endfor %}{% else %}<div class="small">Nog geen data.</div>{% endif %}</div></div><div class="chips"><span class="chip">Avg MFE <b class="good">{{'%.2f'|format(perf.avg_mfe)}}R</b></span><span class="chip">Avg MAE <b class="bad">{{'%.2f'|format(perf.avg_mae)}}R</b></span><span class="chip">Avg giveback <b class="warn">{{'%.2f'|format(perf.avg_giveback)}}R</b></span><span class="chip">Current DD <b class="{{'bad' if s.current_dd<0 else ''}}">{{'%.2f'|format(s.current_dd)}}%</b></span></div></div>
<div class="card section"><div class="section-title"><div><h2>Recent Candidate Decisions</h2><p>NO_SETUP-ruis verborgen; alleen echte candidates</p></div><span class="badge">{{decisions|length}} shown</span></div>{% if decisions %}<div class="tablewrap"><table><tr><th>Time</th><th>Strategy</th><th>Side</th><th>Decision</th><th>Setup</th><th>Edge</th><th>Regime</th></tr>{% for d in decisions %}<tr><td>{{d.time[11:16] if d.time else '—'}}</td><td class="strong">{{d.strategy}}</td><td>{{d.side}}</td><td class="{{'good' if d.decision=='OPEN' else 'warn'}}">{{d.decision}}</td><td class="reason">{{d.setup}}</td><td>{{'%.0f'|format(d.edge_score) if d.edge_score is not none else '—'}}</td><td class="reason">{{d.regime}}</td></tr>{% endfor %}</table></div><div class="small" style="margin-top:8px">{{f.no_setup}} NO_SETUP evaluations blijven in Decisions CSV.</div>{% else %}<div class="empty">Nog geen relevante candidate decisions.</div>{% endif %}</div>
<div class="card section"><div class="section-title"><div><h2>Performance per Regime</h2><p>Forward-resultaten per marktcontext</p></div></div>{% if regimes %}<div class="tablewrap"><table><tr><th>Regime</th><th>Trades</th><th>WR</th><th>Net R</th><th>P/L</th></tr>{% for x in regimes %}<tr><td class="strong">{{x.regime}}</td><td>{{x.trades}}</td><td>{{'%.1f'|format(x.winrate)}}%</td><td class="{{'good' if x.net_r>0 else 'bad' if x.net_r<0 else ''}}">{{'%+.2f'|format(x.net_r)}}R</td><td>€{{'%+.2f'|format(x.pnl)}}</td></tr>{% endfor %}</table></div>{% else %}<div class="empty">Regime-statistieken verschijnen na gesloten trades.</div>{% endif %}</div>
<div class="card section"><div class="section-title"><div><h2>Trade History</h2><p>R, MFE, MAE, giveback en exit reason — zonder fictieve costs</p></div></div>{% if recent %}<div class="tablewrap"><table><tr><th>Exit</th><th>Strategy</th><th>Side</th><th>Reason</th><th>R</th><th>MFE</th><th>MAE</th><th>Giveback</th><th>P/L</th><th>Balance</th></tr>{% for t in recent %}<tr><td>{{t.exit_time[5:16]|replace('T',' ')}}</td><td class="strong">{{t.strategy}}</td><td>{{t.side}}</td><td>{{t.reason}}</td><td class="{{'good' if t.net_R>=0 else 'bad'}}">{{'%+.2f'|format(t.net_R)}}R</td><td class="good">{{'%.2f'|format(t.MFE_R)}}R</td><td class="bad">{{'%.2f'|format(t.MAE_R)}}R</td><td class="warn">{{'%.2f'|format(t.giveback_R)}}R</td><td>€{{'%+.2f'|format(t.pnl_eur)}}</td><td>€{{'%.2f'|format(t.balance)}}</td></tr>{% endfor %}</table></div>{% else %}<div class="empty">Nog geen gesloten V1.8 trades.</div>{% endif %}</div>
<div class="card section"><div class="section-title"><div><h2>Data & Exports</h2><p>Volledige researchdataset</p></div></div><div class="buttons"><a class="btn" href="/download/trades">↓ Trades CSV</a><a class="btn" href="/download/features">↓ Entry Features</a><a class="btn" href="/download/decisions">↓ Decisions</a><a class="btn" href="/download/events">↓ Management</a><a class="btn" href="/download/shadow">↓ Shadow Outcomes</a><a class="btn" href="/api/status">API Status</a></div></div><div class="footer">Trading core {{meta.core}} · {{meta.ui}} · paper research · last processed {{meta.processed}}</div></div></body></html>'''


def dashboard():
    st = core.load_state()
    c = core.load_candles()
    r = core.regime_snapshot(c)
    price = float(c.close.iloc[-1]) if len(c) else 0.0
    return render_template_string(DASH, s=ui_stats(st), r=r, strat=strategy_cards(st), openpos=core.open_positions_dashboard(st, price), decisions=important_decisions(30), regimes=core.regime_performance(), recent=core._read_csv_records(core.TRADES_FILE, 30), periods=period_metrics(), f=research_funnel(), shadow=shadow_lab(), perf=performance_lab(), meta=ui_meta(c, st))


core.app.view_functions["dashboard"] = dashboard

if __name__ == "__main__":
    core.main()
