import os, json, time, threading
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, render_template_string, send_file
import requests
import pandas as pd
import numpy as np

# BTC V1.3 — stable decision schema + cost transparency
INST_ID = os.getenv('INST_ID', 'BTC-USDT')
START_BALANCE = float(os.getenv('START_BALANCE', '500'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.01'))
TAKER_FEE = float(os.getenv('TAKER_FEE', '0.00035'))
SLIPPAGE_BPS = float(os.getenv('SLIPPAGE_BPS', '1.0'))
POLL_SECONDS = int(os.getenv('POLL_SECONDS', '20'))
BOOTSTRAP_DAYS = int(os.getenv('BOOTSTRAP_DAYS', '90'))
MAX_TOTAL_RISK = float(os.getenv('MAX_TOTAL_RISK', '0.015'))
COOLDOWN_AFTER_LOSSES = 3
COOLDOWN_HOURS = 8
OKX_URL = os.getenv('OKX_PUBLIC_BASE', 'https://www.okx.com')
HISTORY_ENDPOINT = '/api/v5/market/history-candles'
LIVE_ENDPOINT = '/api/v5/market/candles'
DATA_DIR = os.getenv('DATA_DIR', '/data')
try: os.makedirs(DATA_DIR, exist_ok=True)
except PermissionError:
    DATA_DIR = './data'; os.makedirs(DATA_DIR, exist_ok=True)
STATE_FILE = os.path.join(DATA_DIR, 'btc_v1_state.json')
TRADES_FILE = os.path.join(DATA_DIR, 'btc_v1_trades.csv')
FEATURES_FILE = os.path.join(DATA_DIR, 'btc_v1_entry_features.csv')
DECISIONS_FILE = os.path.join(DATA_DIR, 'btc_v1_decisions.csv')
CANDLES_FILE = os.path.join(DATA_DIR, 'btc_v1_candles_5m.csv')

STRATEGIES = {
 'SMC_SWEEP': {'label':'SMC Liquidity Sweep','rr':2.5,'stop_atr':1.35,'max_hours':6,'enabled':True},
 'EMA_SCALP': {'label':'EMA Scalp','rr':1.8,'stop_atr':1.15,'max_hours':3,'enabled':True},
 'MOMENTUM': {'label':'Momentum Continuation','rr':2.0,'stop_atr':1.30,'max_hours':4,'enabled':True},
 'BREAKOUT': {'label':'Confirmed Breakout','rr':2.4,'stop_atr':1.65,'max_hours':8,'enabled':True},
 # Rebuilt, but kept shadow-only until it proves an edge in the new feature log.
 'TREND_PULLBACK': {'label':'Trend Pullback 2.0','rr':2.4,'stop_atr':1.55,'max_hours':8,'enabled':False},
 'MEAN_REVERSION': {'label':'Mean Reversion 2.0','rr':1.6,'stop_atr':1.35,'max_hours':4,'enabled':False},
}

session=requests.Session(); session.headers.update({'User-Agent':'BTC-V1/1.3'})
state_lock=threading.RLock()
def utc_now(): return datetime.now(timezone.utc)
def finite(x):
    try: return bool(np.isfinite(float(x)))
    except Exception: return False

def strategy_default():
    return {'position':None,'loss_streak':0,'cooldown_until':None,'scans':0,'signals':0,'blocked':0,'last_signal':'—','last_signal_time':None}
def default_state():
    return {'version':'BTC-V1.3','created':utc_now().isoformat(),'last_processed_5m':None,'balance':START_BALANCE,'peak_balance':START_BALANCE,'max_drawdown':0.0,'total_trades':0,'winning_trades':0,'losing_trades':0,'gross_profit':0.0,'gross_loss':0.0,'strategies':{k:strategy_default() for k in STRATEGIES}}
def normalize_state(s):
    base=default_state()
    for k,v in base.items(): s.setdefault(k,v)
    s.setdefault('strategies',{})
    for k in STRATEGIES:
        fresh=strategy_default(); fresh.update(s['strategies'].get(k,{})); s['strategies'][k]=fresh
    return s
def save_state(s):
    with state_lock:
        tmp=STATE_FILE+'.tmp'
        with open(tmp,'w') as f: json.dump(s,f,indent=2)
        os.replace(tmp,STATE_FILE)
def load_state():
    if not os.path.exists(STATE_FILE):
        s=default_state(); save_state(s); return s
    try:
        with open(STATE_FILE) as f: return normalize_state(json.load(f))
    except Exception: return default_state()

def okx_get(endpoint,params,retries=8):
    for attempt in range(retries):
        try:
            r=session.get(OKX_URL+endpoint,params=params,timeout=20)
            if r.status_code==429: time.sleep(2+attempt); continue
            r.raise_for_status(); out=r.json()
            if out.get('code')!='0': raise RuntimeError(out)
            return out['data']
        except Exception as e:
            print(f'[WARN] OKX {e!r}; retry {attempt+1}',flush=True); time.sleep(2+attempt)
    raise RuntimeError('OKX request failed')
def rows_to_df(rows):
    if not rows:return pd.DataFrame()
    cols=['timestamp_ms','open','high','low','close','volume','volume_ccy','volume_quote','confirm']
    d=pd.DataFrame(rows,columns=cols); d['timestamp_ms']=pd.to_numeric(d['timestamp_ms'],errors='coerce')
    d['timestamp']=pd.to_datetime(d['timestamp_ms'],unit='ms',utc=True)
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d[d['confirm'].astype(str)=='1']
    return d[['timestamp','open','high','low','close','volume']].dropna().drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
def download_bootstrap():
    target=utc_now()-timedelta(days=BOOTSTRAP_DAYS); target_ms=int(target.timestamp()*1000); cursor=None; rows=[]
    while True:
        p={'instId':INST_ID,'bar':'5m','limit':'300'}
        if cursor is not None:p['after']=str(cursor)
        x=okx_get(HISTORY_ENDPOINT,p)
        if not x:break
        rows.extend(x); oldest=min(int(r[0]) for r in x); cursor=oldest-1
        if oldest<=target_ms:break
        time.sleep(.13)
    d=rows_to_df(rows); d=d[d.timestamp>=target]; d.to_csv(CANDLES_FILE,index=False); return d
def load_candles():
    if not os.path.exists(CANDLES_FILE):return download_bootstrap()
    d=pd.read_csv(CANDLES_FILE); d['timestamp']=pd.to_datetime(d['timestamp'],utc=True); return d.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
def update_candles(c):
    latest=rows_to_df(okx_get(LIVE_ENDPOINT,{'instId':INST_ID,'bar':'5m','limit':'300'}))
    d=pd.concat([c,latest],ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    d=d[d.timestamp>=d.timestamp.max()-pd.Timedelta(days=100)].reset_index(drop=True); d.to_csv(CANDLES_FILE,index=False); return d

def resample_ohlcv(data,rule):
    x=data.set_index('timestamp')
    return x.resample(rule,label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
def closed_resample(data,rule):
    # With confirmed 5m bars and right-closed buckets, only timestamps <= latest confirmed 5m close are valid.
    if data is None or len(data)==0:return pd.DataFrame()
    latest=pd.Timestamp(data.timestamp.iloc[-1]); return resample_ohlcv(data,rule).loc[lambda x:x.index<=latest].copy()
def atr(d,n=14):
    pc=d.close.shift(); tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1); return tr.ewm(alpha=1/n,adjust=False).mean()
def rsi(s,n=14):
    z=s.diff(); up=z.clip(lower=0); dn=-z.clip(upper=0); ag=up.ewm(alpha=1/n,adjust=False).mean(); al=dn.ewm(alpha=1/n,adjust=False).mean(); return 100-100/(1+ag/al.replace(0,np.nan))
def adx(d,n=14):
    up=d.high.diff(); down=-d.low.diff(); plus=pd.Series(np.where((up>down)&(up>0),up,0.),index=d.index); minus=pd.Series(np.where((down>up)&(down>0),down,0.),index=d.index); a=atr(d,n).replace(0,np.nan); pdi=100*plus.ewm(alpha=1/n,adjust=False).mean()/a; mdi=100*minus.ewm(alpha=1/n,adjust=False).mean()/a; dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan); return dx.ewm(alpha=1/n,adjust=False).mean()
def enrich(d):
    d=d.copy()
    for n in [9,20,21,50,200]: d[f'ema{n}']=d.close.ewm(span=n,adjust=False).mean()
    d['atr']=atr(d); d['rsi']=rsi(d.close); d['vol_ma']=d.volume.rolling(20).mean(); d['adx']=adx(d)
    d['ema20_slope3']=d.ema20/d.ema20.shift(3)-1; d['body_atr']=(d.close-d.open).abs()/d.atr.replace(0,np.nan); d['volume_ratio']=d.volume/d.vol_ma.replace(0,np.nan)
    return d
def frames(c):
    d5=enrich(c.set_index('timestamp')); d15=enrich(closed_resample(c,'15min')); d1=enrich(closed_resample(c,'1h')); d4=enrich(closed_resample(c,'4h')); return d5,d15,d1,d4

def regime_snapshot(c):
    try:
        d5,d15,d1,d4=frames(c)
        if min(len(d1),len(d4))<200:return {'regime':'WARMUP','direction':'MIXED'}
        d1['er24']=(d1.close-d1.close.shift(24)).abs()/d1.close.diff().abs().rolling(24).sum().replace(0,np.nan)
        d1['atr_pct']=d1.atr/d1.close; d1['atr_med30d']=d1.atr_pct.rolling(24*30,min_periods=24*20).median(); d1['vol_ratio']=d1.atr_pct/d1.atr_med30d.replace(0,np.nan)
        a=d1.iloc[-1]; b=d4.iloc[-1]
        bull=a.ema20>a.ema50 and a.ema20_slope3>0 and b.ema20>b.ema50 and b.ema20_slope3>0
        bear=a.ema20<a.ema50 and a.ema20_slope3<0 and b.ema20<b.ema50 and b.ema20_slope3<0
        direction='BULL' if bull else 'BEAR' if bear else 'MIXED'; vr=float(a.vol_ratio) if finite(a.vol_ratio) else None; er=float(a.er24) if finite(a.er24) else None; ax=float(a.adx) if finite(a.adx) else None
        low=vr is not None and er is not None and ax is not None and vr<.90 and ax<20 and er<.18
        expansion=vr is not None and er is not None and ax is not None and vr>=1.15 and ax>=23 and er>=.22
        regime=('BULL_EXPANSION' if bull and expansion else 'BEAR_EXPANSION' if bear and expansion else 'CHOP_LOWVOL' if low else 'TRANSITION')
        return {'regime':regime,'direction':direction,'adx':ax,'er24':er,'vol_ratio':vr,'ema20_slope3_1h':float(a.ema20_slope3),'ema20_slope3_4h':float(b.ema20_slope3),'price':float(d5.close.iloc[-1])}
    except Exception as e:
        print('[WARN] regime',repr(e),flush=True); return {'regime':'WARMUP','direction':'MIXED'}

def router(key,side,snap):
    r=snap.get('regime'); direction=snap.get('direction'); ad=snap.get('adx') or 0; vr=snap.get('vol_ratio')
    if r=='WARMUP': return False,'REGIME_WARMUP'
    if r=='CHOP_LOWVOL': return False,'LOW_VOL_CHOP'
    if key=='SMC_SWEEP': return (r=='TRANSITION' or (r.endswith('EXPANSION') and ((direction=='BULL' and side=='LONG') or (direction=='BEAR' and side=='SHORT')))),'SMC_CONTEXT'
    if key=='EMA_SCALP': return (r=='TRANSITION' and ad>=18 and (vr is None or vr>=.90)),'EMA_TRANSITION'
    if key in {'MOMENTUM','BREAKOUT','TREND_PULLBACK'}:
        aligned=(direction=='BULL' and side=='LONG') or (direction=='BEAR' and side=='SHORT')
        return (r in {'TRANSITION','BULL_EXPANSION','BEAR_EXPANSION'} and aligned and ad>=20),'TREND_CONTEXT'
    if key=='MEAN_REVERSION': return False,'SHADOW_REBUILD'
    return False,'NO_ROUTE'

def features(key,side,reason,d5,d15,d1,d4,snap):
    c5,c15,c1,c4=d5.iloc[-1],d15.iloc[-1],d1.iloc[-1],d4.iloc[-1]
    return {'time':d5.index[-1].isoformat(),'strategy':key,'side':side or 'NONE','setup':reason,'regime':snap.get('regime'),'direction':snap.get('direction'),'price':float(c5.close),'rsi_5m':float(c5.rsi) if finite(c5.rsi) else None,'adx_1h':snap.get('adx'),'er24_1h':snap.get('er24'),'vol_ratio_1h':snap.get('vol_ratio'),'atr_pct_5m':float(c5.atr/c5.close) if finite(c5.atr) else None,'body_atr_5m':float(c5.body_atr) if finite(c5.body_atr) else None,'volume_ratio_5m':float(c5.volume_ratio) if finite(c5.volume_ratio) else None,'volume_ratio_15m':float(c15.volume_ratio) if finite(c15.volume_ratio) else None,'ema20_slope3_15m':float(c15.ema20_slope3) if finite(c15.ema20_slope3) else None,'ema20_slope3_1h':float(c1.ema20_slope3) if finite(c1.ema20_slope3) else None,'ema20_slope3_4h':float(c4.ema20_slope3) if finite(c4.ema20_slope3) else None}
DECISION_COLUMNS=[
    'time','strategy','side','setup','regime','direction','price','rsi_5m','adx_1h','er24_1h',
    'vol_ratio_1h','atr_pct_5m','body_atr_5m','volume_ratio_5m','volume_ratio_15m',
    'ema20_slope3_15m','ema20_slope3_1h','ema20_slope3_4h',
    'router_allowed','router_reason','decision','open_risk_pct'
]

def append_csv(path,row):
    exists=os.path.exists(path)
    pd.DataFrame([row]).to_csv(path,mode='a' if exists else 'w',header=not exists,index=False)

def normalize_decision(row):
    # One immutable schema for NO_SETUP / OPEN / SHADOW / every BLOCK state.
    clean={k:None for k in DECISION_COLUMNS}
    for k in DECISION_COLUMNS:
        if k in row: clean[k]=row.get(k)
    return clean

def append_decision(row):
    clean=normalize_decision(row)
    exists=os.path.exists(DECISIONS_FILE) and os.path.getsize(DECISIONS_FILE)>0
    pd.DataFrame([clean],columns=DECISION_COLUMNS).to_csv(
        DECISIONS_FILE,mode='a' if exists else 'w',header=not exists,index=False
    )

def repair_decisions_file():
    """Rotate an old malformed decisions CSV instead of silently losing it."""
    if not os.path.exists(DECISIONS_FILE) or os.path.getsize(DECISIONS_FILE)==0:return
    bad=False
    try:
        with open(DECISIONS_FILE,newline='',encoding='utf-8') as f:
            rows=list(csv.reader(f))
        if not rows:return
        if rows[0] != DECISION_COLUMNS: bad=True
        elif any(len(r)!=len(DECISION_COLUMNS) for r in rows[1:]): bad=True
    except Exception:
        bad=True
    if bad:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        backup=DECISIONS_FILE.replace('.csv',f'_legacy_{stamp}.csv')
        os.replace(DECISIONS_FILE,backup)
        print(f'[MIGRATION] malformed decisions CSV rotated to {backup}',flush=True)

def signal_for(key,c,snap):
    if len(c)<600:return None
    d5,d15,d1,d4=frames(c)
    if min(len(d15),len(d1),len(d4))<200:return None
    t=d5.index[-1]; c5,p5=d5.iloc[-1],d5.iloc[-2]; c15,p15=d15.iloc[-1],d15.iloc[-2]; c1=d1.iloc[-1]; c4=d4.iloc[-1]
    side=None; note='NO_SETUP'; dist=None

    if key=='SMC_SWEEP':
        hi=d5.high.shift(1).rolling(18).max().iloc[-1]; lo=d5.low.shift(1).rolling(18).min().iloc[-1]
        swept_low=c5.low<lo; reclaimed_low=c5.close>lo
        swept_high=c5.high>hi; rejected_high=c5.close<hi
        bull=swept_low and reclaimed_low and c5.close>c5.open and c5.rsi>=p5.rsi and c5.body_atr>=.12
        bear=swept_high and rejected_high and c5.close<c5.open and c5.rsi<=p5.rsi and c5.body_atr>=.12
        if bull:side,note='LONG','LOW_SWEEP_RECLAIM'
        elif bear:side,note='SHORT','HIGH_SWEEP_REJECT'
        elif not (swept_low or swept_high): note='WAIT_LIQUIDITY_SWEEP'
        elif swept_low and not reclaimed_low: note='WAIT_LOW_RECLAIM'
        elif swept_high and not rejected_high: note='WAIT_HIGH_REJECTION'
        else: note='WAIT_SWEEP_CONFIRMATION'
        if side:dist=max(float(c5.atr*STRATEGIES[key]['stop_atr']),abs(float(c5.close-c5.low)) if side=='LONG' else abs(float(c5.high-c5.close)))

    elif key=='EMA_SCALP':
        # A valid cross may have happened during one of the last three confirmed 5m bars.
        cross_up=((d5.ema9.shift(1)<=d5.ema21.shift(1))&(d5.ema9>d5.ema21)).tail(3).any()
        cross_dn=((d5.ema9.shift(1)>=d5.ema21.shift(1))&(d5.ema9<d5.ema21)).tail(3).any()
        trend_up=c15.ema20>c15.ema50 and c15.ema20_slope3>0
        trend_dn=c15.ema20<c15.ema50 and c15.ema20_slope3<0
        bull=cross_up and trend_up and c5.ema9>c5.ema21 and c5.rsi>=50 and c5.volume_ratio>=.70
        bear=cross_dn and trend_dn and c5.ema9<c5.ema21 and c5.rsi<=50 and c5.volume_ratio>=.70
        if bull:side,note='LONG','EMA9_21_CONFIRMED'
        elif bear:side,note='SHORT','EMA9_21_CONFIRMED'
        elif not (cross_up or cross_dn): note='WAIT_EMA_CROSS'
        elif cross_up and not trend_up: note='WAIT_15M_BULL_ALIGNMENT'
        elif cross_dn and not trend_dn: note='WAIT_15M_BEAR_ALIGNMENT'
        elif c5.volume_ratio<.70: note='WAIT_VOLUME'
        else: note='WAIT_EMA_CONFIRMATION'
        if side:dist=float(c5.atr*STRATEGIES[key]['stop_atr'])

    elif key=='MOMENTUM':
        # Look for a recent impulse, then require continuation instead of chasing the impulse candle itself.
        recent=d5.iloc[-4:-1]
        up_imp=((recent.body_atr>=.45)&(recent.volume_ratio>=1.05)&(recent.rsi>=55))
        dn_imp=((recent.body_atr>=.45)&(recent.volume_ratio>=1.05)&(recent.rsi<=45))
        impulse_up=bool(up_imp.any()); impulse_dn=bool(dn_imp.any())
        trend_up=c15.ema20>c15.ema50 and c15.ema20_slope3>0
        trend_dn=c15.ema20<c15.ema50 and c15.ema20_slope3<0
        bull=impulse_up and trend_up and c5.close>c5.ema9 and c5.close>p5.close and c5.rsi>=53 and c5.volume_ratio>=.75
        bear=impulse_dn and trend_dn and c5.close<c5.ema9 and c5.close<p5.close and c5.rsi<=47 and c5.volume_ratio>=.75
        if bull:side,note='LONG','RECENT_IMPULSE_CONTINUATION'
        elif bear:side,note='SHORT','RECENT_IMPULSE_CONTINUATION'
        elif not (impulse_up or impulse_dn): note='WAIT_MOMENTUM_IMPULSE'
        elif impulse_up and not trend_up: note='WAIT_15M_BULL_ALIGNMENT'
        elif impulse_dn and not trend_dn: note='WAIT_15M_BEAR_ALIGNMENT'
        elif c5.volume_ratio<.75: note='WAIT_VOLUME'
        else: note='WAIT_CONTINUATION_CONFIRMATION'
        if side:dist=float(c5.atr*STRATEGIES[key]['stop_atr'])

    elif key=='BREAKOUT':
        hi=d15.high.shift(1).rolling(8).max().iloc[-1]; lo=d15.low.shift(1).rolling(8).min().iloc[-1]
        trend_up=c1.ema20>c1.ema50 and c1.ema20_slope3>0
        trend_dn=c1.ema20<c1.ema50 and c1.ema20_slope3<0
        broke_up=c15.close>hi; broke_dn=c15.close<lo
        bull=trend_up and broke_up and c15.volume_ratio>=1.05 and c15.body_atr>=.30
        bear=trend_dn and broke_dn and c15.volume_ratio>=1.05 and c15.body_atr>=.30
        if bull:side,note='LONG','15M_CONFIRMED_BREAKOUT'
        elif bear:side,note='SHORT','15M_CONFIRMED_BREAKOUT'
        elif not (broke_up or broke_dn): note='WAIT_15M_BREAKOUT'
        elif broke_up and not trend_up: note='WAIT_1H_BULL_ALIGNMENT'
        elif broke_dn and not trend_dn: note='WAIT_1H_BEAR_ALIGNMENT'
        elif c15.volume_ratio<1.05: note='WAIT_BREAKOUT_VOLUME'
        else: note='WAIT_BREAKOUT_BODY'
        if side:dist=float(c15.atr*STRATEGIES[key]['stop_atr'])

    elif key=='TREND_PULLBACK':
        bulltrend=c1.ema20>c1.ema50 and c1.ema20_slope3>0 and c4.ema20>c4.ema50 and c4.ema20_slope3>0 and c1.adx>=20
        beartrend=c1.ema20<c1.ema50 and c1.ema20_slope3<0 and c4.ema20<c4.ema50 and c4.ema20_slope3<0 and c1.adx>=20
        bull=bulltrend and p15.low<=p15.ema20*1.002 and p15.close>=p15.ema50 and c15.close>c15.ema20 and c15.close>p15.close and c15.volume_ratio>=.75
        bear=beartrend and p15.high>=p15.ema20*.998 and p15.close<=p15.ema50 and c15.close<c15.ema20 and c15.close<p15.close and c15.volume_ratio>=.75
        if bull:side,note='LONG','HTF_PULLBACK_RESUME'
        elif bear:side,note='SHORT','HTF_PULLBACK_RESUME'
        elif not (bulltrend or beartrend): note='WAIT_HTF_TREND'
        else: note='WAIT_PULLBACK_RESUME'
        if side:dist=float(c15.atr*STRATEGIES[key]['stop_atr'])

    elif key=='MEAN_REVERSION':
        z=(c5.close-c5.ema20)/c5.atr if c5.atr else 0
        bull=z<-1.8 and c5.rsi<30 and c5.close>c5.open and c5.volume_ratio>=.70
        bear=z>1.8 and c5.rsi>70 and c5.close<c5.open and c5.volume_ratio>=.70
        if bull:side,note='LONG','EXTREME_SNAPBACK'
        elif bear:side,note='SHORT','EXTREME_SNAPBACK'
        elif abs(z)<1.8: note='WAIT_PRICE_EXTREME'
        elif 30<=c5.rsi<=70: note='WAIT_RSI_EXTREME'
        else: note='WAIT_REVERSAL_CANDLE'
        if side:dist=float(c5.atr*STRATEGIES[key]['stop_atr'])

    feat=features(key,side,note,d5,d15,d1,d4,snap)
    if not side:return {'signal':None,'time':t,'reason':note,'features':feat}
    allowed,why=router(key,side,snap); feat['router_allowed']=allowed; feat['router_reason']=why
    return {'signal':side,'time':t,'entry':float(c5.close),'stop_distance':float(dist),'reason':note,'allowed':allowed,'router_reason':why,'features':feat}

def is_cooldown(s,now=None):
    if not s.get('cooldown_until'):return False
    now=now or pd.Timestamp.now(tz='UTC'); until=pd.Timestamp(s['cooldown_until'])
    if now>=until:s['cooldown_until']=None;return False
    return True
def open_risk(state): return sum(RISK_PER_TRADE for s in state['strategies'].values() if s.get('position'))
def can_open(state): return open_risk(state)+RISK_PER_TRADE<=MAX_TOTAL_RISK+1e-12

def open_position(key,state,s,sig):
    entry,dist,side=sig['entry'],sig['stop_distance'],sig['signal']; cfg=STRATEGIES[key]; stop=entry-dist if side=='LONG' else entry+dist; target=entry+cfg['rr']*dist if side=='LONG' else entry-cfg['rr']*dist
    risk_eur=state['balance']*RISK_PER_TRADE; qty=risk_eur/dist; notional=qty*entry
    s['position']={'side':side,'entry_time':sig['time'].isoformat(),'entry_price':entry,'stop':stop,'target':target,'stop_distance':dist,'risk_eur':risk_eur,'qty':qty,'notional':notional,'setup':sig['reason'],'mfe_r':0.0,'mae_r':0.0}
    s['signals']+=1;s['last_signal']=f"{side} · {sig['reason']}";s['last_signal_time']=sig['time'].isoformat(); append_csv(FEATURES_FILE,{**sig['features'],'entry':entry,'stop':stop,'target':target,'stop_pct':dist/entry,'risk_eur':risk_eur,'notional':notional})
    print(f'[ENTRY][{key}] {side} {entry:.2f} stop={stop:.2f} tp={target:.2f}',flush=True)
def close_position(key,state,s,exit_price,exit_time,reason):
    p=s['position']; raw_r=(exit_price-p['entry_price'])/p['stop_distance'] if p['side']=='LONG' else (p['entry_price']-exit_price)/p['stop_distance']
    gross=p['risk_eur']*raw_r; fee=(p['notional'] + p['qty']*exit_price)*TAKER_FEE; slip=(p['notional'] + p['qty']*exit_price)*(SLIPPAGE_BPS/10000.0); pnl=gross-fee-slip; net_r=pnl/p['risk_eur'] if p['risk_eur'] else 0
    state['balance']+=pnl;state['peak_balance']=max(state['peak_balance'],state['balance']);state['max_drawdown']=min(state['max_drawdown'],state['balance']/state['peak_balance']-1);state['total_trades']+=1
    if pnl>0:state['winning_trades']+=1;state['gross_profit']+=pnl;s['loss_streak']=0
    else:
        state['losing_trades']+=1;state['gross_loss']+=abs(pnl);s['loss_streak']+=1
        if s['loss_streak']>=COOLDOWN_AFTER_LOSSES:s['cooldown_until']=(exit_time+timedelta(hours=COOLDOWN_HOURS)).isoformat()
    append_csv(TRADES_FILE,{'strategy':key,'entry_time':p['entry_time'],'exit_time':exit_time.isoformat(),'side':p['side'],'setup':p['setup'],'entry':p['entry_price'],'exit':exit_price,'stop':p['stop'],'target':p['target'],'raw_R':raw_r,'net_R':net_r,'reason':reason,'gross_pnl_eur':gross,'fees_eur':fee,'slippage_eur':slip,'pnl_eur':pnl,'balance':state['balance'],'MFE_R':p['mfe_r'],'MAE_R':p['mae_r']})
    s['position']=None; print(f'[EXIT][{key}] {reason} rawR={raw_r:.2f} netR={net_r:.2f} pnl=€{pnl:+.2f}',flush=True)
def check_position(key,state,s,candle):
    p=s.get('position')
    if not p:return
    h,l,cl,t=float(candle.high),float(candle.low),float(candle.close),candle.timestamp
    fav=(h-p['entry_price'])/p['stop_distance'] if p['side']=='LONG' else (p['entry_price']-l)/p['stop_distance']; adv=(l-p['entry_price'])/p['stop_distance'] if p['side']=='LONG' else (p['entry_price']-h)/p['stop_distance']; p['mfe_r']=max(p.get('mfe_r',0),fav);p['mae_r']=min(p.get('mae_r',0),adv)
    # Conservative same-candle ordering: stop wins if both stop and target are touched.
    if p['side']=='LONG':
        if l<=p['stop']:return close_position(key,state,s,p['stop'],t,'STOP')
        if h>=p['target']:return close_position(key,state,s,p['target'],t,'TAKE_PROFIT')
    else:
        if h>=p['stop']:return close_position(key,state,s,p['stop'],t,'STOP')
        if l<=p['target']:return close_position(key,state,s,p['target'],t,'TAKE_PROFIT')
    if t-pd.Timestamp(p['entry_time'])>=pd.Timedelta(hours=STRATEGIES[key]['max_hours']):close_position(key,state,s,cl,t,'TIME_EXIT')

def stats(state):
    t=state['total_trades']; w=state['winning_trades']; pf=state['gross_profit']/state['gross_loss'] if state['gross_loss'] else (999 if state['gross_profit'] else 0)
    return {'balance':state['balance'],'peak_balance':state.get('peak_balance',state['balance']),'return_pct':(state['balance']/START_BALANCE-1)*100,'net_pnl':state['balance']-START_BALANCE,'net_r':(state['balance']-START_BALANCE)/(START_BALANCE*RISK_PER_TRADE) if RISK_PER_TRADE else 0,'trades':t,'wins':w,'losses':state.get('losing_trades',0),'winrate':100*w/t if t else 0,'pf':pf,'max_dd':100*state['max_drawdown'],'open_risk_pct':100*open_risk(state)}


def _read_csv_records(path,n=50):
    try:
        if not os.path.exists(path): return []
        d=pd.read_csv(path).replace({np.nan:None})
        return d.tail(n).iloc[::-1].to_dict('records')
    except Exception:return []

def strategy_dashboard(state):
    trades=pd.DataFrame(_read_csv_records(TRADES_FILE,100000))
    out=[]
    for key,cfg in STRATEGIES.items():
        st=state['strategies'][key]
        d=trades[trades.strategy==key].copy() if len(trades) and 'strategy' in trades else pd.DataFrame()
        n=len(d)
        pnl_series=pd.to_numeric(d['pnl_eur'],errors='coerce').fillna(0) if n else pd.Series(dtype=float)
        wins=int((pnl_series>0).sum()) if n else 0
        pnl=float(pnl_series.sum()) if n else 0.0
        nr=float(pd.to_numeric(d['net_R'],errors='coerce').fillna(0).sum()) if n else 0.0
        gp=float(pnl_series[pnl_series>0].sum()) if n else 0.0
        gl=abs(float(pnl_series[pnl_series<0].sum())) if n else 0.0
        pf=gp/gl if gl else (999 if gp else 0)
        out.append({'key':key,'label':cfg['label'],'mode':'LIVE' if cfg['enabled'] else 'SHADOW',
                    'trades':n,'winrate':100*wins/n if n else 0,'pnl':pnl,'net_r':nr,'pf':pf,
                    'signals':st.get('signals',0),'blocked':st.get('blocked',0),
                    'loss_streak':st.get('loss_streak',0),'cooldown_until':st.get('cooldown_until'),
                    'last_signal':st.get('last_signal','—')})
    return out

def open_positions_dashboard(state,last_price):
    out=[]
    for key,st in state['strategies'].items():
        p=st.get('position')
        if not p: continue
        cur_r=((last_price-p['entry_price'])/p['stop_distance']) if p['side']=='LONG' else ((p['entry_price']-last_price)/p['stop_distance'])
        out.append({**p,'strategy':key,'current_price':last_price,'current_r':cur_r})
    return out

def regime_performance():
    try:
        t=pd.read_csv(TRADES_FILE); f=pd.read_csv(FEATURES_FILE)
        if not len(t) or not len(f) or 'regime' not in f:return []
        m=t.merge(f[['strategy','time','regime']],left_on=['strategy','entry_time'],right_on=['strategy','time'],how='left')
        out=[]
        for rg,d in m.groupby('regime',dropna=True):
            pnl=pd.to_numeric(d.pnl_eur,errors='coerce').fillna(0)
            nr=pd.to_numeric(d.net_R,errors='coerce').fillna(0)
            out.append({'regime':rg,'trades':len(d),'winrate':100*(pnl>0).mean(),'pnl':float(pnl.sum()),'net_r':float(nr.sum())})
        return sorted(out,key=lambda x:x['trades'],reverse=True)
    except Exception:return []
DASH='''<!doctype html><html lang="nl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="20"><title>BTC V1 Control Center</title><style>
:root{--bg:#0b0e13;--card:#151922;--card2:#10141c;--line:#2a3140;--text:#f5f7fb;--muted:#8e98aa;--green:#55d68b;--red:#ff6b72;--amber:#f3c969}*{box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--text);margin:0;padding:14px}.w{max-width:1250px;margin:auto}.top{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:8px 2px 16px}.title{font-size:34px;font-weight:900}.sub,.muted{color:var(--muted)}.running{font-size:12px;padding:6px 9px;border-radius:999px;background:#153222;color:var(--green);font-weight:800}.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:9px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:13px}.label{font-size:11px;color:var(--muted);text-transform:uppercase}.value{font-size:24px;font-weight:850;margin-top:4px}.mini{font-size:12px;margin-top:3px}.pos{color:var(--green)!important}.neg{color:var(--red)!important}.amber{color:var(--amber)!important}.section{margin-top:10px}.section h2{font-size:16px;margin:0 0 10px}.market{display:grid;grid-template-columns:1.4fr repeat(4,1fr);gap:9px}.tag{display:inline-block;padding:4px 7px;border-radius:999px;font-size:10px}.live{background:#153222;color:var(--green)}.shadow{background:#332d18;color:var(--amber)}.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch}table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}td,th{padding:9px 8px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--muted);font-size:10px;text-transform:uppercase}.strategy{font-weight:750}.pill{padding:3px 6px;border-radius:6px;background:#242b39;font-size:10px}.two{display:grid;grid-template-columns:1.35fr 1fr;gap:10px}.empty{padding:18px;text-align:center;color:var(--muted);background:var(--card2);border-radius:10px}.downloads{display:flex;gap:8px;flex-wrap:wrap}.btn{color:var(--text);text-decoration:none;background:#242b39;border:1px solid #343d4f;border-radius:9px;padding:8px 10px;font-size:12px}.small{font-size:10px}
@media(max-width:900px){.grid{grid-template-columns:repeat(3,1fr)}.market{grid-template-columns:repeat(3,1fr)}.two{grid-template-columns:1fr}}@media(max-width:520px){body{padding:10px}.grid{grid-template-columns:repeat(2,1fr)}.market{grid-template-columns:repeat(2,1fr)}.market>div:first-child{grid-column:span 2}.card{padding:11px}.value{font-size:21px}}
</style></head><body><div class="w"><div class="top"><div><div class="title">BTC V1</div><div class="sub">Control Center · paper trading · refresh 20s</div></div><div class="running">● RUNNING</div></div>
<div class="grid">
<div class="card"><div class="label">Balance</div><div class="value {{'pos' if s.return_pct>=0 else 'neg'}}">€{{'%.2f'|format(s.balance)}}</div><div class="mini">{{'%+.2f'|format(s.return_pct)}}%</div></div>
<div class="card"><div class="label">Net P/L</div><div class="value {{'pos' if s.net_pnl>=0 else 'neg'}}">€{{'%+.2f'|format(s.net_pnl)}}</div><div class="mini">{{'%+.2f'|format(s.net_r)}}R totaal</div></div>
<div class="card"><div class="label">Trades</div><div class="value">{{s.trades}}</div><div class="mini">{{s.wins}}W · {{s.losses}}L</div></div>
<div class="card"><div class="label">Winrate</div><div class="value">{{'%.1f'|format(s.winrate)}}%</div><div class="mini">PF {{'%.2f'|format(s.pf) if s.pf<900 else '∞'}}</div></div>
<div class="card"><div class="label">Max DD</div><div class="value {{'neg' if s.max_dd<0 else ''}}">{{'%.2f'|format(s.max_dd)}}%</div><div class="mini">Peak €{{'%.2f'|format(s.peak_balance)}}</div></div>
<div class="card"><div class="label">Open risk</div><div class="value">{{'%.2f'|format(s.open_risk_pct)}}%</div><div class="mini">cap {{'%.2f'|format(maxrisk)}}%</div></div></div>
<div class="section market"><div class="card"><div class="label">Market regime</div><div class="value">{{r.regime}}</div><div class="mini">{{r.direction}} · BTC ${{'{:,.0f}'.format(r.price) if r.price else '—'}}</div></div><div class="card"><div class="label">ADX 1H</div><div class="value">{{'%.1f'|format(r.adx) if r.adx is not none else '—'}}</div></div><div class="card"><div class="label">ER24 1H</div><div class="value">{{'%.3f'|format(r.er24) if r.er24 is not none else '—'}}</div></div><div class="card"><div class="label">Vol ratio</div><div class="value">{{'%.2f'|format(r.vol_ratio) if r.vol_ratio is not none else '—'}}</div></div><div class="card"><div class="label">Open positions</div><div class="value">{{openpos|length}}</div></div></div>
<div class="card section"><h2>Engine monitor</h2><div class="tablewrap"><table><tr><th>Engine</th><th>Mode</th><th>Trades</th><th>WR</th><th>Net R</th><th>P/L</th><th>PF</th><th>Signals</th><th>Blocked</th><th>Loss streak</th><th>Laatste signal</th></tr>{% for x in strat %}<tr><td><div class="strategy">{{x.label}}</div><div class="muted small">{{x.key}}</div></td><td><span class="tag {{'live' if x.mode=='LIVE' else 'shadow'}}">{{x.mode}}</span></td><td>{{x.trades}}</td><td>{{'%.1f'|format(x.winrate)}}%</td><td class="{{'pos' if x.net_r>0 else 'neg' if x.net_r<0 else ''}}">{{'%+.2f'|format(x.net_r)}}</td><td class="{{'pos' if x.pnl>0 else 'neg' if x.pnl<0 else ''}}">€{{'%+.2f'|format(x.pnl)}}</td><td>{{'%.2f'|format(x.pf) if x.pf<900 else '∞'}}</td><td>{{x.signals}}</td><td>{{x.blocked}}</td><td>{{x.loss_streak}}</td><td>{{x.last_signal}}{% if x.cooldown_until %}<div class="amber small">Cooldown → {{x.cooldown_until[11:16]}}</div>{% endif %}</td></tr>{% endfor %}</table></div></div>
<div class="card section"><h2>Open trades</h2>{% if openpos %}<div class="tablewrap"><table><tr><th>Strategy</th><th>Side</th><th>Entry</th><th>Current</th><th>SL</th><th>TP</th><th>Current R</th><th>MFE</th><th>MAE</th><th>Risk €</th></tr>{% for p in openpos %}<tr><td class="strategy">{{p.strategy}}</td><td>{{p.side}}</td><td>${{'{:,.0f}'.format(p.entry_price)}}</td><td>${{'{:,.0f}'.format(p.current_price)}}</td><td>${{'{:,.0f}'.format(p.stop)}}</td><td>${{'{:,.0f}'.format(p.target)}}</td><td class="{{'pos' if p.current_r>=0 else 'neg'}}">{{'%+.2f'|format(p.current_r)}}R</td><td class="pos">{{'%.2f'|format(p.mfe_r)}}R</td><td class="neg">{{'%.2f'|format(p.mae_r)}}R</td><td>€{{'%.2f'|format(p.risk_eur)}}</td></tr>{% endfor %}</table></div>{% else %}<div class="empty">Geen open trades — V1 wacht op een geldige setup.</div>{% endif %}</div>
<div class="two section"><div class="card"><h2>Recente decisions / signals</h2>{% if decisions %}<div class="tablewrap"><table><tr><th>Tijd</th><th>Strategy</th><th>Side</th><th>Decision</th><th>Setup</th><th>Regime</th><th>ADX</th></tr>{% for d in decisions %}<tr><td>{{d.time[11:16] if d.time else '—'}}</td><td class="strategy">{{d.strategy}}</td><td>{{d.side}}</td><td><span class="pill {{'pos' if d.decision=='OPEN' else 'amber' if d.decision=='SHADOW' else ''}}">{{d.decision}}</span></td><td>{{d.setup}}</td><td>{{d.regime}}</td><td>{{'%.1f'|format(d.adx_1h) if d.adx_1h is not none else '—'}}</td></tr>{% endfor %}</table></div>{% else %}<div class="empty">Nog geen signalen gelogd.</div>{% endif %}</div><div class="card"><h2>Performance per regime</h2>{% if regimes %}<div class="tablewrap"><table><tr><th>Regime</th><th>Trades</th><th>WR</th><th>Net R</th><th>P/L</th></tr>{% for x in regimes %}<tr><td class="strategy">{{x.regime}}</td><td>{{x.trades}}</td><td>{{'%.1f'|format(x.winrate)}}%</td><td class="{{'pos' if x.net_r>0 else 'neg' if x.net_r<0 else ''}}">{{'%+.2f'|format(x.net_r)}}</td><td class="{{'pos' if x.pnl>0 else 'neg' if x.pnl<0 else ''}}">€{{'%+.2f'|format(x.pnl)}}</td></tr>{% endfor %}</table></div>{% else %}<div class="empty">Regime-statistieken verschijnen na de eerste gesloten trades.</div>{% endif %}</div></div>
<div class="card section"><h2>Trade history</h2>{% if recent %}<div class="tablewrap"><table><tr><th>Exit</th><th>Strategy</th><th>Side</th><th>Setup</th><th>Entry</th><th>Exit</th><th>Reason</th><th>Gross R</th><th>Costs R</th><th>Net R</th><th>MFE</th><th>MAE</th><th>Costs €</th><th>P/L</th><th>Balance</th></tr>{% for t in recent %}<tr><td>{{t.exit_time[5:16]|replace('T',' ')}}</td><td class="strategy">{{t.strategy}}</td><td>{{t.side}}</td><td>{{t.setup}}</td><td>${{'{:,.0f}'.format(t.entry)}}</td><td>${{'{:,.0f}'.format(t.exit)}}</td><td>{{t.reason}}</td><td>{{'%+.2f'|format(t.raw_R)}}R</td><td class="amber">-{{'%.2f'|format(t.costs_R)}}R</td><td class="{{'pos' if t.net_R>=0 else 'neg'}}">{{'%+.2f'|format(t.net_R)}}R</td><td class="pos">{{'%.2f'|format(t.MFE_R)}}R</td><td class="neg">{{'%.2f'|format(t.MAE_R)}}R</td><td>€{{'%.2f'|format(t.costs_eur)}}</td><td class="{{'pos' if t.pnl_eur>=0 else 'neg'}}">€{{'%+.2f'|format(t.pnl_eur)}}</td><td>€{{'%.2f'|format(t.balance)}}</td></tr>{% endfor %}</table></div>{% else %}<div class="empty">Nog geen gesloten trades.</div>{% endif %}</div>
<div class="card section"><h2>Data & exports</h2><div class="downloads"><a class="btn" href="/download/trades">↓ Trades CSV</a><a class="btn" href="/download/features">↓ Entry features CSV</a><a class="btn" href="/download/decisions">↓ Decisions CSV</a><a class="btn" href="/api/status">API status</a></div></div></div></body></html>'''
repair_decisions_file()

app=Flask(__name__)
def recent(n=30):
    rows=_read_csv_records(TRADES_FILE,n)
    for t in rows:
        try:
            risk=abs(float(t.get('gross_pnl_eur',0))/float(t.get('raw_R',0))) if float(t.get('raw_R',0)) else 0
            costs=float(t.get('fees_eur',0) or 0)+float(t.get('slippage_eur',0) or 0)
            t['costs_eur']=costs
            t['costs_R']=costs/risk if risk else 0
        except Exception:
            t['costs_eur']=0;t['costs_R']=0
    return rows
@app.get('/')
def dashboard():
    st=load_state(); c=load_candles(); r=regime_snapshot(c); price=float(c.close.iloc[-1]) if len(c) else 0
    return render_template_string(DASH,s=stats(st),r=r,strat=strategy_dashboard(st),openpos=open_positions_dashboard(st,price),decisions=_read_csv_records(DECISIONS_FILE,36),regimes=regime_performance(),recent=recent(),maxrisk=MAX_TOTAL_RISK*100)
@app.get('/api/status')
def status():
    st=load_state(); c=load_candles(); return jsonify({'version':'BTC-V1.3','stats':stats(st),'regime':regime_snapshot(c),'strategies':STRATEGIES})
def dl(path,name):
    if not os.path.exists(path):return {'error':'Nog geen bestand.'},404
    return send_file(path,mimetype='text/csv',as_attachment=True,download_name=name)
@app.get('/download/trades')
def d1():return dl(TRADES_FILE,'btc_v1_trades.csv')
@app.get('/download/features')
def d2():return dl(FEATURES_FILE,'btc_v1_entry_features.csv')
@app.get('/download/decisions')
def d3():return dl(DECISIONS_FILE,'btc_v1_decisions.csv')
@app.get('/health')
def health():return {'status':'ok','version':'BTC-V1.3'},200
def run_dashboard():app.run(host='0.0.0.0',port=int(os.getenv('PORT','8080')),threaded=True,use_reloader=False)

def main():
    threading.Thread(target=run_dashboard,daemon=True).start(); print('BTC V1 — PAPER ONLY',flush=True)
    state=load_state(); candles=update_candles(load_candles())
    while True:
        try:
            candles=update_candles(candles); newest=candles.timestamp.iloc[-1]; prev=state.get('last_processed_5m')
            if prev is None or newest>pd.Timestamp(prev):
                new_rows=candles if prev is None else candles[candles.timestamp>pd.Timestamp(prev)]
                for _,bar in new_rows.iterrows():
                    for key in STRATEGIES:check_position(key,state,state['strategies'][key],bar)
                    state['last_processed_5m']=bar.timestamp.isoformat()
                snap=regime_snapshot(candles)
                for key,cfg in STRATEGIES.items():
                    s=state['strategies'][key];s['scans']+=1;sig=signal_for(key,candles,snap)
                    if sig:
                        if sig.get('signal'):
                            s['last_signal']=f"{sig['signal']} · {sig['reason']}";s['last_signal_time']=sig['time'].isoformat()
                            decision='SHADOW' if not cfg['enabled'] else 'ALLOW' if sig.get('allowed') else 'BLOCK_REGIME'
                            if cfg['enabled'] and not sig.get('allowed'): s['blocked']+=1
                            if cfg['enabled'] and sig.get('allowed') and s.get('position') is None and not is_cooldown(s):
                                if can_open(state):open_position(key,state,s,sig);decision='OPEN'
                                else:s['blocked']+=1;decision='BLOCK_RISK_CAP'
                            elif cfg['enabled'] and sig.get('allowed') and s.get('position') is not None:
                                decision='BLOCK_ALREADY_OPEN'; s['blocked']+=1
                            elif cfg['enabled'] and sig.get('allowed') and is_cooldown(s):
                                decision='BLOCK_COOLDOWN'; s['blocked']+=1
                            append_decision({**sig['features'],'decision':decision,'router_reason':sig.get('router_reason'),'open_risk_pct':100*open_risk(state)})
                        else:
                            # Diagnostic heartbeat: every confirmed 5m scan now records why each engine did not fire.
                            append_decision({**sig['features'],'decision':'NO_SETUP','router_reason':'NOT_EVALUATED_NO_SIGNAL','open_risk_pct':100*open_risk(state)})
                save_state(state);print(f"[STATUS] BTC={candles.close.iloc[-1]:,.0f} regime={snap.get('regime')} balance=€{state['balance']:.2f} trades={state['total_trades']}",flush=True)
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:save_state(state);break
        except Exception as e:print('[ERROR]',repr(e),flush=True);save_state(state);time.sleep(60)
if __name__=='__main__':main()
