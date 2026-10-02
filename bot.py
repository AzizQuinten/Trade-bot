import os, json, time, threading, csv, hashlib
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, render_template_string, send_file
import requests
import pandas as pd
import numpy as np

# BTC V1.7.2 RESEARCH — independent paper account per strategy
INST_ID = os.getenv('INST_ID', 'BTC-USDT')
START_BALANCE = float(os.getenv('START_BALANCE', '500'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.01'))
TAKER_FEE = float(os.getenv('TAKER_FEE', '0.00035'))
SLIPPAGE_BPS = float(os.getenv('SLIPPAGE_BPS', '1.0'))
POLL_SECONDS = int(os.getenv('POLL_SECONDS', '20'))
BOOTSTRAP_DAYS = int(os.getenv('BOOTSTRAP_DAYS', '90'))
RUN_MODE = os.getenv('RUN_MODE','RESEARCH').upper()
RESEARCH_MODE = RUN_MODE == 'RESEARCH'
RESEARCH_EMERGENCY_DD = float(os.getenv('RESEARCH_EMERGENCY_DD','0.25'))
# V1.7: no shared portfolio cap. Each strategy owns an independent paper account.
COOLDOWN_AFTER_LOSSES = 3
COOLDOWN_HOURS = 8
MIN_SCORE = float(os.getenv('MIN_SCORE','65'))
MAX_COST_R = float(os.getenv('MAX_COST_R','0.35'))
DAILY_LOSS_LIMIT = float(os.getenv('DAILY_LOSS_LIMIT','0.03'))
DRAWDOWN_SOFT = float(os.getenv('DRAWDOWN_SOFT','0.06'))
DRAWDOWN_HARD = float(os.getenv('DRAWDOWN_HARD','0.09'))
GLOBAL_CONSEC_LOSS_LIMIT = int(os.getenv('GLOBAL_CONSEC_LOSS_LIMIT','4'))
BE_TRIGGER_R = float(os.getenv('BE_TRIGGER_R','1.00'))
LOCK_TRIGGER_R = float(os.getenv('LOCK_TRIGGER_R','1.25'))
LOCK_PROFIT_R = float(os.getenv('LOCK_PROFIT_R','0.60'))
TRAIL_TRIGGER_R = float(os.getenv('TRAIL_TRIGGER_R','1.75'))
TRAIL_GAP_R = float(os.getenv('TRAIL_GAP_R','0.85'))
PROFIT_ZONE_R = float(os.getenv('PROFIT_ZONE_R','0.55'))
DYNAMIC_PROTECT_R = float(os.getenv('DYNAMIC_PROTECT_R','0.75'))
DETERIORATION_GIVEBACK_R = float(os.getenv('DETERIORATION_GIVEBACK_R','0.45'))
DETERIORATION_CLOSE_R = float(os.getenv('DETERIORATION_CLOSE_R','0.35'))
OKX_URL = os.getenv('OKX_PUBLIC_BASE', 'https://www.okx.com')
HISTORY_ENDPOINT = '/api/v5/market/history-candles'
LIVE_ENDPOINT = '/api/v5/market/candles'
DATA_DIR = os.getenv('DATA_DIR', '/data')
try: os.makedirs(DATA_DIR, exist_ok=True)
except PermissionError:
    DATA_DIR = './data'; os.makedirs(DATA_DIR, exist_ok=True)
STATE_FILE = os.path.join(DATA_DIR, 'btc_v172_state.json')
TRADES_FILE = os.path.join(DATA_DIR, 'btc_v172_trades.csv')
FEATURES_FILE = os.path.join(DATA_DIR, 'btc_v172_entries.csv')
DECISIONS_FILE = os.path.join(DATA_DIR, 'btc_v172_decisions.csv')
CANDLES_FILE = os.path.join(DATA_DIR, 'btc_v172_candles_5m.csv')

STRATEGIES = {
 'SMC_SWEEP': {'label':'Liquidity Sweep / Reversal','rr':2.5,'stop_atr':1.35,'max_hours':6,'enabled':True},
 'EMA_SCALP': {'label':'Trend Pullback / EMA Retest','rr':2.0,'stop_atr':1.25,'max_hours':4,'enabled':True},
 'MOMENTUM': {'label':'Momentum Expansion','rr':2.2,'stop_atr':1.40,'max_hours':5,'enabled':True},
 'BREAKOUT': {'label':'Breakout / Retest','rr':2.5,'stop_atr':1.70,'max_hours':8,'enabled':True},
 'TREND_PULLBACK': {'label':'HTF Trend Continuation','rr':2.5,'stop_atr':1.65,'max_hours':8,'enabled':True},
 'MEAN_REVERSION': {'label':'Range Mean Reversion','rr':1.8,'stop_atr':1.45,'max_hours':5,'enabled':True},
}


session=requests.Session(); session.headers.update({'User-Agent':'BTC-V1/1.7.2-RESEARCH'})
state_lock=threading.RLock()
def utc_now(): return datetime.now(timezone.utc)
def finite(x):
    try: return bool(np.isfinite(float(x)))
    except Exception: return False

def strategy_default():
    return {'position':None,'balance':START_BALANCE,'peak_balance':START_BALANCE,'max_drawdown':0.0,'risk_day':None,'day_start_balance':START_BALANCE,'loss_streak':0,'cooldown_until':None,'scans':0,'signals':0,'blocked':0,'last_signal':'—','last_signal_time':None}
def default_state():
    return {'version':'BTC-V1.7.2-RESEARCH','created':utc_now().isoformat(),'last_processed_5m':None,'balance':START_BALANCE,'peak_balance':START_BALANCE,'max_drawdown':0.0,'total_trades':0,'winning_trades':0,'losing_trades':0,'gross_profit':0.0,'gross_loss':0.0,'global_loss_streak':0,'risk_day':None,'day_start_balance':START_BALANCE,'shadow_candidates':{},'strategies':{k:strategy_default() for k in STRATEGIES}}
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
    """Orthogonal regime dimensions: direction, strength, volatility and market state."""
    try:
        d5,d15,d1,d4=frames(c)
        if min(len(d1),len(d4))<200:return {'regime':'WARMUP','direction':'NEUTRAL','strength':'WEAK','volatility':'UNKNOWN','market_state':'WARMUP'}
        d1=d1.copy()
        d1['er24']=(d1.close-d1.close.shift(24)).abs()/d1.close.diff().abs().rolling(24).sum().replace(0,np.nan)
        d1['atr_pct']=d1.atr/d1.close
        d1['atr_med30d']=d1.atr_pct.rolling(24*30,min_periods=24*20).median()
        d1['vol_ratio']=d1.atr_pct/d1.atr_med30d.replace(0,np.nan)
        x=d1.iloc[-1]; h=d4.iloc[-1]
        ax=float(x.adx) if finite(x.adx) else 0.0
        er=float(x.er24) if finite(x.er24) else 0.0
        vr=float(x.vol_ratio) if finite(x.vol_ratio) else 1.0
        # Direction does not require perfect 1H+4H agreement anymore.
        bull_pts=int(x.ema20>x.ema50)+int(x.ema20_slope3>0)+int(h.ema20>h.ema50)+int(h.ema20_slope3>0)
        bear_pts=int(x.ema20<x.ema50)+int(x.ema20_slope3<0)+int(h.ema20<h.ema50)+int(h.ema20_slope3<0)
        direction='BULL' if bull_pts>=3 and bull_pts>bear_pts else 'BEAR' if bear_pts>=3 and bear_pts>bull_pts else 'NEUTRAL'
        strength='STRONG' if ax>=25 and er>=.22 else 'MEDIUM' if ax>=18 and er>=.10 else 'WEAK'
        volatility='HIGH' if vr>=1.25 else 'LOW' if vr<=.80 else 'NORMAL'
        expansion=(ax>=22 and er>=.18 and vr>=1.05)
        range_like=(ax<18 and er<.10)
        contraction=(vr<.85 and ax<20)
        if expansion: market_state='EXPANSION'
        elif range_like: market_state='RANGE_CHOP'
        elif contraction: market_state='CONTRACTION'
        elif direction!='NEUTRAL' and strength in {'MEDIUM','STRONG'}: market_state='TREND'
        else: market_state='TRANSITION'
        regime=f"{direction}_{strength}_{volatility}_{market_state}"
        return {'regime':regime,'direction':direction,'strength':strength,'volatility':volatility,'market_state':market_state,
                'adx':ax,'er24':er,'vol_ratio':vr,'ema20_slope3_1h':float(x.ema20_slope3),
                'ema20_slope3_4h':float(h.ema20_slope3),'price':float(d5.close.iloc[-1])}
    except Exception as e:
        print('[WARN] regime',repr(e),flush=True)
        return {'regime':'WARMUP','direction':'NEUTRAL','strength':'WEAK','volatility':'UNKNOWN','market_state':'WARMUP'}

def router(key,side,snap):
    state=snap.get('market_state'); direction=snap.get('direction'); strength=snap.get('strength')
    if state=='WARMUP': return False,'REGIME_WARMUP'
    if key in {'BREAKOUT','MOMENTUM'} and state in {'RANGE_CHOP','CONTRACTION'}: return False,'WRONG_REGIME_TREND_ENGINE'
    if key in {'EMA_SCALP','TREND_PULLBACK'} and state=='RANGE_CHOP': return False,'WRONG_REGIME_CONTINUATION'
    if key=='SMC_SWEEP' and state=='EXPANSION' and strength=='STRONG': return False,'NO_FADE_STRONG_EXPANSION'
    if key=='MEAN_REVERSION':
        if state!='RANGE_CHOP': return False,'MEAN_REVERSION_RANGE_ONLY'
    if direction=='BULL' and side=='SHORT' and strength=='STRONG': return False,'HOSTILE_HTF_DIRECTION'
    if direction=='BEAR' and side=='LONG' and strength=='STRONG': return False,'HOSTILE_HTF_DIRECTION'
    return True,'CONTEXT_OK'

def base_features(key,side,setup,d5,d15,d1,d4,snap):
    c5,c15=d5.iloc[-1],d15.iloc[-1]
    return {'time':d5.index[-1].isoformat(),'strategy':key,'side':side or 'NONE','setup':setup,
            'regime':snap.get('regime'),'direction':snap.get('direction'),'strength':snap.get('strength'),
            'volatility':snap.get('volatility'),'market_state':snap.get('market_state'),'price':float(c5.close),
            'rsi_5m':float(c5.rsi) if finite(c5.rsi) else None,'adx_1h':snap.get('adx'),'er24_1h':snap.get('er24'),
            'vol_ratio_1h':snap.get('vol_ratio'),'atr_pct_5m':float(c5.atr/c5.close) if finite(c5.atr) else None,
            'body_atr_5m':float(c5.body_atr) if finite(c5.body_atr) else None,
            'volume_ratio_5m':float(c5.volume_ratio) if finite(c5.volume_ratio) else None,
            'volume_ratio_15m':float(c15.volume_ratio) if finite(c15.volume_ratio) else None,
            'ema20_slope3_15m':float(c15.ema20_slope3) if finite(c15.ema20_slope3) else None,
            'ema20_slope3_1h':snap.get('ema20_slope3_1h'),'ema20_slope3_4h':snap.get('ema20_slope3_4h'),
            'trend_aligned': (side=='LONG' and snap.get('direction')=='BULL') or (side=='SHORT' and snap.get('direction')=='BEAR') if side else None,
            'countertrend': (side=='LONG' and snap.get('direction')=='BEAR') or (side=='SHORT' and snap.get('direction')=='BULL') if side else None}

def score_setup(key,side,trigger,ctx,snap):
    """Transparent 0-100 score. Components are logged; weights are hypotheses, not fitted to the 8 legacy trades."""
    direction=snap.get('direction'); strength=snap.get('strength'); state=snap.get('market_state')
    trend=20 if direction==side.replace('LONG','BULL').replace('SHORT','BEAR') else 10 if direction=='NEUTRAL' else 2
    regime=20 if ((key in {'BREAKOUT','MOMENTUM'} and state in {'EXPANSION','TREND'}) or
                  (key in {'EMA_SCALP','TREND_PULLBACK'} and state in {'TREND','TRANSITION','EXPANSION'}) or
                  (key=='SMC_SWEEP' and state in {'TRANSITION','RANGE_CHOP','CONTRACTION'}) or
                  (key=='MEAN_REVERSION' and state=='RANGE_CHOP')) else 7
    trigger_score=min(25,max(0,float(trigger)))
    momentum=min(15,max(0,float(ctx.get('momentum',7))))
    volume=min(10,max(0,float(ctx.get('volume',5))))
    structure=min(10,max(0,float(ctx.get('structure',5))))
    total=trend+regime+trigger_score+momentum+volume+structure
    return {'score_total':float(total),'score_trend':float(trend),'score_regime':float(regime),
            'score_trigger':float(trigger_score),'score_momentum':float(momentum),
            'score_volume':float(volume),'score_structure':float(structure)}

def detect_smc(d5,d15,d1,d4,snap):
    c,p=d5.iloc[-1],d5.iloc[-2]; hi=d5.high.shift(1).rolling(18).max().iloc[-1]; lo=d5.low.shift(1).rolling(18).min().iloc[-1]
    if c.low<lo and c.close>lo and c.close>c.open:
        depth=(lo-c.low)/c.atr if c.atr else 0; return 'LONG','LOW_SWEEP_RECLAIM',max(float(c.atr*1.35),float(c.close-c.low)),20+min(5,depth*10)
    if c.high>hi and c.close<hi and c.close<c.open:
        depth=(c.high-hi)/c.atr if c.atr else 0; return 'SHORT','HIGH_SWEEP_REJECT',max(float(c.atr*1.35),float(c.high-c.close)),20+min(5,depth*10)
    return None,'WAIT_LIQUIDITY_SWEEP',None,0

def detect_ema(d5,d15,d1,d4,snap):
    c,p=d5.iloc[-1],d5.iloc[-2]
    long_retest=p.low<=p.ema21*1.0015 and c.close>c.ema9>c.ema21 and c.close>p.close
    short_retest=p.high>=p.ema21*.9985 and c.close<c.ema9<c.ema21 and c.close<p.close
    if long_retest and c.rsi>=48:return 'LONG','EMA_PULLBACK_RECLAIM',float(c.atr*1.25),22
    if short_retest and c.rsi<=52:return 'SHORT','EMA_PULLBACK_REJECT',float(c.atr*1.25),22
    return None,'WAIT_EMA_RETEST',None,0

def detect_momentum(d5,d15,d1,d4,snap):
    c,p=d5.iloc[-1],d5.iloc[-2]; recent=d5.iloc[-7:-1]
    up=bool(((recent.body_atr>=.40)&(recent.volume_ratio>=.95)&(recent.rsi>=54)).any())
    dn=bool(((recent.body_atr>=.40)&(recent.volume_ratio>=.95)&(recent.rsi<=46)).any())
    if up and c.close>c.ema9 and c.close>p.close and c.rsi>=51:return 'LONG','IMPULSE_CONTINUATION',float(c.atr*1.40),21
    if dn and c.close<c.ema9 and c.close<p.close and c.rsi<=49:return 'SHORT','IMPULSE_CONTINUATION',float(c.atr*1.40),21
    return None,'WAIT_MOMENTUM',None,0

def detect_breakout(d5,d15,d1,d4,snap):
    c,p=d15.iloc[-1],d15.iloc[-2]; hi=d15.high.shift(2).rolling(8).max().iloc[-1]; lo=d15.low.shift(2).rolling(8).min().iloc[-1]
    long_retest=p.close>hi and c.low<=hi*1.002 and c.close>hi
    short_retest=p.close<lo and c.high>=lo*.998 and c.close<lo
    if long_retest:return 'LONG','BREAKOUT_RETEST_ACCEPT',float(c.atr*1.70),24
    if short_retest:return 'SHORT','BREAKOUT_RETEST_ACCEPT',float(c.atr*1.70),24
    return None,'WAIT_BREAKOUT_RETEST',None,0

def detect_trend_pullback(d5,d15,d1,d4,snap):
    c,p=d15.iloc[-1],d15.iloc[-2]; h=d1.iloc[-1]
    bull=h.ema20>h.ema50 and h.ema20_slope3>0 and h.adx>=18
    bear=h.ema20<h.ema50 and h.ema20_slope3<0 and h.adx>=18
    if bull and p.low<=p.ema20*1.003 and c.close>c.ema20 and c.close>p.close:return 'LONG','HTF_PULLBACK_RESUME',float(c.atr*1.65),23
    if bear and p.high>=p.ema20*.997 and c.close<c.ema20 and c.close<p.close:return 'SHORT','HTF_PULLBACK_RESUME',float(c.atr*1.65),23
    return None,'WAIT_HTF_PULLBACK',None,0

def detect_mean_reversion(d5,d15,d1,d4,snap):
    c=d5.iloc[-1]; z=(c.close-c.ema20)/c.atr if c.atr else 0
    if z<-1.7 and c.rsi<32 and c.close>c.open:return 'LONG','RANGE_EXTREME_SNAPBACK',float(c.atr*1.45),22
    if z>1.7 and c.rsi>68 and c.close<c.open:return 'SHORT','RANGE_EXTREME_SNAPBACK',float(c.atr*1.45),22
    return None,'WAIT_RANGE_EXTREME',None,0

DETECTORS={'SMC_SWEEP':detect_smc,'EMA_SCALP':detect_ema,'MOMENTUM':detect_momentum,
           'BREAKOUT':detect_breakout,'TREND_PULLBACK':detect_trend_pullback,'MEAN_REVERSION':detect_mean_reversion}

DECISION_COLUMNS=['schema_version','candidate_id','time','strategy','side','setup','regime','direction','strength','volatility','market_state','run_mode','context_fit','context_reason','score_pass',
 'price','rsi_5m','adx_1h','er24_1h','vol_ratio_1h','atr_pct_5m','body_atr_5m','volume_ratio_5m','volume_ratio_15m',
 'ema20_slope3_15m','ema20_slope3_1h','ema20_slope3_4h','trend_aligned','countertrend','smc_sweep_depth_atr','smc_reclaim_quality','smc_wick_body_ratio','score_total','score_trend','score_regime','score_trigger',
 'score_momentum','score_volume','score_structure','estimated_cost_R','initial_rr','router_allowed','router_reason','decision','open_risk_pct']
ENTRY_COLUMNS=DECISION_COLUMNS+['entry','initial_stop','initial_target','stop_distance','stop_pct','risk_eur','qty','notional']
TRADE_COLUMNS=['schema_version','candidate_id','strategy','entry_time','exit_time','side','setup','regime','score_total','entry','exit',
 'initial_stop','final_stop','target','raw_R','net_R','reason','gross_pnl_eur','fees_eur','slippage_eur','pnl_eur','balance',
 'MFE_R','MAE_R','giveback_R','bars_held']
EVENT_COLUMNS=['schema_version','candidate_id','time','strategy','event','price','r_value','old_stop','new_stop','detail']
SHADOW_COLUMNS=['schema_version','candidate_id','strategy','side','setup','decision','block_reason','entry_time','finish_time','entry',
 'stop','target','MFE_R','MAE_R','first_touch','bars','regime','score_total']

EVENTS_FILE=os.path.join(DATA_DIR,'btc_v172_management_events.csv')
SHADOW_FILE=os.path.join(DATA_DIR,'btc_v172_shadow_outcomes.csv')

def append_fixed(path,row,columns):
    clean={k:row.get(k) for k in columns}
    exists=os.path.exists(path) and os.path.getsize(path)>0
    pd.DataFrame([clean],columns=columns).to_csv(path,mode='a' if exists else 'w',header=not exists,index=False)

def append_decision(row): append_fixed(DECISIONS_FILE,row,DECISION_COLUMNS)
def append_entry(row): append_fixed(FEATURES_FILE,row,ENTRY_COLUMNS)
def log_event(row): append_fixed(EVENTS_FILE,row,EVENT_COLUMNS)

def candidate_id(key,t,side,setup):
    raw=f"{key}|{pd.Timestamp(t).isoformat()}|{side}|{setup}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]

def signal_for(key,c,snap):
    if len(c)<600:return None
    d5,d15,d1,d4=frames(c)
    if min(len(d15),len(d1),len(d4))<200:return None
    side,setup,dist,trigger=DETECTORS[key](d5,d15,d1,d4,snap)
    feat=base_features(key,side,setup,d5,d15,d1,d4,snap); feat['schema_version']='3.1'
    if not side:return {'signal':None,'time':d5.index[-1],'reason':setup,'features':feat}
    c5=d5.iloc[-1]
    # V1.7.2 diagnostics: measure hypotheses without using them as hard research filters.
    feat['smc_sweep_depth_atr']=None; feat['smc_reclaim_quality']=None; feat['smc_wick_body_ratio']=None
    if key=='SMC_SWEEP':
        try:
            prev_hi=float(d5.high.shift(1).rolling(18).max().iloc[-1]); prev_lo=float(d5.low.shift(1).rolling(18).min().iloc[-1]); a=max(float(c5.atr),1e-9)
            if side=='LONG':
                feat['smc_sweep_depth_atr']=max(0.0,(prev_lo-float(c5.low))/a); feat['smc_reclaim_quality']=(float(c5.close)-prev_lo)/a
                wick=max(0.0,min(float(c5.open),float(c5.close))-float(c5.low))
            else:
                feat['smc_sweep_depth_atr']=max(0.0,(float(c5.high)-prev_hi)/a); feat['smc_reclaim_quality']=(prev_hi-float(c5.close))/a
                wick=max(0.0,float(c5.high)-max(float(c5.open),float(c5.close)))
            body=max(abs(float(c5.close)-float(c5.open)),a*0.02); feat['smc_wick_body_ratio']=wick/body
        except Exception: pass
    ctx={'momentum':15 if (side=='LONG' and c5.rsi>=55) or (side=='SHORT' and c5.rsi<=45) else 9,
         'volume':10 if c5.volume_ratio>=1.10 else 7 if c5.volume_ratio>=.80 else 3,
         'structure':10 if abs(float(c5.close-c5.ema20))/float(c5.atr or 1)<1.5 else 6}
    scores=score_setup(key,side,trigger,ctx,snap); feat.update(scores)
    cid=candidate_id(key,d5.index[-1],side,setup); feat['candidate_id']=cid
    entry=float(c5.close); base_dist=float(dist)
    # Cost-aware stop floor keeps risk fixed at 1% while avoiding economically absurd micro-stops.
    roundtrip_rate=2*(TAKER_FEE+SLIPPAGE_BPS/10000.0)
    min_dist=entry*roundtrip_rate/MAX_COST_R
    dist=max(base_dist,min_dist)
    est=(entry/dist)*roundtrip_rate
    feat['estimated_cost_R']=est; feat['initial_rr']=STRATEGIES[key]['rr']
    context_fit,context_reason=router(key,side,snap)
    score_pass=bool(scores['score_total']>=MIN_SCORE)
    feat['run_mode']=RUN_MODE; feat['context_fit']=context_fit; feat['context_reason']=context_reason; feat['score_pass']=score_pass
    if RESEARCH_MODE:
        allowed = snap.get('market_state') != 'WARMUP'
        why = 'RESEARCH_EXECUTE' if allowed else 'REGIME_WARMUP'
    else:
        allowed=context_fit and score_pass
        why=context_reason if not context_fit else ('SCORE_BELOW_BASELINE' if not score_pass else 'CONTEXT_OK')
    feat['router_allowed']=allowed;feat['router_reason']=why
    return {'signal':side,'time':d5.index[-1],'entry':entry,'stop_distance':dist,'reason':setup,'allowed':allowed,
            'router_reason':why,'context_fit':context_fit,'context_reason':context_reason,'score_pass':score_pass,
            'features':feat,'score':scores['score_total'],'estimated_cost_R':est,'candidate_id':cid,'regime':snap.get('regime')}

def start_shadow(state,sig,decision):
    cid=sig['candidate_id']
    if cid in state['shadow_candidates']:return
    entry=sig['entry']; dist=sig['stop_distance']; side=sig['signal']; rr=STRATEGIES[sig['features']['strategy']]['rr']
    state['shadow_candidates'][cid]={'candidate_id':cid,'strategy':sig['features']['strategy'],'side':side,'setup':sig['reason'],
      'decision':decision,'block_reason':sig.get('router_reason'),'entry_time':sig['time'].isoformat(),'entry':entry,
      'stop':entry-dist if side=='LONG' else entry+dist,'target':entry+rr*dist if side=='LONG' else entry-rr*dist,
      'dist':dist,'MFE_R':0.0,'MAE_R':0.0,'first_touch':'NONE','bars':0,'regime':sig.get('regime'),'score_total':sig.get('score')}

def update_shadows(state,bar):
    done=[]
    for cid,x in list(state.get('shadow_candidates',{}).items()):
        if pd.Timestamp(bar.timestamp)<=pd.Timestamp(x['entry_time']):continue
        h,l=float(bar.high),float(bar.low); x['bars']+=1
        fav=(h-x['entry'])/x['dist'] if x['side']=='LONG' else (x['entry']-l)/x['dist']
        adv=(l-x['entry'])/x['dist'] if x['side']=='LONG' else (x['entry']-h)/x['dist']
        x['MFE_R']=max(x['MFE_R'],fav);x['MAE_R']=min(x['MAE_R'],adv)
        stop_hit=l<=x['stop'] if x['side']=='LONG' else h>=x['stop']
        tp_hit=h>=x['target'] if x['side']=='LONG' else l<=x['target']
        if stop_hit and tp_hit:x['first_touch']='AMBIGUOUS'
        elif stop_hit:x['first_touch']='STOP'
        elif tp_hit:x['first_touch']='TARGET'
        if x['first_touch']!='NONE' or x['bars']>=96:
            if x['first_touch']=='NONE':x['first_touch']='TIME'
            append_fixed(SHADOW_FILE,{**x,'schema_version':'3.1','finish_time':pd.Timestamp(bar.timestamp).isoformat()},SHADOW_COLUMNS);done.append(cid)
    for cid in done:state['shadow_candidates'].pop(cid,None)

def is_cooldown(s,now=None):
    if not s.get('cooldown_until'):return False
    now=now or pd.Timestamp.now(tz='UTC'); until=pd.Timestamp(s['cooldown_until'])
    if now>=until:s['cooldown_until']=None;return False
    return True

def open_risk(state):
    # Use the actual risk stored on each live position. This matters when
    # drawdown mode reduces a new trade from 1.0% to 0.5%.
    return sum(float(x['position'].get('risk_pct', RISK_PER_TRADE))
               for x in state['strategies'].values() if x.get('position'))

def open_position_count(state):
    return sum(1 for x in state['strategies'].values() if x.get('position'))

def refresh_risk_day(account,now=None):
    # Every strategy has its own independent daily-risk clock and equity.
    now=now or utc_now(); day=now.date().isoformat()
    if account.get('risk_day')!=day:
        account['risk_day']=day
        account['day_start_balance']=account['balance']

def risk_permission(account):
    refresh_risk_day(account)
    if RISK_PER_TRADE>0.0100001:return False,'RISK_PER_TRADE_GT_1PCT'
    dd=account['balance']/max(account.get('peak_balance',account['balance']),1e-9)-1
    if RESEARCH_MODE:
        if dd<=-RESEARCH_EMERGENCY_DD:return False,'RESEARCH_EMERGENCY_DD'
        return True,'RESEARCH_RISK_OK'
    day_start=max(float(account.get('day_start_balance') or account['balance']),1e-9)
    if account['balance']/day_start-1<=-DAILY_LOSS_LIMIT:return False,'STRATEGY_DAILY_LOSS_LIMIT'
    if dd<=-DRAWDOWN_HARD:return False,'STRATEGY_HARD_DRAWDOWN_MODE'
    return True,'RISK_OK'

def effective_risk_pct(account):
    if RESEARCH_MODE:return min(RISK_PER_TRADE,0.01)
    dd=account['balance']/max(account.get('peak_balance',account['balance']),1e-9)-1
    return min(RISK_PER_TRADE,0.005 if dd<=-DRAWDOWN_SOFT else RISK_PER_TRADE)

def open_position(key,state,s,sig):
    entry,dist,side=sig['entry'],sig['stop_distance'],sig['signal'];cfg=STRATEGIES[key]
    stop=entry-dist if side=='LONG' else entry+dist;target=entry+cfg['rr']*dist if side=='LONG' else entry-cfg['rr']*dist
    risk_pct=effective_risk_pct(s);risk_eur=s['balance']*risk_pct;qty=risk_eur/dist;notional=qty*entry
    p={'candidate_id':sig['candidate_id'],'side':side,'entry_time':sig['time'].isoformat(),'entry_price':entry,
       'initial_stop':stop,'stop':stop,'target':target,'stop_distance':dist,'risk_eur':risk_eur,'risk_pct':risk_pct,
       'qty':qty,'notional':notional,'setup':sig['reason'],'regime':sig.get('regime'),'mfe_r':0.0,'mae_r':0.0,
       'protected_stage':0,'exit_state':'UNPROTECTED','profit_zone_seen':False,'score_total':sig.get('score'),'estimated_cost_R':sig.get('estimated_cost_R'),'bars_held':0}
    s['position']=p;s['signals']+=1;s['last_signal']=f"{side} · {sig['reason']}";s['last_signal_time']=sig['time'].isoformat()
    append_entry({**sig['features'],'entry':entry,'initial_stop':stop,'initial_target':target,'stop_distance':dist,
                  'stop_pct':dist/entry,'risk_eur':risk_eur,'qty':qty,'notional':notional})
    log_event({'schema_version':'3.1','candidate_id':p['candidate_id'],'time':p['entry_time'],'strategy':key,'event':'OPEN',
               'price':entry,'r_value':0,'old_stop':stop,'new_stop':stop,'detail':f"risk_pct={risk_pct:.4f}"})
    print(f'[ENTRY][{key}] {side} {entry:.2f} stop={stop:.2f} tp={target:.2f} score={sig.get("score"):.0f}',flush=True)

def close_position(key,state,s,exit_price,exit_time,reason):
    p=s['position'];raw_r=(exit_price-p['entry_price'])/p['stop_distance'] if p['side']=='LONG' else (p['entry_price']-exit_price)/p['stop_distance']
    gross=p['risk_eur']*raw_r;fee=(p['notional']+p['qty']*exit_price)*TAKER_FEE;slip=(p['notional']+p['qty']*exit_price)*(SLIPPAGE_BPS/10000.0)
    pnl=gross-fee-slip;net_r=pnl/p['risk_eur'] if p['risk_eur'] else 0
    s['balance']+=pnl;s['peak_balance']=max(s['peak_balance'],s['balance']);s['max_drawdown']=min(s['max_drawdown'],s['balance']/s['peak_balance']-1);state['total_trades']+=1
    if pnl>0:
        state['winning_trades']+=1;state['gross_profit']+=pnl;s['loss_streak']=0
    else:
        state['losing_trades']+=1;state['gross_loss']+=abs(pnl);s['loss_streak']+=1
        if (not RESEARCH_MODE) and s['loss_streak']>=COOLDOWN_AFTER_LOSSES:s['cooldown_until']=(exit_time+timedelta(hours=COOLDOWN_HOURS)).isoformat()
    giveback=max(0,float(p.get('mfe_r',0))-raw_r)
    append_fixed(TRADES_FILE,{'schema_version':'3.1','candidate_id':p['candidate_id'],'strategy':key,'entry_time':p['entry_time'],
      'exit_time':exit_time.isoformat(),'side':p['side'],'setup':p['setup'],'regime':p.get('regime'),'score_total':p.get('score_total'),
      'entry':p['entry_price'],'exit':exit_price,'initial_stop':p['initial_stop'],'final_stop':p['stop'],'target':p['target'],
      'raw_R':raw_r,'net_R':net_r,'reason':reason,'gross_pnl_eur':gross,'fees_eur':fee,'slippage_eur':slip,'pnl_eur':pnl,
      'balance':s['balance'],'MFE_R':p['mfe_r'],'MAE_R':p['mae_r'],'giveback_R':giveback,'bars_held':p.get('bars_held',0)},TRADE_COLUMNS)
    log_event({'schema_version':'3.1','candidate_id':p['candidate_id'],'time':exit_time.isoformat(),'strategy':key,'event':'EXIT',
               'price':exit_price,'r_value':raw_r,'old_stop':p['stop'],'new_stop':p['stop'],'detail':reason})
    s['position']=None;print(f'[EXIT][{key}] {reason} rawR={raw_r:.2f} netR={net_r:.2f}',flush=True)

def move_stop(key,p,t,new_stop,event,r_value):
    old=p['stop']
    if p['side']=='LONG':new_stop=max(old,new_stop)
    else:new_stop=min(old,new_stop)
    if abs(new_stop-old)>1e-9:
        p['stop']=new_stop
        log_event({'schema_version':'3.1','candidate_id':p['candidate_id'],'time':t.isoformat(),'strategy':key,'event':event,
                   'price':None,'r_value':r_value,'old_stop':old,'new_stop':new_stop,'detail':''})

def check_position(key,state,s,candle):
    p=s.get('position')
    if not p:return
    h,l,cl,t=float(candle.high),float(candle.low),float(candle.close),candle.timestamp;p['bars_held']=p.get('bars_held',0)+1
    fav=(h-p['entry_price'])/p['stop_distance'] if p['side']=='LONG' else (p['entry_price']-l)/p['stop_distance']
    adv=(l-p['entry_price'])/p['stop_distance'] if p['side']=='LONG' else (p['entry_price']-h)/p['stop_distance']
    p['mfe_r']=max(p.get('mfe_r',0),fav);p['mae_r']=min(p.get('mae_r',0),adv)
    stop_hit=l<=p['stop'] if p['side']=='LONG' else h>=p['stop'];tp_hit=h>=p['target'] if p['side']=='LONG' else l<=p['target']
    if stop_hit and tp_hit:
        log_event({'schema_version':'3.1','candidate_id':p['candidate_id'],'time':t.isoformat(),'strategy':key,'event':'INTRABAR_AMBIGUOUS',
                   'price':cl,'r_value':None,'old_stop':p['stop'],'new_stop':p['stop'],'detail':'stop_and_target_same_5m_bar; conservative stop-first'})
        return close_position(key,state,s,p['stop'],t,'AMBIGUOUS_STOP_FIRST')
    if stop_hit:return close_position(key,state,s,p['stop'],t,'STOP' if p.get('protected_stage',0)==0 else 'PROTECTED_STOP')
    if tp_hit:return close_position(key,state,s,p['target'],t,'TAKE_PROFIT')

    # Exit Engine V2: stateful protection. Thresholds are deliberately broad hypotheses, not fitted per strategy.
    mfe=p['mfe_r'];close_r=(cl-p['entry_price'])/p['stop_distance'] if p['side']=='LONG' else (p['entry_price']-cl)/p['stop_distance']
    prev_state=p.get('exit_state','UNPROTECTED')
    if mfe>=PROFIT_ZONE_R and not p.get('profit_zone_seen'):
        p['profit_zone_seen']=True;p['exit_state']='PROFIT_ZONE'
        log_event({'schema_version':'3.1','candidate_id':p['candidate_id'],'time':t.isoformat(),'strategy':key,'event':'PROFIT_ZONE_ENTERED','price':cl,'r_value':close_r,'old_stop':p['stop'],'new_stop':p['stop'],'detail':f'mfe={mfe:.2f}'})
    # Deterioration = meaningful giveback after a useful excursion plus weak close. Protect modestly, never widen risk.
    giveback=max(0.0,mfe-close_r)
    deterioration=(mfe>=DYNAMIC_PROTECT_R and giveback>=DETERIORATION_GIVEBACK_R and close_r<=DETERIORATION_CLOSE_R)
    if deterioration and p.get('protected_stage',0)<1:
        # Lock around cost-aware scratch rather than forcing BE on every +0.5R trade.
        cushion=min(.22,max(.05,float(p.get('estimated_cost_R') or .10)+.02))
        ns=p['entry_price']+cushion*p['stop_distance'] if p['side']=='LONG' else p['entry_price']-cushion*p['stop_distance']
        move_stop(key,p,t,ns,'DETERIORATION_PROTECT',close_r);p['protected_stage']=1;p['exit_state']='PROTECTED'
    if mfe>=TRAIL_TRIGGER_R and close_r>=1.0:
        lock=max(LOCK_PROFIT_R,close_r-TRAIL_GAP_R)
        ns=p['entry_price']+lock*p['stop_distance'] if p['side']=='LONG' else p['entry_price']-lock*p['stop_distance']
        move_stop(key,p,t,ns,'TRAIL_UPDATED',close_r);p['protected_stage']=3;p['exit_state']='TRAILING'
    elif mfe>=LOCK_TRIGGER_R and close_r>=.65 and p.get('protected_stage',0)<2:
        ns=p['entry_price']+LOCK_PROFIT_R*p['stop_distance'] if p['side']=='LONG' else p['entry_price']-LOCK_PROFIT_R*p['stop_distance']
        move_stop(key,p,t,ns,'MFE_LOCK_ARMED',close_r);p['protected_stage']=2;p['exit_state']='PROTECTED'
    elif mfe>=BE_TRIGGER_R and close_r>=.35 and p.get('protected_stage',0)<1:
        cushion=min(.30,max(.08,float(p.get('estimated_cost_R') or .10)+.03))
        ns=p['entry_price']+cushion*p['stop_distance'] if p['side']=='LONG' else p['entry_price']-cushion*p['stop_distance']
        move_stop(key,p,t,ns,'BE_ARMED',close_r);p['protected_stage']=1;p['exit_state']='PROTECTED'
    if p.get('exit_state')!=prev_state:
        log_event({'schema_version':'3.1','candidate_id':p['candidate_id'],'time':t.isoformat(),'strategy':key,'event':'EXIT_STATE_CHANGE','price':cl,'r_value':close_r,'old_stop':p['stop'],'new_stop':p['stop'],'detail':f'{prev_state}->{p.get("exit_state")};mfe={mfe:.2f};giveback={giveback:.2f}'})

    # Strategy invalidation on confirmed close, not on a wick.
    if key=='BREAKOUT' and p['bars_held']>=2:
        fail=(p['side']=='LONG' and close_r<-.35) or (p['side']=='SHORT' and close_r<-.35)
        if fail:return close_position(key,state,s,cl,t,'FAILED_BREAKOUT')
    if t-pd.Timestamp(p['entry_time'])>=pd.Timedelta(hours=STRATEGIES[key]['max_hours']):
        return close_position(key,state,s,cl,t,'TIME_EXIT')

def stats(state):
    # Aggregate display only; execution/risk remains completely strategy-isolated.
    live=[state['strategies'][k] for k,cfg in STRATEGIES.items() if cfg['enabled']]
    n_accounts=max(len(live),1)
    combined_balance=sum(float(a.get('balance',START_BALANCE)) for a in live)
    combined_start=START_BALANCE*n_accounts
    combined_peak=sum(float(a.get('peak_balance',START_BALANCE)) for a in live)
    worst_dd=min([float(a.get('max_drawdown',0)) for a in live] or [0.0])
    t=state['total_trades']; w=state['winning_trades']
    pf=state['gross_profit']/state['gross_loss'] if state['gross_loss'] else (999 if state['gross_profit'] else 0)
    return {'balance':combined_balance,'peak_balance':combined_peak,'return_pct':(combined_balance/combined_start-1)*100,
            'net_pnl':combined_balance-combined_start,'net_r':sum(float(x.get('net_R') or 0) for x in _read_csv_records(TRADES_FILE,100000)),
            'trades':t,'wins':w,'losses':state.get('losing_trades',0),'winrate':100*w/t if t else 0,'pf':pf,
            'max_dd':100*worst_dd,'open_risk_pct':100*open_risk(state),'accounts':n_accounts}


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
                    'last_signal':st.get('last_signal','—'),'account_balance':float(st.get('balance',START_BALANCE)),'account_return':(float(st.get('balance',START_BALANCE))/START_BALANCE-1)*100})
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
</style></head><body><div class="w"><div class="top"><div><div class="title">BTC V1</div><div class="sub">Control Center · independent strategy accounts · paper trading · refresh 20s</div></div><div class="running">● RUNNING</div></div>
<div class="grid">
<div class="card"><div class="label">Combined strategy equity</div><div class="value {{'pos' if s.return_pct>=0 else 'neg'}}">€{{'%.2f'|format(s.balance)}}</div><div class="mini">{{'%+.2f'|format(s.return_pct)}}%</div></div>
<div class="card"><div class="label">Net P/L</div><div class="value {{'pos' if s.net_pnl>=0 else 'neg'}}">€{{'%+.2f'|format(s.net_pnl)}}</div><div class="mini">{{'%+.2f'|format(s.net_r)}}R totaal</div></div>
<div class="card"><div class="label">Trades</div><div class="value">{{s.trades}}</div><div class="mini">{{s.wins}}W · {{s.losses}}L</div></div>
<div class="card"><div class="label">Winrate</div><div class="value">{{'%.1f'|format(s.winrate)}}%</div><div class="mini">PF {{'%.2f'|format(s.pf) if s.pf<900 else '∞'}}</div></div>
<div class="card"><div class="label">Max DD</div><div class="value {{'neg' if s.max_dd<0 else ''}}">{{'%.2f'|format(s.max_dd)}}%</div><div class="mini">Worst strategy DD</div></div>
<div class="card"><div class="label">Open risk</div><div class="value">{{'%.2f'|format(s.open_risk_pct)}}%</div><div class="mini">1.00% max per strategy account</div></div></div>
<div class="section market"><div class="card"><div class="label">Market regime</div><div class="value">{{r.regime}}</div><div class="mini">{{r.direction}} · BTC ${{'{:,.0f}'.format(r.price) if r.price else '—'}}</div></div><div class="card"><div class="label">ADX 1H</div><div class="value">{{'%.1f'|format(r.adx) if r.adx is not none else '—'}}</div></div><div class="card"><div class="label">ER24 1H</div><div class="value">{{'%.3f'|format(r.er24) if r.er24 is not none else '—'}}</div></div><div class="card"><div class="label">Vol ratio</div><div class="value">{{'%.2f'|format(r.vol_ratio) if r.vol_ratio is not none else '—'}}</div></div><div class="card"><div class="label">Open positions</div><div class="value">{{openpos|length}}</div></div></div>
<div class="card section"><h2>Engine monitor</h2><div class="tablewrap"><table><tr><th>Engine</th><th>Mode</th><th>Account</th><th>Return</th><th>Trades</th><th>WR</th><th>Net R</th><th>P/L</th><th>PF</th><th>Signals</th><th>Blocked</th><th>Loss streak</th><th>Laatste signal</th></tr>{% for x in strat %}<tr><td><div class="strategy">{{x.label}}</div><div class="muted small">{{x.key}}</div></td><td><span class="tag {{'live' if x.mode=='LIVE' else 'shadow'}}">{{x.mode}}</span></td><td>€{{'%.2f'|format(x.account_balance)}}</td><td class="{{'pos' if x.account_return>=0 else 'neg'}}">{{'%+.2f'|format(x.account_return)}}%</td><td>{{x.trades}}</td><td>{{'%.1f'|format(x.winrate)}}%</td><td class="{{'pos' if x.net_r>0 else 'neg' if x.net_r<0 else ''}}">{{'%+.2f'|format(x.net_r)}}</td><td class="{{'pos' if x.pnl>0 else 'neg' if x.pnl<0 else ''}}">€{{'%+.2f'|format(x.pnl)}}</td><td>{{'%.2f'|format(x.pf) if x.pf<900 else '∞'}}</td><td>{{x.signals}}</td><td>{{x.blocked}}</td><td>{{x.loss_streak}}</td><td>{{x.last_signal}}{% if x.cooldown_until %}<div class="amber small">Cooldown → {{x.cooldown_until[11:16]}}</div>{% endif %}</td></tr>{% endfor %}</table></div></div>
<div class="card section"><h2>Open trades</h2>{% if openpos %}<div class="tablewrap"><table><tr><th>Strategy</th><th>Side</th><th>Entry</th><th>Current</th><th>SL</th><th>TP</th><th>Current R</th><th>MFE</th><th>MAE</th><th>Risk €</th></tr>{% for p in openpos %}<tr><td class="strategy">{{p.strategy}}</td><td>{{p.side}}</td><td>${{'{:,.0f}'.format(p.entry_price)}}</td><td>${{'{:,.0f}'.format(p.current_price)}}</td><td>${{'{:,.0f}'.format(p.stop)}}</td><td>${{'{:,.0f}'.format(p.target)}}</td><td class="{{'pos' if p.current_r>=0 else 'neg'}}">{{'%+.2f'|format(p.current_r)}}R</td><td class="pos">{{'%.2f'|format(p.mfe_r)}}R</td><td class="neg">{{'%.2f'|format(p.mae_r)}}R</td><td>€{{'%.2f'|format(p.risk_eur)}}</td></tr>{% endfor %}</table></div>{% else %}<div class="empty">Geen open trades — V1 wacht op een geldige setup.</div>{% endif %}</div>
<div class="two section"><div class="card"><h2>Recente decisions / signals</h2>{% if decisions %}<div class="tablewrap"><table><tr><th>Tijd</th><th>Strategy</th><th>Side</th><th>Decision</th><th>Setup</th><th>Regime</th><th>ADX</th></tr>{% for d in decisions %}<tr><td>{{d.time[11:16] if d.time else '—'}}</td><td class="strategy">{{d.strategy}}</td><td>{{d.side}}</td><td><span class="pill {{'pos' if d.decision=='OPEN' else 'amber' if d.decision=='SHADOW' else ''}}">{{d.decision}}</span></td><td>{{d.setup}}</td><td>{{d.regime}}</td><td>{{'%.1f'|format(d.adx_1h) if d.adx_1h is not none else '—'}}</td></tr>{% endfor %}</table></div>{% else %}<div class="empty">Nog geen signalen gelogd.</div>{% endif %}</div><div class="card"><h2>Performance per regime</h2>{% if regimes %}<div class="tablewrap"><table><tr><th>Regime</th><th>Trades</th><th>WR</th><th>Net R</th><th>P/L</th></tr>{% for x in regimes %}<tr><td class="strategy">{{x.regime}}</td><td>{{x.trades}}</td><td>{{'%.1f'|format(x.winrate)}}%</td><td class="{{'pos' if x.net_r>0 else 'neg' if x.net_r<0 else ''}}">{{'%+.2f'|format(x.net_r)}}</td><td class="{{'pos' if x.pnl>0 else 'neg' if x.pnl<0 else ''}}">€{{'%+.2f'|format(x.pnl)}}</td></tr>{% endfor %}</table></div>{% else %}<div class="empty">Regime-statistieken verschijnen na de eerste gesloten trades.</div>{% endif %}</div></div>
<div class="card section"><h2>Trade history</h2>{% if recent %}<div class="tablewrap"><table><tr><th>Exit</th><th>Strategy</th><th>Side</th><th>Setup</th><th>Entry</th><th>Exit</th><th>Reason</th><th>Gross R</th><th>Costs R</th><th>Net R</th><th>MFE</th><th>MAE</th><th>Costs €</th><th>P/L</th><th>Balance</th></tr>{% for t in recent %}<tr><td>{{t.exit_time[5:16]|replace('T',' ')}}</td><td class="strategy">{{t.strategy}}</td><td>{{t.side}}</td><td>{{t.setup}}</td><td>${{'{:,.0f}'.format(t.entry)}}</td><td>${{'{:,.0f}'.format(t.exit)}}</td><td>{{t.reason}}</td><td>{{'%+.2f'|format(t.raw_R)}}R</td><td class="amber">-{{'%.2f'|format(t.costs_R)}}R</td><td class="{{'pos' if t.net_R>=0 else 'neg'}}">{{'%+.2f'|format(t.net_R)}}R</td><td class="pos">{{'%.2f'|format(t.MFE_R)}}R</td><td class="neg">{{'%.2f'|format(t.MAE_R)}}R</td><td>€{{'%.2f'|format(t.costs_eur)}}</td><td class="{{'pos' if t.pnl_eur>=0 else 'neg'}}">€{{'%+.2f'|format(t.pnl_eur)}}</td><td>€{{'%.2f'|format(t.balance)}}</td></tr>{% endfor %}</table></div>{% else %}<div class="empty">Nog geen gesloten trades.</div>{% endif %}</div>
<div class="card section"><h2>Data & exports</h2><div class="downloads"><a class="btn" href="/download/trades">↓ Trades CSV</a><a class="btn" href="/download/features">↓ Entry features CSV</a><a class="btn" href="/download/decisions">↓ Decisions CSV</a><a class="btn" href="/download/events">↓ Management events</a><a class="btn" href="/download/shadow">↓ Shadow outcomes</a><a class="btn" href="/api/status">API status</a></div></div></div></body></html>'''
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
    return render_template_string(DASH,s=stats(st),r=r,strat=strategy_dashboard(st),openpos=open_positions_dashboard(st,price),decisions=_read_csv_records(DECISIONS_FILE,36),regimes=regime_performance(),recent=recent())
@app.get('/api/status')
def status():
    st=load_state(); c=load_candles(); return jsonify({'version':'BTC-V1.7.2-RESEARCH','run_mode':RUN_MODE,'stats':stats(st),'regime':regime_snapshot(c),'strategies':STRATEGIES})
def dl(path,name):
    if not os.path.exists(path):return {'error':'Nog geen bestand.'},404
    return send_file(path,mimetype='text/csv',as_attachment=True,download_name=name)
@app.get('/download/trades')
def d1():return dl(TRADES_FILE,'btc_v172_trades.csv')
@app.get('/download/features')
def d2():return dl(FEATURES_FILE,'btc_v172_entries.csv')
@app.get('/download/decisions')
def d3():return dl(DECISIONS_FILE,'btc_v172_decisions.csv')
@app.get('/download/events')
def d4():return dl(EVENTS_FILE,'btc_v172_management_events.csv')
@app.get('/download/shadow')
def d5():return dl(SHADOW_FILE,'btc_v172_shadow_outcomes.csv')
@app.get('/health')
def health():return {'status':'ok','version':'BTC-V1.7.2-RESEARCH','run_mode':RUN_MODE,'risk_per_trade':RISK_PER_TRADE},200
def run_dashboard():app.run(host='0.0.0.0',port=int(os.getenv('PORT','8080')),threaded=True,use_reloader=False)

def main():
    threading.Thread(target=run_dashboard,daemon=True).start();print(f'BTC V1.7.2 RESEARCH — PAPER ONLY — mode={RUN_MODE} — independent strategy accounts — risk/trade={RISK_PER_TRADE:.2%}',flush=True)
    state=load_state();candles=update_candles(load_candles())
    while True:
        try:
            candles=update_candles(candles);newest=candles.timestamp.iloc[-1];prev=state.get('last_processed_5m')
            if prev is None or newest>pd.Timestamp(prev):
                new_rows=candles if prev is None else candles[candles.timestamp>pd.Timestamp(prev)]
                for _,bar in new_rows.iterrows():
                    update_shadows(state,bar)
                    for key in STRATEGIES:check_position(key,state,state['strategies'][key],bar)
                    state['last_processed_5m']=bar.timestamp.isoformat()
                snap=regime_snapshot(candles)
                for key,cfg in STRATEGIES.items():
                    st=state['strategies'][key];st['scans']+=1;sig=signal_for(key,candles,snap)
                    if not sig:continue
                    if not sig.get('signal'):
                        append_decision({**sig['features'],'decision':'NO_SETUP','router_reason':'NO_CANDIDATE',
                                         'router_allowed':False,'open_risk_pct':100*open_risk(state)})
                        continue
                    st['last_signal']=f"{sig['signal']} · {sig['reason']}";st['last_signal_time']=sig['time'].isoformat()
                    decision=None
                    if not cfg['enabled']:
                        decision='SHADOW_STRATEGY'
                    elif not sig.get('allowed'):
                        decision='BLOCK_'+str(sig.get('router_reason'))
                    elif st.get('position') is not None:
                        decision='BLOCK_ALREADY_OPEN'
                    elif (not RESEARCH_MODE) and is_cooldown(st):
                        decision='BLOCK_STRATEGY_COOLDOWN'
                    else:
                        risk_ok,risk_reason=risk_permission(st)
                        if not risk_ok:decision='BLOCK_'+risk_reason
                        else:
                            open_position(key,state,st,sig);decision='OPEN'
                    if decision!='OPEN':
                        st['blocked']+=1
                        start_shadow(state,sig,decision)
                    append_decision({**sig['features'],'decision':decision,'router_reason':sig.get('router_reason'),
                                     'router_allowed':sig.get('allowed'),'open_risk_pct':100*open_risk(state)})
                save_state(state)
                print(f"[STATUS] BTC={candles.close.iloc[-1]:,.0f} state={snap.get('market_state')} dir={snap.get('direction')} combined_equity=€{stats(state)['balance']:.2f} trades={state['total_trades']}",flush=True)
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:save_state(state);break
        except Exception as e:print('[ERROR]',repr(e),flush=True);save_state(state);time.sleep(60)
if __name__=='__main__':main()
