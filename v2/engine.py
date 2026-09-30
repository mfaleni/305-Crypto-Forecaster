from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class CostModel:
    maker_fee_bps: float = 5.0
    taker_fee_bps: float = 10.0
    slippage_bps: float = 1.0
    entry_is_maker: bool = True
    exit_is_maker: bool = True
    def fee_rate(self, maker: bool) -> float:
        return (self.maker_fee_bps if maker else self.taker_fee_bps) / 10000.0
    def estimated_round_trip_rate(self) -> float:
        return self.fee_rate(self.entry_is_maker)+self.fee_rate(self.exit_is_maker)+2*self.slippage_bps/10000.0

@dataclass(frozen=True)
class StrategyConfig:
    ema_fast:int=20; ema_slow:int=50; rsi_len:int=14; atr_len:int=14; bb_len:int=20; bb_std:float=2.0; adx_len:int=14
    range_adx_max:float=22.0; trend_adx_min:float=24.0; range_rsi_buy:float=34.0; trend_rsi_buy_min:float=42.0; trend_rsi_buy_max:float=58.0
    stop_atr:float=1.15; target_atr_range:float=1.35; target_atr_trend:float=1.8; max_hold_bars:int=36
    risk_per_trade:float=0.005; max_notional_fraction:float=1.0; min_edge_multiple_of_cost:float=1.35

@dataclass
class Trade:
    signal_time:object; entry_time:object; exit_time:object; regime:str; entry:float; exit:float; qty:float; stop:float; target:float; exit_reason:str
    gross_pnl:float; fees:float; slippage:float; net_pnl:float; return_on_notional:float; bars_held:int

def _rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def add_features(df,cfg):
    x=df.copy(); c,h,l=x.close,x.high,x.low
    x['ema_fast']=c.ewm(span=cfg.ema_fast,adjust=False).mean(); x['ema_slow']=c.ewm(span=cfg.ema_slow,adjust=False).mean()
    d=c.diff(); rs=_rma(d.clip(lower=0),cfg.rsi_len)/_rma((-d).clip(lower=0),cfg.rsi_len).replace(0,np.nan); x['rsi']=100-100/(1+rs)
    prev=c.shift(1); tr=pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1); x['atr']=_rma(tr,cfg.atr_len)
    mid=c.rolling(cfg.bb_len).mean(); sd=c.rolling(cfg.bb_len).std(ddof=0); x['bb_mid']=mid; x['bb_lo']=mid-cfg.bb_std*sd; x['bb_hi']=mid+cfg.bb_std*sd
    up=h.diff(); dn=-l.diff(); plus=up.where((up>dn)&(up>0),0.0); minus=dn.where((dn>up)&(dn>0),0.0); atr=x.atr.replace(0,np.nan)
    pdi=100*_rma(plus,cfg.adx_len)/atr; mdi=100*_rma(minus,cfg.adx_len)/atr; dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan); x['adx']=_rma(dx,cfg.adx_len)
    x['vol_med']=x.volume.rolling(30).median(); x['ema_slope']=x.ema_slow/x.ema_slow.shift(6)-1
    return x

def signal_row(r,cfg,costs):
    if any(pd.isna(r.get(k)) for k in ['atr','rsi','adx','bb_lo','ema_fast','ema_slow']) or r.atr<=0 or r.close<=0: return None
    if r.adx<=cfg.range_adx_max and r.close<=r.bb_lo and r.rsi<=cfg.range_rsi_buy: regime='RANGE'; target=r.close+cfg.target_atr_range*r.atr
    elif r.adx>=cfg.trend_adx_min and r.ema_fast>r.ema_slow and r.ema_slope>0 and cfg.trend_rsi_buy_min<=r.rsi<=cfg.trend_rsi_buy_max and r.close<=r.ema_fast*1.002: regime='TREND'; target=r.close+cfg.target_atr_trend*r.atr
    else: return None
    stop=r.close-cfg.stop_atr*r.atr
    if (target-r.close)/r.close < cfg.min_edge_multiple_of_cost*costs.estimated_round_trip_rate(): return None
    return 'BUY',regime,stop,target

def load_ohlcv(path):
    df=pd.read_csv(path); df.columns=[c.strip().lower() for c in df.columns]; df=df.rename(columns={'date':'timestamp','datetime':'timestamp','time':'timestamp'})
    req=['timestamp','open','high','low','close','volume']; miss=set(req)-set(df.columns)
    if miss: raise ValueError(f'Missing columns: {sorted(miss)}')
    df=df[req].copy(); df.timestamp=pd.to_datetime(df.timestamp,utc=True)
    for c in req[1:]: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.dropna().sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

def backtest(df,capital,cfg,costs):
    x=add_features(df,cfg).reset_index(drop=True); cash=float(capital); trades=[]; i=max(cfg.ema_slow,cfg.bb_len,cfg.adx_len)+8
    while i<len(x)-1:
        sig=signal_row(x.iloc[i],cfg,costs)
        if sig is None: i+=1; continue
        _,regime,_,_=sig; eb=x.iloc[i+1]; entry=float(eb.open); atr=float(x.iloc[i].atr); stop=entry-cfg.stop_atr*atr; target=entry+(cfg.target_atr_range if regime=='RANGE' else cfg.target_atr_trend)*atr
        qty=min((cash*cfg.risk_per_trade)/max(entry-stop,1e-9),(cash*cfg.max_notional_fraction)/(entry*(1+costs.fee_rate(costs.entry_is_maker))))
        if qty<=0: i+=1; continue
        j=i+1; end=min(j+cfg.max_hold_bars,len(x)-1); reason='TIME'; exit_px=float(x.iloc[end].close)
        for k in range(j,end+1):
            b=x.iloc[k]
            if b.low<=stop: exit_px=stop; reason='STOP'; end=k; break
            if b.high>=target: exit_px=target; reason='TARGET'; end=k; break
        fees=entry*qty*costs.fee_rate(costs.entry_is_maker)+exit_px*qty*costs.fee_rate(costs.exit_is_maker); slip=(entry+exit_px)*qty*costs.slippage_bps/10000.0
        gross=(exit_px-entry)*qty; net=gross-fees-slip; notional=entry*qty; cash+=net
        trades.append(Trade(x.iloc[i].timestamp,eb.timestamp,x.iloc[end].timestamp,regime,entry,exit_px,qty,stop,target,reason,gross,fees,slip,net,net/notional if notional else 0,end-j+1)); i=end+1
    ledger=pd.DataFrame([asdict(t) for t in trades]); return ledger,performance(ledger,capital,cash)

def performance(t,initial,final):
    if t.empty: return {'trades':0,'initial_capital':initial,'final_capital':final,'net_pnl':0.0,'return_pct':0.0}
    wins=t.net_pnl>0; gw=t.loc[wins,'net_pnl'].sum(); gl=-t.loc[~wins,'net_pnl'].sum(); eq=initial+t.net_pnl.cumsum(); dd=(eq-eq.cummax())/eq.cummax()
    return {'trades':int(len(t)),'win_rate':float(wins.mean()),'profit_factor':float(gw/gl) if gl>0 else math.inf,'avg_net_pnl':float(t.net_pnl.mean()),'median_net_pnl':float(t.net_pnl.median()),'fees_paid':float(t.fees.sum()),'slippage_paid':float(t.slippage.sum()),'initial_capital':float(initial),'final_capital':float(final),'net_pnl':float(final-initial),'return_pct':float((final/initial-1)*100),'max_drawdown_pct':float(dd.min()*100),'target_rate':float((t.exit_reason=='TARGET').mean()),'stop_rate':float((t.exit_reason=='STOP').mean())}
