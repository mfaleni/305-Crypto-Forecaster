from __future__ import annotations
import pandas as pd
from .binance_client import BinanceUSClient
from .engine import add_features,signal_row,StrategyConfig,CostModel
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']
def current_signal(symbol='BTCUSD',interval='5m',maker_bps=0.0,taker_bps=2.0,slippage_bps=1.0):
    c=BinanceUSClient(); d=pd.DataFrame(c.klines(symbol,interval,500),columns=COLS)
    df=pd.DataFrame({'timestamp':pd.to_datetime(d.open_time,unit='ms',utc=True),'open':pd.to_numeric(d.open),'high':pd.to_numeric(d.high),'low':pd.to_numeric(d.low),'close':pd.to_numeric(d.close),'volume':pd.to_numeric(d.volume)})
    cfg=StrategyConfig(); costs=CostModel(maker_bps,taker_bps,slippage_bps); r=add_features(df,cfg).iloc[-2]
    sig=signal_row(r,cfg,costs); out={'timestamp':str(r.timestamp),'close':float(r.close),'rsi':float(r.rsi),'adx':float(r.adx),'atr':float(r.atr),'signal':'NO_TRADE'}
    if sig:
        _,regime,stop,target=sig; out.update(signal='BUY',regime=regime,stop=float(stop),target=float(target),gross_edge_pct=float((target/r.close-1)*100),estimated_cost_pct=float(costs.estimated_round_trip_rate()*100))
    return out
