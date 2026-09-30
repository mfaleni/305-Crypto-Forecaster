import pandas as pd
from v2.engine import CostModel,StrategyConfig,add_features
def test_cost_math():
    c=CostModel(maker_fee_bps=50,taker_fee_bps=60,slippage_bps=1,entry_is_maker=True,exit_is_maker=True); assert abs(c.estimated_round_trip_rate()-0.0102)<1e-12
def test_features_no_future_fill():
    n=120; df=pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=n,freq='5min',tz='UTC'),'open':range(100,220),'high':range(101,221),'low':range(99,219),'close':range(100,220),'volume':[10]*n}); x=add_features(df,StrategyConfig()); assert 'rsi' in x and 'atr' in x and len(x)==n
