import pandas as pd
from v2.engine import CostModel,StrategyConfig,add_features
from v2.scalp import ScalpPosition, rebuy_metrics, eth_reentry_plan, btc_reentry_plan

def test_cost_math():
    c=CostModel(maker_fee_bps=50,taker_fee_bps=60,slippage_bps=1,entry_is_maker=True,exit_is_maker=True)
    assert abs(c.estimated_round_trip_rate()-0.0102)<1e-12

def test_features_no_future_fill():
    n=120
    df=pd.DataFrame({"timestamp":pd.date_range("2026-01-01",periods=n,freq="5min",tz="UTC"),"open":range(100,220),"high":range(101,221),"low":range(99,219),"close":range(100,220),"volume":[10]*n})
    x=add_features(df,StrategyConfig())
    assert "rsi" in x and "atr" in x and len(x)==n

def test_eth_scalp_rebuy_accumulates_coin_below_exit():
    p=ScalpPosition("ETHUSDC",6.692951,2577.0,6.692951*2577.0)
    m=rebuy_metrics(p,2500.0,0.0)
    assert m["rebuy_quantity"] > p.quantity_sold
    assert m["coin_gain"] > 0

def test_eth_plan_allocates_entire_scalp_pool():
    p=ScalpPosition("ETHUSDC",6.692951,2577.0,6.692951*2577.0)
    plan=eth_reentry_plan(p)
    assert abs(sum(x["fraction"] for x in plan)-1.0)<1e-12
    assert abs(sum(x["allocation_usdc"] for x in plan)-p.net_proceeds)<1e-6

def test_btc_plan_keeps_expected_tranches():
    plan=btc_reentry_plan()
    assert [x["tranche_usdc"] for x in plan]==[20000.0,25000.0,30000.0,27400.0]
