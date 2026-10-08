import argparse,json
from pathlib import Path
from v2.engine import load_ohlcv,backtest,StrategyConfig,CostModel
p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--capital',type=float,default=100000); p.add_argument('--maker-fee-bps',type=float,default=5); p.add_argument('--taker-fee-bps',type=float,default=10); p.add_argument('--slippage-bps',type=float,default=1); p.add_argument('--out',default='v2_results'); a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
df=load_ohlcv(a.csv); ledger,metrics=backtest(df,a.capital,StrategyConfig(),CostModel(a.maker_fee_bps,a.taker_fee_bps,a.slippage_bps)); ledger.to_csv(out/'trades.csv',index=False); (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
