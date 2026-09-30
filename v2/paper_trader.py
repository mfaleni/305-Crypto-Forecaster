from __future__ import annotations
import argparse,json,time
from pathlib import Path
from datetime import datetime,timezone
from .live_signal import current_signal
def main():
    p=argparse.ArgumentParser(); p.add_argument('--symbol',default='BTCUSD'); p.add_argument('--interval',default='5m'); p.add_argument('--maker-bps',type=float,default=0); p.add_argument('--taker-bps',type=float,default=2); p.add_argument('--slippage-bps',type=float,default=1); p.add_argument('--once',action='store_true'); p.add_argument('--log',default='paper_signals.jsonl'); a=p.parse_args(); path=Path(a.log)
    while True:
        try: rec=current_signal(a.symbol,a.interval,a.maker_bps,a.taker_bps,a.slippage_bps); rec['observed_at']=datetime.now(timezone.utc).isoformat(); print(json.dumps(rec)); path.open('a').write(json.dumps(rec)+'\n')
        except Exception as e: print(json.dumps({'error':str(e),'observed_at':datetime.now(timezone.utc).isoformat()}))
        if a.once: break
        time.sleep(300)
if __name__=='__main__': main()
