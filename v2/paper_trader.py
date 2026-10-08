from __future__ import annotations
import argparse,json,time
from pathlib import Path
from datetime import datetime,timezone
from .live_signal import current_signal

def load_state(path,capital):
    if path.exists():
        return json.loads(path.read_text())
    return {"cash":float(capital),"qty":0.0,"entry_price":None,"realized_pnl":0.0,"trades":0}

def save_state(path,state):
    path.write_text(json.dumps(state,indent=2))

def mark(state,price):
    equity=state["cash"]+state["qty"]*price
    unrealized=0.0 if not state["qty"] or state["entry_price"] is None else state["qty"]*(price-state["entry_price"])
    return equity,unrealized

def step(state,rec,risk_fraction=0.25):
    price=float(rec["close"])
    action="HOLD"
    if state["qty"]<=0 and rec.get("signal")=="BUY":
        spend=max(0.0,state["cash"]*risk_fraction)
        qty=spend/price if price>0 else 0.0
        if qty>0:
            state["cash"]-=qty*price; state["qty"]=qty; state["entry_price"]=price; action="PAPER_BUY"
    elif state["qty"]>0:
        stop=float(rec.get("stop",0) or 0); target=float(rec.get("target",0) or 0)
        if (stop and price<=stop) or (target and price>=target):
            proceeds=state["qty"]*price
            pnl=state["qty"]*(price-state["entry_price"])
            state["cash"]+=proceeds; state["realized_pnl"]+=pnl; state["qty"]=0.0; state["entry_price"]=None; state["trades"]+=1
            action="PAPER_EXIT"
    equity,unrealized=mark(state,price)
    return action,equity,unrealized

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--symbol",default="BTCUSDC"); p.add_argument("--interval",default="5m")
    p.add_argument("--maker-bps",type=float,default=0); p.add_argument("--taker-bps",type=float,default=2); p.add_argument("--slippage-bps",type=float,default=1)
    p.add_argument("--capital",type=float,default=100000); p.add_argument("--risk-fraction",type=float,default=0.25)
    p.add_argument("--once",action="store_true"); p.add_argument("--log",default="paper_signals.jsonl"); p.add_argument("--state",default="paper_state.json")
    a=p.parse_args(); log=Path(a.log); sp=Path(a.state); state=load_state(sp,a.capital)
    while True:
        try:
            rec=current_signal(a.symbol,a.interval,a.maker_bps,a.taker_bps,a.slippage_bps)
            action,equity,unrealized=step(state,rec,a.risk_fraction)
            rec.update(observed_at=datetime.now(timezone.utc).isoformat(),paper_action=action,paper_cash=state["cash"],paper_qty=state["qty"],paper_equity=equity,unrealized_pnl=unrealized,realized_pnl=state["realized_pnl"],closed_trades=state["trades"])
            print(json.dumps(rec)); log.open("a").write(json.dumps(rec)+"\n"); save_state(sp,state)
        except Exception as e:
            print(json.dumps({"error":str(e),"observed_at":datetime.now(timezone.utc).isoformat()}))
        if a.once: break
        time.sleep(300)
if __name__=="__main__": main()
