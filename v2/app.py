from __future__ import annotations
import pandas as pd
import streamlit as st
import sys
from pathlib import Path

# Streamlit runs this file as a script; include the repository root for package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v2.binance_client import BinanceUSClient
from v2.engine import CostModel, StrategyConfig, add_features, backtest, signal_row
from v2.scalp import ScalpPosition, rebuy_metrics, eth_reentry_plan

st.set_page_config(page_title="305 Crypto Scalping Agent V2", page_icon="₿", layout="wide")
st.title("₿ 305 Crypto Scalping Agent V2")
st.caption("BTC-led regime console • BTC / ETH / ADA • paper/validation mode • LIVE MONEY LOCKED")

with st.sidebar:
    st.header("Trading model")
    symbol = st.selectbox("Chart market", ["BTCUSDC","ETHUSDC","ADAUSDC","BTCUSD","ETHUSD","ADAUSD"])
    interval = st.selectbox("Timeframe", ["5m","15m","1h"], index=0)
    capital = st.number_input("Backtest capital ($)", 1000.0, 10000000.0, 100000.0, 1000.0)
    maker = st.number_input("Maker fee (bps)", 0.0, 100.0, 0.0, 0.1)
    taker = st.number_input("Taker fee (bps)", 0.0, 100.0, 2.0, 0.1)
    slip = st.number_input("Slippage (bps / side)", 0.0, 50.0, 1.0, 0.1)
    st.error("LIVE MONEY: LOCKED")
    st.caption("No order-placement control is exposed. Validate and paper trade first.")

costs=CostModel(maker,taker,slip); cfg=StrategyConfig()

@st.cache_data(ttl=30, show_spinner=False)
def market_frame(sym,tf,limit=1000):
    raw=BinanceUSClient().klines(sym,tf,limit)
    cols=["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    d=pd.DataFrame(raw,columns=cols)
    return pd.DataFrame({"timestamp":pd.to_datetime(d.open_time,unit="ms",utc=True),"open":pd.to_numeric(d.open),"high":pd.to_numeric(d.high),"low":pd.to_numeric(d.low),"close":pd.to_numeric(d.close),"volume":pd.to_numeric(d.volume)})

def snapshot(sym):
    try:
        d=market_frame(sym,interval,500); f=add_features(d,cfg); r=f.iloc[-2]; sig=signal_row(r,cfg,costs)
        regime=sig[1] if sig else ("TREND DOWN" if r.ema_fast<r.ema_slow and r.ema_slope<0 else "WAIT")
        return d,f,r,sig,regime,None
    except Exception as exc:
        return pd.DataFrame(),pd.DataFrame(),None,None,"DATA ERROR",str(exc)

st.subheader("Market monitor")
cols=st.columns(3)
snapshots={}
for i,sym in enumerate(["BTCUSDC","ETHUSDC","ADAUSDC"]):
    d,f,r,sig,regime,err=snapshot(sym); snapshots[sym]=(d,f,r,sig,regime)
    with cols[i]:
        st.markdown(f"**{sym.replace('USDC','')}**")
        if r is None: st.error(err)
        else:
            st.metric("Price",f"${float(r.close):,.4f}" if float(r.close)<10 else f"${float(r.close):,.2f}")
            st.write(f"Regime: **{regime}**")
            st.write(f"RSI {float(r.rsi):.1f} • ADX {float(r.adx):.1f}")

st.subheader("Your two capital books")
a,b,c,d=st.columns(4)
a.metric("BTC accumulation pool","$102,423")
b.metric("ETH scalp exit","$2,577")
c.metric("ETH sold","6.692951 ETH")
eth=ScalpPosition("ETHUSDC",6.692951,2577.0,6.692951*2577.0)
d.metric("ETH scalp pool",f"${eth.net_proceeds:,.0f}")

st.markdown("**BTC accumulation pool — strategy levels are signal-driven**")
st.info("No static BTC buy ladder is shown. Use the live BTC regime, backtest evidence, and paper results below before changing accumulation levels.")

st.markdown("**Open ETH → USDC scalp**")
eth_px=float(snapshots["ETHUSDC"][2].close) if snapshots["ETHUSDC"][2] is not None else 2577.0
m=rebuy_metrics(eth,eth_px,maker)
x,y,z=st.columns(3)
x.metric("Current ETH",f"${eth_px:,.2f}")
y.metric("ETH if rebought now",f"{m['rebuy_quantity']:.6f}",f"{m['coin_gain']:+.6f} ETH")
z.metric("Coin accumulation",f"{m['coin_gain_pct']:+.2f}%")
st.dataframe(pd.DataFrame(eth_reentry_plan(eth)),use_container_width=True,hide_index=True)
st.caption("Re-entry ladder is a paper plan, not an automatic order. BTC structure remains the gating signal for ETH/ADA scalps.")

st.subheader("Selected market chart")
df,feat,row,sig,regime,err=snapshot(symbol)
if row is not None:
    chart=feat[["timestamp","close","ema_fast","ema_slow"]].dropna().tail(300).set_index("timestamp")
    st.line_chart(chart,height=330)
    st.write(f"Action: **{'LONG SETUP' if sig else 'WAIT'}** • Regime: **{regime}**")
    ledger,perf=backtest(df,capital,cfg,costs)
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("Trades",perf.get("trades",0)); m2.metric("Win rate",f"{perf.get('win_rate',0)*100:.1f}%")
    pf=perf.get("profit_factor",0); m3.metric("Profit factor","∞" if pf==float("inf") else f"{pf:.2f}")
    m4.metric("Net P&L",f"${perf.get('net_pnl',0):,.2f}"); m5.metric("Max DD",f"{perf.get('max_drawdown_pct',0):.2f}%")
else:
    st.error(err)

st.subheader("Validation gate")
trades=int(perf.get("trades",0)) if row is not None else 0
pf=float(perf.get("profit_factor",0)) if row is not None else 0
positive=float(perf.get("net_pnl",0))>0 if row is not None else False
checks=pd.DataFrame([
    ["Deterministic engine","PASS","No discretionary order path"],
    ["Multi-asset monitor","PASS","BTC / ETH / ADA"],
    ["After-cost P&L","PASS" if positive else "NOT YET","Must be positive"],
    ["Profit factor","PASS" if pf>=1.20 else "NOT YET",f"{pf:.2f} / 1.20 minimum"],
    ["Sample size","PASS" if trades>=200 else "NOT YET",f"{trades} / 200 minimum"],
    ["Live execution","LOCKED","Requires walk-forward + paper validation"],
],columns=["Gate","Status","Evidence"])
st.dataframe(checks,use_container_width=True,hide_index=True)
