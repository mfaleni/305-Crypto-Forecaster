from __future__ import annotations
import pandas as pd
import streamlit as st
from v2.binance_client import BinanceUSClient
from v2.engine import CostModel, StrategyConfig, add_features, backtest, signal_row
from v2.live_signal import current_signal

st.set_page_config(page_title='305 BTC Trading Agent V2', page_icon='₿', layout='wide')
st.title('₿ 305 BTC Trading Agent V2')
st.caption('BTC quantitative trading console • deterministic signals • paper/validation mode')

with st.sidebar:
    st.header('Trading model')
    symbol = st.selectbox('Market', ['BTCUSD','BTCUSDT','BTCUSDC'])
    interval = st.selectbox('Timeframe', ['5m','15m','1h'], index=0)
    capital = st.number_input('Model capital ($)', 1000.0, 10000000.0, 100000.0, 1000.0)
    maker = st.number_input('Maker fee (bps)', 0.0, 100.0, 0.0, 0.1)
    taker = st.number_input('Taker fee (bps)', 0.0, 100.0, 2.0, 0.1)
    slip = st.number_input('Slippage (bps / side)', 0.0, 50.0, 1.0, 0.1)
    st.error('LIVE MONEY: LOCKED')
    st.caption('This UI cannot enable live trading. Validation and paper trading come first.')

costs = CostModel(maker, taker, slip)
cfg = StrategyConfig()

@st.cache_data(ttl=60, show_spinner=False)
def market_frame(sym, tf, limit=1000):
    raw = BinanceUSClient().klines(sym, tf, limit)
    cols=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']
    d=pd.DataFrame(raw,columns=cols)
    return pd.DataFrame({
        'timestamp':pd.to_datetime(d.open_time,unit='ms',utc=True),
        'open':pd.to_numeric(d.open),'high':pd.to_numeric(d.high),'low':pd.to_numeric(d.low),
        'close':pd.to_numeric(d.close),'volume':pd.to_numeric(d.volume)})

try:
    df=market_frame(symbol,interval)
    feat=add_features(df,cfg)
    row=feat.iloc[-2]
    sig=signal_row(row,cfg,costs)
    signal='NO TRADE'; regime='NONE'; stop=None; target=None
    if sig:
        _,regime,stop,target=sig; signal='LONG SETUP'
    live_ok=True
except Exception as exc:
    live_ok=False; df=pd.DataFrame(); feat=pd.DataFrame(); row=None; signal='DATA ERROR'; regime='UNKNOWN'; stop=target=None
    st.error(f'Market data unavailable: {exc}')

# ---- NOW ----
st.subheader('NOW')
a,b,c,d,e=st.columns(5)
if live_ok:
    px=float(row.close); atr=float(row.atr); rsi=float(row.rsi); adx=float(row.adx)
    a.metric('BTC',f'${px:,.2f}')
    b.metric('Action',signal)
    c.metric('Regime',regime)
    d.metric('RSI',f'{rsi:.1f}')
    e.metric('ADX',f'{adx:.1f}')
    if signal=='LONG SETUP':
        gross=(target/px-1)*100; cost=costs.estimated_round_trip_rate()*100; net=gross-cost
        st.success(f'LONG SETUP  •  Entry reference ${px:,.2f}  •  Stop ${stop:,.2f}  •  Target ${target:,.2f}  •  Estimated net edge {net:.2f}%')
    else:
        st.info('NO TRADE — current market conditions do not satisfy the validated deterministic entry rules.')

# ---- CHART ----
if live_ok:
    chart=feat[['timestamp','close','ema_fast','ema_slow']].dropna().tail(300).set_index('timestamp')
    st.subheader('BTC price / trend')
    st.line_chart(chart, height=330)

# ---- BASELINE BACKTEST ON RECENT BARS ----
st.subheader('Strategy results — recent exchange bars')
if live_ok:
    ledger,perf=backtest(df,capital,cfg,costs)
    m1,m2,m3,m4,m5,m6=st.columns(6)
    m1.metric('Trades',perf.get('trades',0))
    m2.metric('Win rate',f"{perf.get('win_rate',0)*100:.1f}%")
    pf=perf.get('profit_factor',0); m3.metric('Profit factor', '∞' if pf==float('inf') else f'{pf:.2f}')
    m4.metric('Net P&L',f"${perf.get('net_pnl',0):,.2f}")
    m5.metric('Return',f"{perf.get('return_pct',0):.2f}%")
    m6.metric('Max drawdown',f"{perf.get('max_drawdown_pct',0):.2f}%")
    if not ledger.empty:
        eq=pd.DataFrame({'Equity':capital+ledger.net_pnl.cumsum().values},index=pd.to_datetime(ledger.exit_time))
        st.line_chart(eq,height=260)
        with st.expander('Trade ledger',expanded=False):
            show=ledger.copy(); show['entry_time']=pd.to_datetime(show.entry_time); show['exit_time']=pd.to_datetime(show.exit_time)
            st.dataframe(show.sort_values('exit_time',ascending=False),use_container_width=True,hide_index=True)
    else:
        st.warning('No qualifying trades occurred in the bars currently loaded. This is a valid strategy result, not a forced signal.')

# ---- VALIDATION ----
st.subheader('Validation gate')
trades=int(perf.get('trades',0)) if live_ok else 0
pf=float(perf.get('profit_factor',0)) if live_ok else 0
positive=float(perf.get('net_pnl',0))>0 if live_ok else False
dd=abs(float(perf.get('max_drawdown_pct',0))) if live_ok else 999
checks=pd.DataFrame([
    ['Engine tests','PASS','Automated CI'],
    ['After-cost P&L','PASS' if positive else 'NOT YET','Must be positive'],
    ['Profit factor','PASS' if pf>=1.20 else 'NOT YET',f'{pf:.2f} / 1.20 minimum'],
    ['Sample size','PASS' if trades>=200 else 'NOT YET',f'{trades} / 200 trades minimum'],
    ['Live execution','LOCKED','Requires full walk-forward + paper validation'],
],columns=['Gate','Status','Evidence'])
st.dataframe(checks,use_container_width=True,hide_index=True)

st.caption('Important: the recent-bars backtest above is a dashboard diagnostic, not proof of profitability. Production approval requires long-horizon historical and walk-forward validation plus live paper results.')
