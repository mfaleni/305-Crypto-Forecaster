import streamlit as st
from v2.live_signal import current_signal
st.set_page_config(page_title='305 BTC Trading Agent V2',layout='wide'); st.title('305 BTC Trading Agent V2'); st.caption('Quantitative signal engine • BTC only • live trading disabled by default')
with st.sidebar:
    symbol=st.selectbox('Pair',['BTCUSD','BTCUSDT','BTCUSDC']); interval=st.selectbox('Bar interval',['5m','15m','1h']); maker=st.number_input('Maker fee (bps)',0.0,100.0,0.0,.1); taker=st.number_input('Taker fee (bps)',0.0,100.0,2.0,.1); slip=st.number_input('Slippage assumption (bps)',0.0,50.0,1.0,.1); st.warning('Execution is PAPER/READ-ONLY unless live trading is explicitly enabled outside this UI.')
if st.button('Refresh signal',type='primary'):
    try: st.session_state['sig']=current_signal(symbol,interval,maker,taker,slip)
    except Exception as e: st.error(str(e))
s=st.session_state.get('sig')
if s:
    a,b,c,d=st.columns(4); a.metric('Signal',s['signal']); b.metric('BTC close',f"${s['close']:,.2f}"); c.metric('RSI',f"{s['rsi']:.1f}"); d.metric('ADX',f"{s['adx']:.1f}"); st.json(s)
st.divider(); st.subheader('Validation gate'); st.write('Live execution remains blocked until walk-forward and paper-trading thresholds are met. No LLM-generated trade decisions are used.')
