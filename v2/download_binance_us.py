import argparse,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
import requests,pandas as pd
BASE='https://api.binance.us/api/v3/klines'; COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']
def ms(dt): return int(dt.timestamp()*1000)
def parse_dt(s):
    if s.lower()=='now': return datetime.now(timezone.utc)
    d=datetime.fromisoformat(s.replace('Z','+00:00')); return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
def download(symbol,interval,start,end):
    cur=ms(start); end_ms=ms(end); rows=[]; sess=requests.Session()
    while cur<end_ms:
        r=sess.get(BASE,params={'symbol':symbol,'interval':interval,'startTime':cur,'endTime':end_ms,'limit':1000},timeout=30); r.raise_for_status(); batch=r.json()
        if not batch: break
        rows.extend(batch); nxt=int(batch[-1][0])+1
        if nxt<=cur: break
        cur=nxt; time.sleep(.08)
    df=pd.DataFrame(rows,columns=COLS)
    if df.empty: return df
    out=pd.DataFrame({'timestamp':pd.to_datetime(df.open_time,unit='ms',utc=True),'open':pd.to_numeric(df.open),'high':pd.to_numeric(df.high),'low':pd.to_numeric(df.low),'close':pd.to_numeric(df.close),'volume':pd.to_numeric(df.volume),'quote_volume':pd.to_numeric(df.quote_volume),'trades':pd.to_numeric(df.trades),'taker_buy_base':pd.to_numeric(df.taker_buy_base),'taker_buy_quote':pd.to_numeric(df.taker_buy_quote)})
    return out.drop_duplicates('timestamp').sort_values('timestamp')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--symbol',default='BTCUSD'); p.add_argument('--interval',default='5m'); p.add_argument('--days',type=int,default=365); p.add_argument('--start'); p.add_argument('--end',default='now'); p.add_argument('--out',default='btc_us_5m.csv'); a=p.parse_args()
    end=parse_dt(a.end); start=parse_dt(a.start) if a.start else end-timedelta(days=a.days); df=download(a.symbol,a.interval,start,end); Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False); print(f'wrote {len(df):,} bars to {a.out}')
