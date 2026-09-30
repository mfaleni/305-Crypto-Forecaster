from __future__ import annotations
import os,time,hmac,hashlib,urllib.parse,requests
class BinanceUSClient:
    BASE='https://api.binance.us'
    def __init__(self,api_key=None,api_secret=None,timeout=15):
        self.key=api_key or os.getenv('BINANCE_US_API_KEY'); self.secret=api_secret or os.getenv('BINANCE_US_API_SECRET'); self.timeout=timeout; self.s=requests.Session()
    def public(self,path,params=None):
        r=self.s.get(self.BASE+path,params=params or {},timeout=self.timeout); r.raise_for_status(); return r.json()
    def signed(self,method,path,params=None):
        if not self.key or not self.secret: raise RuntimeError('Binance.US API credentials are not configured')
        p=dict(params or {}); p['timestamp']=int(time.time()*1000); q=urllib.parse.urlencode(p); p['signature']=hmac.new(self.secret.encode(),q.encode(),hashlib.sha256).hexdigest()
        r=self.s.request(method,self.BASE+path,params=p,headers={'X-MBX-APIKEY':self.key},timeout=self.timeout); r.raise_for_status(); return r.json()
    def ticker(self,symbol='BTCUSD'): return self.public('/api/v3/ticker/bookTicker',{'symbol':symbol})
    def klines(self,symbol='BTCUSD',interval='5m',limit=500): return self.public('/api/v3/klines',{'symbol':symbol,'interval':interval,'limit':limit})
    def account(self): return self.signed('GET','/api/v3/account')
    def commissions(self,symbol='BTCUSD'): return self.signed('GET','/api/v3/account/commission',{'symbol':symbol})
    def test_limit_maker(self,symbol,side,quantity,price): return self.signed('POST','/api/v3/order/test',{'symbol':symbol,'side':side,'type':'LIMIT_MAKER','quantity':quantity,'price':price})
    def place_limit_maker(self,symbol,side,quantity,price):
        if os.getenv('ENABLE_LIVE_TRADING')!='I_UNDERSTAND_AND_ENABLE_LIVE_TRADING': raise RuntimeError('LIVE TRADING BLOCKED pending validation and explicit review.')
        return self.signed('POST','/api/v3/order',{'symbol':symbol,'side':side,'type':'LIMIT_MAKER','quantity':quantity,'price':price})
