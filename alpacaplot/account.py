import os 
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")
class alpacaAccount:
    def __init__(self,symbol="TSLA",notional=200,time_in_force=TimeInForce.DAY):
        self.trading_client=TradingClient(api_key, secret_key, paper=True)
        self.market_order_data = MarketOrderRequest(symbol=symbol,notional=notional,side=OrderSide.BUY,time_in_force=time_in_force)
        self.details=self.trading_client.get_account()
        self.portfolio=self.trading_client.get_all_positions()
    def order(self):
        self.market_order = self.trading_client.submit_order(order_data=self.market_order_data)
    def update(self):
        self.details=self.trading_client.get_account()
        self.portfolio=self.trading_client.get_all_positions()
    def stats(self):
        return self.details.cash,[(pos.symbol,pos.unrealized_pl) for pos in self.portfolio]
    def currentworth(self):
        return self.trading_client.get_account().equity
    
from time import sleep

if __name__ == "__main__":
    account=alpacaAccount()
    stats=account.stats()
    while True:
        account.update()
        tempstats=account.stats()
        if stats!=tempstats:
            print(tempstats,end="\r")
            stats=tempstats
        sleep(1)
        
