
import threading
from datetime import timedelta, datetime
from alpaca.data.live import StockDataStream
from dataclass import plotdata,gethistoricalstockbars,format_data,api_key,secret_key
import time

async def stock_data_stream_handler(data):
    print(data)
    streamdata.append(format_data(data))




class plotlivedata(plotdata):
    def __init__(self,model,whattoshow,ax,symbols,artists):
        global streamdata
        super().__init__(model,whattoshow,ax,symbols,artists)
        end = (datetime.now().astimezone() - timedelta(minutes=15))
        start = end - timedelta(weeks=1)
        streamdata=gethistoricalstockbars(symbols,start,end,self.timeframe)
        print(len(streamdata))
        stock_data_stream_client = StockDataStream(
            api_key, secret_key)
        
        stock_data_stream_client.subscribe_bars(stock_data_stream_handler, *symbols)
        self.t1 =threading.Thread(target=stock_data_stream_client.run)
        self.t1.start()
        
    def getdata(self):
        return list(streamdata)
    def updatedata(self,framei):
        self.account.update()
        if self.animation_is_running:
            alldatapoints=self.getdata()
            if self.alldatapoints!=alldatapoints:
                self.t+=1
                self.alldatapoints=alldatapoints
                self.allDates=[dat[0] for dat in  self.alldatapoints]
                self.ally=[dat[self.whattoshow] for dat in  self.alldatapoints]
        
        tmax=min(self.t+self.SEQ_LEN+self.WindowXshift,len(self.alldatapoints))
        tmin=self.t+self.WindowXshift
        self.currentdatapoints=self.alldatapoints[tmin:tmax]
        self.Dates=[dat[0] for dat in self.currentdatapoints]
        self.y=[dat[self.whattoshow] for dat in self.currentdatapoints]
        self.updatepred(self.whattoshow)

