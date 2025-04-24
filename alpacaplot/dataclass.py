import numpy as np
import os 
import sys

from datetime import timedelta,timezone
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


import financebot.src.config as config
from financebot.src.predictor import infer_realdata
from account import alpacaAccount


api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")


def format_data(data):
    ts = data.timestamp
    ts = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return [ts,data.open,data.close,data.high,data.low]

def gethistoricalstockbars(symbols,start,end,timeframe):
    client=StockHistoricalDataClient(api_key, secret_key)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=timeframe, start=start, end=end)
    stockbars = client.get_stock_bars(req).data[symbols[0]]
    return [format_data(data) for data in stockbars]

class plotdata:
    def __init__(self,model,whattoshow,ax,symbols,artists):
        self.artists=artists
        self.timeframe=TimeFrame(1, TimeFrameUnit.Hour)
        self.model=model
        self.Dates=[]
        self.pred_dates=[]
        self.alldatapoints=[]
        self.allDates=[]
        self.ally=[]
        self.preds=[]
        self.currentdatapoints=[]
        self.WindowXshift=0
        self.whattoshow=whattoshow
        self.SEQ_LEN=config.SEQ_LEN
        self.predictionL=config.N_PRED_INFERENCE
        self.t=0
        self.ax=ax
        self.animation_is_running=True
        self.account=alpacaAccount()
    def getdata(self):
        return []
    def getdates(self,dateslist,numb):
        return [dateslist[-1]+timedelta(hours=1)*(n+1) for n in range(numb)]
    def updatepred(self,whattoshow):
        if len(self.Dates)>=self.SEQ_LEN:
            preds = infer_realdata(self.model, np.array(self.currentdatapoints).T,self.predictionL)[:,whattoshow-1]
            self.preds=np.insert(preds,0,self.y[-1])
            self.pred_dates=[self.Dates[-1]]+self.getdates(self.Dates,self.predictionL)
            
    def resize(self,ax):
        
        if self.pred_dates:
            xmax=self.pred_dates[-1]
        else:
            xmax=self.Dates[-1]
        xmin=self.Dates[0]
        dt=timedelta(hours=1)

        ax.set_xlim(xmin-dt,xmax+dt)
        ax.set_ylim(0, 1.1*max(self.y))
        ax.legend(loc='upper left')
    def redraw(self,ax):
        self.inpStart=self.Dates[0]
        self.artists['line'].set_label(f"Open Price: i={self.t}")
        self.artists['vline_input_start'].set_xdata([self.inpStart])
        self.artists['vline_input_start'].set_label(f"Input start: t=ti-{self.SEQ_LEN-self.WindowXshift}")
        if len(self.Dates)>=self.SEQ_LEN:
            self.inpEnd=self.Dates[self.SEQ_LEN-1]
            self.predEnd=self.pred_dates[-1]
            self.artists['vline_input_end'].set_xdata([self.inpEnd])
            self.artists['vline_pred_end'].set_xdata([self.predEnd])
            self.artists['vline_pred_end'].set_label(f"Pred End: L={self.predictionL}")
            self.artists['predline'].set_data(self.pred_dates,self.preds)
        self.resize(ax)
        self.artists['line'].set_data(self.allDates, self.ally)
    def updatedata(self,framei):
        return []
    def updatedraw(self,framei):
        self.updatedata(framei)
        self.redraw(self.ax)
        return list(self.artists.values())

    



