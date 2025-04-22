

from datetime import timedelta, datetime
from dataclass import plotdata,gethistoricalstockbars

class plotsimulateddata(plotdata):
    def __init__(self,model,whattoshow,ax,symbols,artists):
        global streamdata
        super().__init__(model,whattoshow,ax,symbols,artists)
        end = (datetime.now().astimezone() - timedelta(minutes=20))
        start = end - timedelta(days=15)
        streamdata=gethistoricalstockbars(symbols,start,end,self.timeframe)
    def getdata(self):
        return streamdata[:min(self.t+self.SEQ_LEN,len(streamdata)-1)]
    def updatedata(self,framei):
        if self.animation_is_running:
            self.t+=1
            self.alldatapoints=self.getdata()
            self.allDates=[dat[0] for dat in  self.alldatapoints]
            self.ally=[dat[self.whattoshow] for dat in  self.alldatapoints]
        
        tmax=min(self.t+self.SEQ_LEN+self.WindowXshift,len(self.alldatapoints))
        tmin=self.t+self.WindowXshift
        self.currentdatapoints=self.alldatapoints[tmin:tmax]
        self.Dates=[dat[0] for dat in self.currentdatapoints]
        self.y=[dat[self.whattoshow] for dat in self.currentdatapoints]
        self.updatepred(self.whattoshow)