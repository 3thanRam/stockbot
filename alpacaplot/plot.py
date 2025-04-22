import os

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.widgets import Button


DATA_TYPE = ['date','open','close','high','low']
api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")
streamdata=[]

def initfig(symbol,whattoshow):
    fig, ax = plt.subplots()
    ax.set_title(f"{symbol} Stock Data")
    ax.set_xlabel("Time")
    ax.set_ylabel("Data")
    ax.grid(True)
    ax.set_ylim(0, 500)
    line, = ax.plot([], [], lw=2, label=f"Live {DATA_TYPE[whattoshow-1]} Price", c='blue', zorder=10)
    predline, = ax.plot([], [], "--", lw=2, label=f"Predicted {DATA_TYPE[whattoshow]}", c='orange', zorder=10)
    nowline=datetime.now().astimezone()
    vline_input_start = ax.axvline(nowline, color='green', linestyle='--', lw=1.5, label=f'Input Start', zorder=5)
    vline_input_end = ax.axvline(nowline, color='green', linestyle='--', lw=1.5, label=f'Input End', zorder=5)
    vline_pred_end = ax.axvline(nowline, color='red', linestyle='-', lw=2, label=f'Prediction End', visible=False, zorder=5)
    artists = {
        'line': line, 'predline': predline,
        'vline_input_start': vline_input_start, 'vline_input_end': vline_input_end,
        'vline_pred_end': vline_pred_end
        }
    artists['line'].set_data([], [])
    artists['predline'].set_data([], [])
    artists['vline_input_start'].set_visible(True)
    artists['vline_input_end'].set_visible(True)
    artists['vline_pred_end'].set_visible(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    ax.legend(loc='upper left')
    return fig, ax, artists

class plotbuttons:
    
    def __init__(self,fig,pdata,buttonpositions=[[0.81, 0.005, 0.1, 0.04],[0.26, 0.005, 0.2, 0.04],[0.66, 0.005, 0.1, 0.04],[0.51, 0.005, 0.1, 0.04]]):
        self.pdata=pdata
        self.buttons=[]
        self.selected=0
        buttonfcts=[self.toggle_pause,self.select_vline,self.plus_action,self.minus_action]
        buttonlabels=["Pause","input shift","+","-"]
        for b,bpos in enumerate(buttonpositions):
            b_ax = fig.add_axes(bpos) 
            btn = Button(b_ax, buttonlabels[b])
            btn.on_clicked(buttonfcts[b])
            self.buttons.append(btn)
    def toggle_pause(self,event):
        if self.pdata.animation_is_running:
            self.buttons[0].label.set_text('Resume')
        else:
            self.buttons[0].label.set_text('Pause')
        self.pdata.animation_is_running=not self.pdata.animation_is_running
    def select_vline(self,event):
        self.selected=(self.selected+1)%2
        if self.selected==0:
            self.buttons[1].label.set_text('input shift')
        else:
            self.buttons[1].label.set_text('pred len')
    def plus_action(self,event):
        if self.selected==0:
            self.pdata.WindowXshift+=1
            self.pdata.WindowXshift=min(self.pdata.WindowXshift,0)
            self.pdata.WindowXshift=max(self.pdata.WindowXshift,-self.pdata.SEQ_LEN,-self.pdata.t)
        else:
            self.pdata.predictionL+=1
            self.pdata.predictionL=max(self.pdata.predictionL,1)
    def minus_action(self,event):
        if self.selected==0:
            self.pdata.WindowXshift-=1
            self.pdata.WindowXshift=min(self.pdata.WindowXshift,0)
            self.pdata.WindowXshift=max(self.pdata.WindowXshift,-self.pdata.SEQ_LEN,-self.pdata.t)
        else:
            self.pdata.predictionL-=1
            self.pdata.predictionL=max(self.pdata.predictionL,1)