import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
sys.path.append(os.path.abspath(os.path.join('Personal')))
sys.path.append(os.path.abspath(os.path.join('Personal','financebot','src')))

import financebot.src.config as config
from financebot.src.model import TransformerModel
from livedata import plotlivedata
from simulatedata import plotsimulateddata
from plot import initfig,plotbuttons

def main(symbols = ["TSLA"],num_frames=10**5,interval=10**3,livedata=True,whattoshow=1):
    model = TransformerModel().to(config.DEVICE)
    model_path=os.path.join(os.getcwd(),'Personal' ,'financebot', 'data', 'modelsave_multistep.pth')
    if not os.path.isfile(model_path):
        print("Can't find model data path")
        exit(0)
    model.load_model(model_path)

    fig, ax,artists =initfig(symbols[0],whattoshow)
    if livedata:
        pdata=plotlivedata(model,whattoshow,ax,symbols,artists)
    else:
        pdata=plotsimulateddata(model,whattoshow,ax,symbols,artists)
    plotbuttons(fig,pdata)

    anim = animation.FuncAnimation(fig, pdata.updatedraw,frames=num_frames, interval=interval, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    main(livedata=True)