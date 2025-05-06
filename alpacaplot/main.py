# alpacabot/main.py

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Get the script's directory and add the parent directory to the system path
# This allows importing custom modules from the financebot directory
plotpathpath=os.path.dirname(os.path.realpath(__file__))
botdir=plotpathpath.strip(plotpathpath.split("/")[-1])[:-1]
sys.path.append(botdir)

# Import application-specific modules and classes
import financebot.src.config as config # Configuration settings
from financebot.src.model import TransformerModel # The model used for analysis/prediction

# Import modules for plotting data
from livedata import plotlivedata # Handles plotting of live financial data
from simulatedata import plotsimulateddata # Handles plotting of simulated financial data
from plot import initfig,plotbuttons # Utility functions for plot initialization and interactive buttons

# Main function to set up and run the financial data visualization
# Initializes model, loads data (live or simulated), sets up plot, and starts animation
def main(symbols = ["TSLA"],num_frames=10**5,interval=10**3,livedata=True,whattoshow=1):
    # Initialize and load the pre-trained model
    model = TransformerModel().to(config.DEVICE)
    model_path=config.MODEL_SAVE_PATH
    if not os.path.isfile(model_path):
        print("Can't find model data path")
        exit(0)
    model.load_model(model_path)

    # Initialize the matplotlib figure and axes, and necessary plot artists
    fig, ax,artists =initfig(symbols[0],whattoshow)
    # Choose between live or simulated data plotting based on the 'livedata' flag
    if livedata:
        pdata=plotlivedata(model,whattoshow,ax,symbols,artists) # Setup for live data
    else:
        pdata=plotsimulateddata(model,whattoshow,ax,symbols,artists) # Setup for simulated data

    # Add interactive buttons to the plot
    plbt=plotbuttons(fig,pdata)

    # Create a matplotlib animation
    # It repeatedly calls pdata.updatedraw to update the plot data and artists
    anim = animation.FuncAnimation(fig, pdata.updatedraw,frames=num_frames, interval=interval, blit=False, repeat=False)
    # Display the plot window
    plt.show()

# Standard Python entry point
if __name__ == "__main__":
    # Run the main function when the script is executed directly
    main(livedata=False)