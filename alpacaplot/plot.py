# alpacabot/plot.py

import os # For accessing environment variables

from datetime import datetime # For handling time
import matplotlib.pyplot as plt # For plotting
import matplotlib.dates as mdates # For formatting dates on plots

from matplotlib.widgets import Button,TextBox # For adding interactive widgets

# Define the types of data that can be plotted/shown
DATA_TYPE = ['date','open','close','high','low']
# Retrieve API keys from environment variables (used elsewhere, defined here)
api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")
# Global list likely intended to hold streaming data (currently unused in this snippet)
streamdata=[]

# Function to initialize the matplotlib figure, axes, and plot elements
# Sets up the basic plot structure, labels, limits, and adds main plot lines and vertical markers
def initfig(symbol,whattoshow):
    # Create a new figure and a set of axes
    fig, ax = plt.subplots()
    # Set plot title and axis labels
    ax.set_title(f"{symbol} Stock Data")
    ax.set_xlabel("Time")
    ax.set_ylabel("Data")
    ax.grid(True) # Add a grid
    ax.set_ylim(0, 500) # Set the initial y-axis limits
    # Add a text box widget to display information (e.g., account balance)
    textbox = TextBox(ax.inset_axes([1.2, 0.9, 0.3, 0.08]), "ACCOUNT_INFO",
                  initial="$")
    # Initialize the line artist for the actual live/simulated data
    line, = ax.plot([], [], lw=2, label=f"Live {DATA_TYPE[whattoshow-1]} Price", c='blue', zorder=10)
    # Initialize the line artist for the model's predictions
    predline, = ax.plot([], [], "--", lw=2, label=f"Predicted {DATA_TYPE[whattoshow]}", c='orange', zorder=10)
    # Get the current time
    nowline=datetime.now().astimezone()
    # Initialize vertical lines to indicate data windows (input start/end, prediction end)
    vline_input_start = ax.axvline(nowline, color='green', linestyle='--', lw=1.5, label=f'Input Start', zorder=5)
    vline_input_end = ax.axvline(nowline, color='green', linestyle='--', lw=1.5, label=f'Input End', zorder=5)
    vline_pred_end = ax.axvline(nowline, color='red', linestyle='-', lw=2, label=f'Prediction End', visible=False, zorder=5)
    # Store all plot artists in a dictionary for easy access and updating
    artists = {
        'line': line, 'predline': predline,
        'vline_input_start': vline_input_start, 'vline_input_end': vline_input_end,
        'vline_pred_end': vline_pred_end,"textbox":textbox
        }
    # Explicitly set data to empty initially (though plot() does this too)
    artists['line'].set_data([], [])
    artists['predline'].set_data([], [])
    # Ensure vertical lines are initially visible (they are often updated later)
    artists['vline_input_start'].set_visible(True)
    artists['vline_input_end'].set_visible(True)
    artists['vline_pred_end'].set_visible(True)
    # Configure date formatting for the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    # Automatically format the date labels on the x-axis to prevent overlap
    fig.autofmt_xdate()
    # Add a legend to identify the plot lines
    ax.legend(loc='upper left')
    # Return the figure, axes, and artists dictionary
    return fig, ax, artists

# Class to manage interactive buttons on the plot
# Handles button creation and the actions triggered when buttons are clicked
class plotbuttons:

    # Constructor: takes the figure, plot data handler (pdata), and optional button positions
    def __init__(self,fig,pdata,buttonpositions=[[0.81, 0.005, 0.1, 0.04],[0.26, 0.005, 0.2, 0.04],[0.66, 0.005, 0.1, 0.04],[0.51, 0.005, 0.1, 0.04]]):
        self.pdata=pdata # Store reference to the plot data handler (likely updates the plot)
        self.buttons=[] # List to hold button objects
        self.selected=0 # State variable for button actions (e.g., which vline to shift)
        # Lists mapping button index to function and label
        buttonfcts=[self.toggle_pause,self.select_vline,self.plus_action,self.minus_action]
        buttonlabels=["Pause","input shift","+","-"]
        # Create buttons based on provided positions
        for b,bpos in enumerate(buttonpositions):
            b_ax = fig.add_axes(bpos) # Add axes for the button
            btn = Button(b_ax, buttonlabels[b]) # Create the button widget
            btn.on_clicked(buttonfcts[b]) # Assign the click action
            self.buttons.append(btn) # Add button to the list
    # Action for the pause button: toggles animation state and button label
    def toggle_pause(self,event):
        if self.pdata.animation_is_running:
            self.buttons[0].label.set_text('Resume')
        else:
            self.buttons[0].label.set_text('Pause')
        self.pdata.animation_is_running=not self.pdata.animation_is_running # Toggle animation state in pdata
    # Action to select which vertical line/window is being manipulated by +/- buttons
    def select_vline(self,event):
        self.selected=(self.selected+1)%2 # Toggle selection state (0 or 1)
        if self.selected==0:
            self.buttons[1].label.set_text('input shift') # Update button label
        else:
            self.buttons[1].label.set_text('pred len') # Update button label
    # Action for the '+' button: shifts input window or increases prediction length
    def plus_action(self,event):
        if self.selected==0: # If 'input shift' is selected
            self.pdata.WindowXshift+=1 # Increase window shift
            # Apply constraints to the shift value
            self.pdata.WindowXshift=min(self.pdata.WindowXshift,0)
            self.pdata.WindowXshift=max(self.pdata.WindowXshift,-self.pdata.SEQ_LEN,-self.pdata.t) # Ensure shift is within valid range
        else: # If 'pred len' is selected
            self.pdata.predictionL+=1 # Increase prediction length
            self.pdata.predictionL=max(self.pdata.predictionL,1) # Ensure prediction length is at least 1
    # Action for the '-' button: shifts input window or decreases prediction length
    def minus_action(self,event):
        if self.selected==0: # If 'input shift' is selected
            self.pdata.WindowXshift-=1 # Decrease window shift
            # Apply constraints to the shift value
            self.pdata.WindowXshift=min(self.pdata.WindowXshift,0)
            self.pdata.WindowXshift=max(self.pdata.WindowXshift,-self.pdata.SEQ_LEN,-self.pdata.t) # Ensure shift is within valid range
        else: # If 'pred len' is selected
            self.pdata.predictionL-=1 # Decrease prediction length
            self.pdata.predictionL=max(self.pdata.predictionL,1) # Ensure prediction length is at least 1