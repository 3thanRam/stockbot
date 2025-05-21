# alpacabot/plot.py

import os # For accessing environment variables

from datetime import datetime, timedelta # For handling time
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
    plt.subplots_adjust(bottom=0.25) # Make space for more buttons
    # Set plot title and axis labels
    ax.set_title(f"{symbol} Stock Data")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{DATA_TYPE[whattoshow-1] if 0 < whattoshow < len(DATA_TYPE) else 'Data'}") # Dynamic Y label
    ax.grid(True) # Add a grid
    # ax.set_ylim(0, 500) # Initial Y limits will be set by plotdata.resize
    # Add a text box widget to display information (e.g., account balance)
    textbox = TextBox(ax.inset_axes([1.02, 0.9, 0.2, 0.08]), "ACC CASH:",
                  initial="$0.00") # Adjusted position slightly, more generic label
    # Initialize the line artist for the actual live/simulated data
    line_label = f"Live {DATA_TYPE[whattoshow-1] if 0 < whattoshow < len(DATA_TYPE) else 'Data'} Price"
    line, = ax.plot([], [], lw=2, label=line_label, c='blue', zorder=10)
    # Initialize the line artist for the model's predictions
    pred_label = f"Predicted {DATA_TYPE[whattoshow-1] if 0 < whattoshow < len(DATA_TYPE) else 'Data'}"
    predline, = ax.plot([], [], "--", lw=2, label=pred_label, c='orange', zorder=10)
    
    # Get the current time (will be updated by plotdata)
    nowline=datetime.now().astimezone()
    # Initialize vertical lines to indicate data windows (input start/end, prediction end)
    vline_input_start = ax.axvline(nowline, color='green', linestyle='--', lw=1.5, label=f'Input Start', zorder=5)
    vline_input_end = ax.axvline(nowline, color='green', linestyle='--', lw=1.5, label=f'Input End', zorder=5)
    vline_pred_end = ax.axvline(nowline, color='red', linestyle='-', lw=2, label=f'Prediction End', zorder=5) # Make visible by default
    
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
    def __init__(self,fig,pdata):
        self.pdata=pdata # Store reference to the plot data handler (likely updates the plot)
        self.buttons=[] # List to hold button objects
        self.selected_param_shift=0 # State variable for +/- buttons (0: input shift, 1: pred len)

        # Define button layouts [left, bottom, width, height]
        # Row 1: Main controls
        button_positions = [
            [0.02, 0.01, 0.12, 0.04],  # Pause/Resume
            [0.15, 0.01, 0.12, 0.04],  # Param Shift Select
            [0.28, 0.01, 0.05, 0.04],  # + (for param shift)
            [0.34, 0.01, 0.05, 0.04],  # - (for param shift)
            [0.40, 0.01, 0.12, 0.04],  # Reset View
        ]
        # Row 2: Zoom and Pan
        button_positions_row2 = [
            [0.02, 0.06, 0.08, 0.04],  # Zoom X In
            [0.11, 0.06, 0.08, 0.04],  # Zoom X Out
            [0.20, 0.06, 0.08, 0.04],  # Zoom Y In
            [0.29, 0.06, 0.08, 0.04],  # Zoom Y Out
            [0.38, 0.06, 0.08, 0.04],  # Pan Left
            [0.47, 0.06, 0.08, 0.04],  # Pan Right
            [0.56, 0.06, 0.08, 0.04],  # Pan Up
            [0.65, 0.06, 0.08, 0.04],  # Pan Down
        ]
        all_positions = button_positions + button_positions_row2

        button_labels=[
            "Pause","Input/Pred Len","+","-", "Reset View",
            "Zoom X+", "Zoom X-", "Zoom Y+", "Zoom Y-",
            "Pan Left", "Pan Right", "Pan Up", "Pan Down"
            ]
        button_actions=[
            self.toggle_pause, self.select_param_shift, self.plus_action, self.minus_action, self.reset_view_action,
            self.zoom_x_in, self.zoom_x_out, self.zoom_y_in, self.zoom_y_out,
            self.pan_left, self.pan_right, self.pan_up, self.pan_down
            ]

        for i, bpos in enumerate(all_positions):
            b_ax = fig.add_axes(bpos) # Add axes for the button
            btn = Button(b_ax, button_labels[i]) # Create the button widget
            btn.on_clicked(button_actions[i]) # Assign the click action
            self.buttons.append(btn) # Add button to the list

    # Action for the pause button: toggles animation state and button label
    def toggle_pause(self,event):
        if self.pdata.animation_is_running:
            self.buttons[0].label.set_text('Resume')
        else:
            self.buttons[0].label.set_text('Pause')
        self.pdata.animation_is_running=not self.pdata.animation_is_running # Toggle animation state in pdata

    # Action to select which vertical line/window is being manipulated by +/- buttons
    def select_param_shift(self,event):
        self.selected_param_shift=(self.selected_param_shift+1)%2 # Toggle selection state (0 or 1)
        if self.selected_param_shift==0:
            self.buttons[1].label.set_text('Input Shift') # Update button label
        else:
            self.buttons[1].label.set_text('Pred Len') # Update button label

    # Action for the '+' button: shifts input window or increases prediction length
    def plus_action(self,event):
        if self.selected_param_shift==0: # If 'input shift' is selected
            self.pdata.WindowXshift+=1 # Increase window shift
            if self.pdata.alldatapoints: # Ensure alldatapoints is not empty
                 self.pdata.WindowXshift=min(self.pdata.WindowXshift,len(self.pdata.alldatapoints)-1-self.pdata.SEQ_LEN-self.pdata.t)
            else:
                self.pdata.WindowXshift = 0 # Or some other default
        else: # If 'pred len' is selected
            self.pdata.predictionL+=1 # Increase prediction length
            self.pdata.predictionL=max(self.pdata.predictionL,1) # Ensure prediction length is at least 1

    # Action for the '-' button: shifts input window or decreases prediction length
    def minus_action(self,event):
        if self.selected_param_shift==0: # If 'input shift' is selected
            self.pdata.WindowXshift-=1 # Decrease window shift
            self.pdata.WindowXshift=max(self.pdata.WindowXshift,-self.pdata.t+1-self.pdata.SEQ_LEN) # Adjusted lower bound
        else: # If 'pred len' is selected
            self.pdata.predictionL-=1 # Decrease prediction length
            self.pdata.predictionL=max(self.pdata.predictionL,1) # Ensure prediction length is at least 1

    # --- New Button Actions for Zoom/Pan ---
    def reset_view_action(self, event):
        self.pdata.reset_view()

    def zoom_x_in(self, event): # Zoom In makes span smaller
        self.pdata.zoom_x(0.8)
    def zoom_x_out(self, event): # Zoom Out makes span larger
        self.pdata.zoom_x(1.25)
    def zoom_y_in(self, event):
        self.pdata.zoom_y(0.8)
    def zoom_y_out(self, event):
        self.pdata.zoom_y(1.25)

    def pan_left(self, event): # Moves view to the left (data appears to move right)
        self.pdata.pan_x(-0.1) # Pan by 10% of current view span
    def pan_right(self, event): # Moves view to the right
        self.pdata.pan_x(0.1)
    def pan_up(self, event): # Moves view up
        self.pdata.pan_y(0.1)
    def pan_down(self, event): # Moves view down
        self.pdata.pan_y(-0.1)