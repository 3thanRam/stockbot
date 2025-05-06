# This script defines the base class for handling and plotting stock data,
# including fetching historical data and integrating a prediction model.

import numpy as np # For numerical operations, especially with predictions
import os # For accessing environment variables
import sys # For system-specific parameters and functions (though not used obviously here)

# For handling time and timezones
from datetime import timedelta,timezone
# Alpaca library components for fetching historical stock data
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# Import application-specific modules
import financebot.src.config as config # Configuration settings (e.g., sequence length)
from financebot.src.predictor import infer_realdata # Function to get model predictions
from account import alpacaAccount # Class for interacting with the Alpaca trading account

# Retrieve API keys from environment variables
api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")

# Function to format a data point (e.g., from Alpaca stream or history)
# Ensures the timestamp is in UTC and extracts relevant fields
def format_data(data):
    ts = data.timestamp
    # Convert timestamp to UTC timezone if not already timezone-aware
    ts = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    # Return a list containing timestamp, open, close, high, low values
    return [ts,data.open,data.close,data.high,data.low]

# Function to fetch historical stock bars from Alpaca
def gethistoricalstockbars(symbols,start,end,timeframe):
    # Initialize the historical data client
    client=StockHistoricalDataClient(api_key, secret_key)
    # Create a request object specifying symbols, timeframe, and date range
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=timeframe, start=start, end=end)
    # Get the stock bars data for the first symbol requested
    stockbars = client.get_stock_bars(req).data[symbols[0]]
    # Format each data point and return as a list
    return [format_data(data) for data in stockbars]

# Base class for handling and plotting stock data (simulated or live)
# Contains common plot state, data storage, prediction logic, and drawing methods
class plotdata:
    # Constructor: initializes plot data state and references to plot artists and model
    def __init__(self,model,whattoshow,ax,symbols,artists):
        self.artists=artists # Dictionary holding matplotlib artists (lines, vlines, textbox)
        self.timeframe=TimeFrame(1, TimeFrameUnit.Hour) # Timeframe for data (1 hour)
        self.model=model # Reference to the prediction model
        # Lists to hold dates and data points for plotting
        self.Dates=[] # Dates for the currently visible plot window
        self.pred_dates=[] # Dates for the prediction line
        self.alldatapoints=[] # All data points fetched/received so far
        self.allDates=[] # Dates for all data points
        self.ally=[] # Y-values for all data points (based on whattoshow)
        self.preds=[] # Model prediction values
        self.currentdatapoints=[] # Data points within the current visible window
        # State variables controlling the plot view and data processing
        self.WindowXshift=0 # Horizontal shift of the plot window
        self.whattoshow=whattoshow # Index indicating which data type (open, close, etc.) to plot
        self.SEQ_LEN=config.SEQ_LEN # Length of the input sequence for the model (from config)
        self.predictionL=config.N_PRED_INFERENCE # Length of the prediction horizon (from config)
        self.t=0 # Simulated time step or index in the data stream
        self.ax=ax # Matplotlib axes object for drawing
        self.animation_is_running=True # Flag to control animation pause state
        self.account=alpacaAccount() # Instance of the Alpaca trading account handler

    # Placeholder method to get data; must be implemented by child classes (simulated/live)
    def getdata(self):
        return []
    # Helper method to generate future dates based on the last known date
    def getdates(self,dateslist,numb):
        # Generates 'numb' dates, each 1 hour after the previous one, starting from the last date in dateslist
        return [dateslist[-1]+timedelta(hours=1)*(n+1) for n in range(numb)]
    # Method to run model inference and update prediction data
    def updatepred(self,whattoshow):
        # Only attempt prediction if enough data points are available for the model's input sequence
        if len(self.Dates)>=self.SEQ_LEN:
            # Run inference using the model; predicts 'predictionL' steps ahead
            # Transpose currentdatapoints to match expected model input shape
            preds = infer_realdata(self.model, np.array(self.currentdatapoints).T,self.predictionL)[:,whattoshow-1]
            # Prepend the last known y-value to the predictions to start the prediction line from there
            self.preds=np.insert(preds,0,self.y[-1])
            # Generate dates for the prediction line, starting from the last date of current data
            self.pred_dates=[self.Dates[-1]]+self.getdates(self.Dates,self.predictionL)

    # Method to resize the plot axes limits
    def resize(self,ax):
        # Determine the maximum x-axis limit based on available data or prediction dates
        if self.pred_dates:
            xmax=self.pred_dates[-1]
        else:
            xmax=self.Dates[-1]
        # The minimum x-axis limit is the first date of the current data
        xmin=self.Dates[0]
        dt=timedelta(hours=1) # A small time delta for padding

        # Set the x-axis limits with padding
        ax.set_xlim(xmin-dt,xmax+dt)
        # Set the y-axis limits based on the maximum y-value in the current window
        ax.set_ylim(0, 1.1*max(self.y))
        # Ensure the legend is displayed
        ax.legend(loc='upper left')
    # Method to update the plot artists with the current data and predictions
    def redraw(self,ax):
        # Determine the start date of the input window
        self.inpStart=self.Dates[0]
        # Update the label for the main data line (showing current time step)
        self.artists['line'].set_label(f"Open Price: i={self.t}") # NOTE: Label still says "Open Price"
        # Update the position of the vertical line indicating the input start
        self.artists['vline_input_start'].set_xdata([self.inpStart])
        # Update the label for the input start vertical line
        self.artists['vline_input_start'].set_label(f"Input start: t=ti-{self.SEQ_LEN-self.WindowXshift}")
        # Update prediction-related artists only if enough data exists for an input sequence
        if len(self.Dates)>=self.SEQ_LEN:
            # Determine the end date of the input window
            self.inpEnd=self.Dates[self.SEQ_LEN-1]
            # Determine the end date of the prediction line
            self.predEnd=self.pred_dates[-1]
            # Update the positions of the vertical lines for input end and prediction end
            self.artists['vline_input_end'].set_xdata([self.inpEnd])
            self.artists['vline_pred_end'].set_xdata([self.predEnd])
            # Update the label for the prediction end vertical line (showing prediction length)
            self.artists['vline_pred_end'].set_label(f"Pred End: L={self.predictionL}")
            # Update the data for the prediction line
            self.artists['predline'].set_data(self.pred_dates,self.preds)

        # Update the text box content (e.g., displaying account cash balance)
        self.artists["textbox"].set_val([self.account.details.cash])
        # Resize the plot axes to fit the updated data
        self.resize(ax)
        # Update the data for the main plot line (showing all data points received so far)
        self.artists['line'].set_data(self.allDates, self.ally)
    # Placeholder method to update data; must be implemented by child classes
    # This method is where new data (simulated or live) is fetched or processed
    def updatedata(self,framei):
        return []
    # Main method called by the matplotlib animation framework
    # Orchestrates data update and plot redrawing
    def updatedraw(self,framei):
        # Call the child class's method to update the underlying data
        self.updatedata(framei)
        # Call the redraw method to update all plot artists based on the latest data
        self.redraw(self.ax)
        # Return the list of artists that need to be redrawn on the canvas
        return list(self.artists.values())