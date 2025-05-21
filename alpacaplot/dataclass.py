# This script defines the base class for handling and plotting stock data,
# including fetching historical data and integrating a prediction model.

import numpy as np # For numerical operations, especially with predictions
import os # For accessing environment variables
import sys # For system-specific parameters and functions (though not used obviously here)

# For handling time and timezones
from datetime import timedelta,timezone, datetime # Added datetime
# Alpaca library components for fetching historical stock data
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# Import application-specific modules
import financebot.src.config as config # Configuration settings (e.g., sequence length)
from financebot.src.predictor import infer_realdata # Function to get model predictions
#from account import alpacaAccount # Class for interacting with the Alpaca trading account

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
        self.Dates=[] # Dates for the currently visible plot window (data fed to model)
        self.pred_dates=[] # Dates for the prediction line
        self.alldatapoints=[] # All data points fetched/received so far
        self.allDates=[] # Dates for all data points
        self.ally=[] # Y-values for all data points (based on whattoshow)
        self.preds=np.array([]) # Model prediction values, ensure it's an array
        self.currentdatapoints=[] # Data points within the current model input window
        # State variables controlling the plot view and data processing
        self.WindowXshift=0 # Horizontal shift of the model input window
        self.whattoshow=whattoshow # Index indicating which data type (open, close, etc.) to plot
        self.SEQ_LEN=config.SEQ_LEN # Length of the input sequence for the model (from config)
        self.predictionL=config.N_PRED_INFERENCE # Length of the prediction horizon (from config)
        self.t=0 # Simulated time step or index in the data stream
        self.ax=ax # Matplotlib axes object for drawing
        self.animation_is_running=True # Flag to control animation pause state
        #self.account=alpacaAccount() # Instance of the Alpaca trading account handler

        # --- New attributes for dynamic zoom/pan ---
        self.view_xlim = None # Tuple (xmin, xmax) for current view
        self.view_ylim = None # Tuple (ymin, ymax) for current view
        self.autofit_view_on_next_draw = True # Flag to trigger autofit on first draw or after reset

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
            preds_raw = infer_realdata(self.model, np.array(self.currentdatapoints).T,self.predictionL)
            if preds_raw.ndim > 1: # Check if preds_raw is 2D
                preds = preds_raw[:,whattoshow-1]
            else: # if preds_raw is 1D (e.g. predicting only one feature)
                preds = preds_raw
            # Prepend the last known y-value to the predictions to start the prediction line from there
            if self.y: # Ensure self.y is not empty
                self.preds=np.insert(preds,0,self.y[-1])
            else: # Fallback if self.y is empty (should not happen if self.Dates is populated)
                self.preds = preds
            # Generate dates for the prediction line, starting from the last date of current data
            self.pred_dates=[self.Dates[-1]]+self.getdates(self.Dates,self.predictionL)
        else:
            self.preds = np.array([])
            self.pred_dates = []

    # Method to resize the plot axes limits
    def resize(self,ax):
        if self.autofit_view_on_next_draw or self.view_xlim is None or self.view_ylim is None:
            # --- Autofit logic: Show all data + predictions ---
            if not self.allDates: # No data yet
                current_time = datetime.now(timezone.utc)
                self.view_xlim = (current_time - timedelta(hours=1), current_time + timedelta(hours=1))
                self.view_ylim = (0, 1) # Default placeholder
            else:
                # X-axis limits
                xmin_data = min(self.allDates)
                xmax_data = max(self.allDates)
                if self.pred_dates:
                    xmax_data = max(xmax_data, max(self.pred_dates))

                # Add padding to x-axis
                x_range_delta = xmax_data - xmin_data
                if x_range_delta.total_seconds() == 0: # Handle single point case
                    x_padding = timedelta(hours=1)
                else:
                    x_padding = x_range_delta * 0.05
                self.view_xlim = (xmin_data - x_padding, xmax_data + x_padding)

                # Y-axis limits
                if not self.ally: # No y-data yet
                     self.view_ylim = (0,1)
                else:
                    ymin_data = min(self.ally)
                    ymax_data = max(self.ally)
                    if self.preds.size > 0: # Check if preds array is not empty
                        ymin_data = min(ymin_data, np.min(self.preds))
                        ymax_data = max(ymax_data, np.max(self.preds))

                    # Add padding to y-axis
                    y_range = ymax_data - ymin_data
                    if y_range == 0: # Handle single point or all same value case
                        y_padding = max(1.0, abs(ymin_data) * 0.1) # 10% or 1 unit
                    else:
                        y_padding = y_range * 0.1
                    self.view_ylim = (ymin_data - y_padding, ymax_data + y_padding)
            
            self.autofit_view_on_next_draw = False # Reset flag after autofitting

        # Apply the determined or manually set limits
        ax.set_xlim(self.view_xlim)
        ax.set_ylim(self.view_ylim)
        ax.legend(loc='upper left')

    # Method to update the plot artists with the current data and predictions
    def redraw(self,ax):
        # Determine the start date of the input window
        if self.Dates: # Ensure Dates is not empty
            self.inpStart=self.Dates[0]
            # Update the position of the vertical line indicating the input start
            self.artists['vline_input_start'].set_xdata([self.inpStart])
            # Update the label for the input start vertical line
            self.artists['vline_input_start'].set_label(f"Input start: t=ti-{self.SEQ_LEN-self.WindowXshift}")
        else: # If Dates is empty, hide or default vline
            self.artists['vline_input_start'].set_xdata([datetime.now(timezone.utc)]) # Default position
            self.artists['vline_input_start'].set_label(f"Input start: (no data)")

        # Update the label for the main data line (showing current time step)
        self.artists['line'].set_label(f"{self.artists['line'].get_label().split(':')[0]}: i={self.t}")

        # Update prediction-related artists only if enough data exists for an input sequence
        if len(self.Dates)>=self.SEQ_LEN and self.pred_dates: # Check pred_dates as well
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
            self.artists['vline_input_end'].set_visible(True)
            self.artists['vline_pred_end'].set_visible(True)
            self.artists['predline'].set_visible(True)
        else: # Not enough data for prediction, hide prediction elements
            self.artists['vline_input_end'].set_visible(False)
            self.artists['vline_pred_end'].set_visible(False)
            self.artists['predline'].set_visible(False)


        # Update the text box content (e.g., displaying account cash balance)
        #self.artists["textbox"].set_val(f"${self.account.details.cash}") # Format as string
        # Resize the plot axes to fit the updated data
        self.resize(ax)
        # Update the data for the main plot line (showing all data points received so far)
        if self.allDates and self.ally: # Ensure there's data to plot
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

    def _adjust_view_limits_datetime(self, current_limits, center_point, new_span_delta):
        """Helper for adjusting datetime limits."""
        return (center_point - new_span_delta / 2, center_point + new_span_delta / 2)

    def _adjust_view_limits_numeric(self, current_limits, center_point, new_span):
        """Helper for adjusting numeric limits."""
        return (center_point - new_span / 2, center_point + new_span / 2)

    def zoom_x(self, factor):
        if self.view_xlim:
            self.autofit_view_on_next_draw = False
            xmin, xmax = self.view_xlim
            center = xmin + (xmax - xmin) / 2
            current_span_delta = xmax - xmin
            new_span_delta = timedelta(seconds=current_span_delta.total_seconds() * factor)
            if new_span_delta.total_seconds() < 60: # Minimum 1 minute span
                new_span_delta = timedelta(minutes=1)
            self.view_xlim = self._adjust_view_limits_datetime(self.view_xlim, center, new_span_delta)

    def zoom_y(self, factor):
        if self.view_ylim:
            self.autofit_view_on_next_draw = False
            ymin, ymax = self.view_ylim
            center = ymin + (ymax - ymin) / 2
            current_span = ymax - ymin
            new_span = current_span * factor
            if abs(new_span) < 0.01 : # Minimum 0.01 span
                new_span = 0.01 * (1 if new_span >=0 else -1)
            self.view_ylim = self._adjust_view_limits_numeric(self.view_ylim, center, new_span)
            if self.view_ylim[0] == self.view_ylim[1]: # Prevent flat line if possible
                 self.view_ylim = (self.view_ylim[0] - 0.005, self.view_ylim[1] + 0.005)


    def pan_x(self, fraction_of_span): # fraction_of_span: e.g., 0.1 for 10% shift
        if self.view_xlim:
            self.autofit_view_on_next_draw = False
            xmin, xmax = self.view_xlim
            span_delta = xmax - xmin
            shift_amount_delta = timedelta(seconds=span_delta.total_seconds() * fraction_of_span)
            self.view_xlim = (xmin + shift_amount_delta, xmax + shift_amount_delta)

    def pan_y(self, fraction_of_span): # fraction_of_span: e.g., 0.1 for 10% shift
        if self.view_ylim:
            self.autofit_view_on_next_draw = False
            ymin, ymax = self.view_ylim
            span = ymax - ymin
            shift_amount = span * fraction_of_span
            self.view_ylim = (ymin + shift_amount, ymax + shift_amount)

    def reset_view(self):
        self.autofit_view_on_next_draw = True
        # The actual reset will happen in resize() on the next draw call