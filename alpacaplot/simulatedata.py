# This script defines a class for plotting simulated live stock data using historical data.

from datetime import timedelta, datetime # For time calculations and handling
# Imports custom data classes/functions (dataclass likely misspelled, should be data structure/class name)
# plotdata: likely a base class for handling plot data and animation state
# gethistoricalstockbars: function to fetch historical data
from dataclass import plotdata,gethistoricalstockbars # NOTE: 'dataclass' module is for creating data classes, this import name seems incorrect. Assuming it's a custom module.

# Define the class for plotting simulated data, inheriting from a base plot data class
class plotsimulateddata(plotdata):
    # Constructor: initializes the simulated data plot handler
    def __init__(self,model,whattoshow,ax,symbols,artists):
        global streamdata # Access the global variable to store fetched historical data
        # Call the constructor of the parent class (plotdata)
        super().__init__(model,whattoshow,ax,symbols,artists)
        # Define the time range for fetching historical data (e.g., 15 days ending 20 mins ago)
        end = (datetime.now().astimezone() - timedelta(minutes=20))
        start = end - timedelta(days=15)
        # Fetch historical stock bars for the specified symbols and time range
        streamdata=gethistoricalstockbars(symbols,start,end,self.timeframe)

    # Method to get the current subset of data based on the simulated time 't'
    # Returns data points up to t + SEQ_LEN (or end of streamdata)
    def getdata(self):
        return streamdata[:min(self.t+self.SEQ_LEN,len(streamdata)-1)]

    # Method called by the animation to update the data points for the next frame
    def updatedata(self,framei):
        # Only update simulated time if the animation is running (not paused)
        if self.animation_is_running:
            self.t+=1 # Increment simulated time step
            # Get the data available up to the new simulated time
            self.alldatapoints=self.getdata()
            # Extract dates and the specific y-value (whattoshow) from the data points
            self.allDates=[dat[0] for dat in  self.alldatapoints]
            self.ally=[dat[self.whattoshow] for dat in  self.alldatapoints]

        # Define the visible window for plotting based on simulated time, sequence length, and window shift
        tmax=min(self.t+self.SEQ_LEN+self.WindowXshift,len(self.alldatapoints))
        tmin=self.t+self.WindowXshift
        # Extract the data points within the current visible window
        self.currentdatapoints=self.alldatapoints[tmin:tmax]
        # Extract dates and y-values for the current visible window
        self.Dates=[dat[0] for dat in self.currentdatapoints]
        self.y=[dat[self.whattoshow] for dat in self.currentdatapoints]
        # Call a method (likely from the parent class) to update the plot predictions
        self.updatepred(self.whattoshow)