# livedata.py

# Module for running tasks in separate threads
import threading
# For time calculations and handling
from datetime import timedelta, datetime
# Alpaca library for connecting to the live stock data stream
from alpaca.data.live import StockDataStream
# Imports custom data classes/functions
# plotdata: Base class for handling plot data and animation state
# gethistoricalstockbars: Function to fetch initial historical data
# format_data: Function to process incoming stream data
# api_key, secret_key: Alpaca API credentials
from dataclass import plotdata,gethistoricalstockbars,format_data,api_key,secret_key


# Global list to store the received stream data points
# Using a global list accessible/modified by both the async handler and the plot class
streamdata=[]

# Async function to handle incoming stock data from the stream
# This function is called by the StockDataStream client whenever new data arrives
async def stock_data_stream_handler(data):
    # Print the raw incoming data (for debugging/monitoring)
    print(data)
    # Format the incoming data and append it to the global streamdata list
    streamdata.append(format_data(data))

# Class for plotting live data from the Alpaca stream, inheriting from plotdata
class plotlivedata(plotdata):
    # Constructor: sets up the live data connection and initializes plot data
    def __init__(self,model,whattoshow,ax,symbols,artists):
        global streamdata # Access the global variable to store stream data
        # Call the constructor of the parent class (plotdata)
        super().__init__(model,whattoshow,ax,symbols,artists)
        
        # Define the time range for fetching initial historical data (e.g., 1 week ending 15 mins ago)
        end = (datetime.now().astimezone() - timedelta(minutes=15))
        start = end - timedelta(days=200)
        # Fetch initial historical data to populate the plot before the stream starts
        streamdata=gethistoricalstockbars(symbols,start,end,self.timeframe)
        # Print the number of historical data points fetched
        print(len(streamdata))
        # Initialize the StockDataStream client with API keys
        stock_data_stream_client = StockDataStream(
            api_key, secret_key)

        # Subscribe the handler function to receive bar data for the specified symbols
        stock_data_stream_client.subscribe_bars(stock_data_stream_handler, *symbols)
        # Create a new thread to run the streaming client
        # The client's run() method is blocking and needs to be in a separate thread
        self.t1 =threading.Thread(target=stock_data_stream_client.run)
        # Start the thread to begin receiving data
        self.t1.start()

    # Method to get the current complete list of data points received so far
    # Returns a copy of the global streamdata list
    def getdata(self):
        return list(streamdata)

    # Method called by the animation to update the plot with new live data
    def updatedata(self,framei):
        # Update account information 
        #self.account.update()
        # Only update data time step if the animation is running (not paused)
        if self.animation_is_running:
            # Get the latest snapshot of the stream data
            alldatapoints=self.getdata()
            # Check if new data has been added to streamdata since the last update
            if self.alldatapoints!=alldatapoints:
                self.t=len(alldatapoints)-self.SEQ_LEN
                # Update internal lists with the new complete data
                self.alldatapoints=alldatapoints
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