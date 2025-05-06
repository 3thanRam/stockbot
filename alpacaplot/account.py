# This script defines a class to interact with an Alpaca trading account.

import os # For accessing environment variables
# Import necessary classes from the Alpaca trading library
from alpaca.trading.client import TradingClient # Client for interacting with the trading API
from alpaca.trading.requests import MarketOrderRequest # Class to define a market order request
from alpaca.trading.enums import OrderSide, TimeInForce # Enums for order side (buy/sell) and time in force (how long order is valid)

# Retrieve API keys from environment variables
api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")

# Class to represent and interact with an Alpaca trading account
class alpacaAccount:
    # Constructor: Initializes the trading client and fetches initial account/portfolio data
    def __init__(self,symbol="TSLA",notional=200,time_in_force=TimeInForce.DAY):
        # Initialize the Alpaca TradingClient, specifying paper=True for paper trading
        self.trading_client=TradingClient(api_key, secret_key, paper=True)
        # Create a default MarketOrderRequest object (e.g., buy $200 of TSLA, good for the day)
        self.market_order_data = MarketOrderRequest(symbol=symbol,notional=notional,side=OrderSide.BUY,time_in_force=time_in_force)
        # Fetch initial account details
        self.details=self.trading_client.get_account()
        # Fetch initial portfolio (positions held)
        self.portfolio=self.trading_client.get_all_positions()

    # Method to submit the predefined market order
    def order(self):
        self.market_order = self.trading_client.submit_order(order_data=self.market_order_data)

    # Method to refresh account and portfolio details from the API
    def update(self):
        self.details=self.trading_client.get_account()
        self.portfolio=self.trading_client.get_all_positions()

    # Method to return a summary of account statistics
    # Returns cash balance and a list of (symbol, unrealized P/L) for each position
    def stats(self):
        return self.details.cash,[(pos.symbol,pos.unrealized_pl) for pos in self.portfolio]

    # Method to get the current total account equity
    def currentworth(self):
        return self.trading_client.get_account().equity

# Standard Python entry point
# This block runs only when the script is executed directly
from time import sleep # Import sleep for adding pauses

if __name__ == "__main__":
    # Create an instance of the alpacaAccount class
    account=alpacaAccount()
    # Get initial account statistics
    stats=account.stats()
    # Enter an infinite loop to continuously monitor account stats
    while True:
        # Update account and portfolio details
        account.update()
        # Get the latest account statistics
        tempstats=account.stats()
        # Check if the statistics have changed
        if stats!=tempstats:
            # If changed, print the new statistics, overwriting the current line
            print(tempstats,end="\r")
            # Update the stored statistics
            stats=tempstats
        # Pause execution for 1 second before the next check
        sleep(1)