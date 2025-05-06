# This script handles fetching, formatting, normalizing, and sequencing
# stock data for the finance bot's training and prediction workflows.

# Module for running tasks in separate threads (used for potentially fetching multiple symbols in parallel,
# though the current _get_historical_bars_threaded is called sequentially in get_training_data)
import threading
import torch

import numpy as np # For numerical operations and array manipulation
import time # For adding delays (e.g., between API calls)
import os # For interacting with the operating system (environment variables, file paths)
# For handling time and timezones
from datetime import datetime, timezone
# Alpaca library components for fetching historical stock data
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
# For plotting dates (though not directly used for plotting *in* this script, relevant to data output)
import matplotlib.dates as mdates
# Import application configuration settings
import financebot.src.config as config

# Global list to track symbols for which data fetching failed
skipped_symbols_list = []

# Retrieve Alpaca API keys from environment variables
api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")

# --- API Key Check ---
# Raise an error if API keys are not found in environment variables
if not api_key or not secret_key:
    raise ValueError(
        "API keys not found. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")

# --- Helper Function: Time Formatting ---
def _formattime(rawdata):
    """
    Converts various input formats to a timezone-aware datetime object (UTC).
    Prioritizes handling existing datetime objects correctly.
    """
    # Check if the input is already a datetime object
    if isinstance(rawdata, datetime):
        # If timezone-naive, make it UTC-aware
        if rawdata.tzinfo is None:
            return rawdata.replace(tzinfo=timezone.utc)
        else:
            # If timezone-aware, convert to UTC
            return rawdata.astimezone(timezone.utc)

    # --- Fallback checks for string/other formats ---
    str_data = str(rawdata)

    # Check for 'YYYYMMDD' format (length 8 string of digits)
    if len(str_data) == 8 and str_data.isdigit():
        try:
            return datetime.strptime(str_data, '%Y%m%d').replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    # Check if it looks like a Unix timestamp (integer or float string)
    try:
        epoch_seconds = int(float(rawdata))
        # Basic check if timestamp is likely recent enough (e.g., after year 2000)
        if epoch_seconds > 946684800:
            return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    except (ValueError, TypeError):
        pass

    # Check for 'YYYYMMDD HH:MM:SS' format
    try:
        return datetime.strptime(str_data, '%Y%m%d %H:%M:%S').replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    # If none of the above formats matched
    print(f"Warning: Unrecognized date format '{rawdata}' (type: {type(rawdata)}). Returning None.")
    return None

# --- Define the expected number of features ---
# Based on the categories defined in the configuration.
# Used to ensure consistent array shapes, especially for empty data.
EXPECTED_NUM_FEATURES = len(config.CATEGORIES)


# --- Internal Helper Function: Fetch Historical Bars per Symbol ---
def _get_historical_bars_threaded(symbol):
    """
    Internal function to fetch historical stock bar data for a single symbol
    using the Alpaca API. Returns data as a NumPy array.
    """
    print(f"--- Fetching Alpaca data for {symbol} ---")
    # Initialize the historical data client with API keys
    client = StockHistoricalDataClient(api_key, secret_key)

    # Define the data request using parameters from the configuration
    req = StockBarsRequest(
        symbol_or_symbols=[symbol], # Request data for a single symbol
        timeframe=config.FREQ, # Timeframe (e.g., 1 hour) from config
        start=config.START, # Start date from config
        end=config.END # End date from config
    )

    # Attempt to fetch data from the Alpaca API
    try:
        stockbars = client.get_stock_bars(req)
        print(f"Alpaca API call successful for {symbol}.")
    except Exception as e:
        # Handle any exceptions during the API call
        print(f"Error fetching data from Alpaca for {symbol}: {e}")
        # Return an empty array with the correct feature dimension if fetching fails
        return np.empty((EXPECTED_NUM_FEATURES, 0), dtype=object)

    # --- Process the received data ---
    processed_data_rows = [] # List to temporarily hold data for each bar as a row

    # Check if data was returned for the requested symbol
    if symbol in stockbars.data:
        bars = stockbars.data[symbol]
        print(f"Processing {len(bars)} bars returned for {symbol}...")

        # Handle case where the data list for the symbol is empty
        if not bars:
            print(f"Warning: No bars in the data list for symbol {symbol}.")
            # Return empty array if no bars are found
            return np.empty((EXPECTED_NUM_FEATURES, 0), dtype=object)

        # Iterate through each bar received
        for bar in bars:
            current_bar_data = []
            try:
                # Extract data points based on the order defined in config.CATEGORIES
                for category in config.CATEGORIES:
                    if category == 'date':
                        # Format the timestamp to timezone-aware UTC datetime
                        ts = bar.timestamp
                        if ts.tzinfo is None:
                            ts_aware = ts.replace(tzinfo=timezone.utc)
                        else:
                            ts_aware = ts.astimezone(timezone.utc)
                        current_bar_data.append(ts_aware)
                    # Append corresponding bar attributes for other categories
                    elif category == 'open':
                        current_bar_data.append(bar.open)
                    elif category == 'high':
                        current_bar_data.append(bar.high)
                    elif category == 'low':
                        current_bar_data.append(bar.low)
                    elif category == 'close':
                        current_bar_data.append(bar.close)
                    elif category == 'volume':
                        current_bar_data.append(bar.volume)
                    else:
                        # Warning for unhandled categories
                        print(f"Warning: Unhandled category '{category}' in config.CATEGORIES. Skipping.")
                        current_bar_data.append(None) # Append None as a placeholder

                # Ensure the number of extracted features matches the expected count
                if len(current_bar_data) == EXPECTED_NUM_FEATURES:
                    processed_data_rows.append(current_bar_data)
                else:
                     # Warning if a bar didn't yield the expected number of features
                     print(f"Warning: Skipping bar due to mismatched feature count. Expected {EXPECTED_NUM_FEATURES}, got {len(current_bar_data)}. Bar: {bar}")

            # Handle errors accessing bar attributes or other exceptions during processing
            except AttributeError as e:
                print(f"Error accessing attribute for bar data: {e}. Bar: {bar}. Skipping bar.")
            except Exception as e:
                print(f"Unexpected error processing bar: {e}. Bar: {bar}. Skipping bar.")

    else:
        # Warning if the symbol key was not found in the API response data
        print(f"Warning: No data key found for symbol '{symbol}' in Alpaca response.")
        # Return empty array if no data for the symbol
        return np.empty((EXPECTED_NUM_FEATURES, 0), dtype=object)

    # --- Final check and array conversion ---
    # Check if any valid bars were processed and added to the list
    if not processed_data_rows:
        print(f"Warning: No valid bars were processed for {symbol} after fetching.")
        # Return empty array if no bars were successfully processed
        return np.empty((EXPECTED_NUM_FEATURES, 0), dtype=object)

    # Convert the list of processed bar data (rows) into a NumPy array
    # Initial shape will be (num_bars, num_features_incl_date)
    data_array_bars_as_rows = np.array(processed_data_rows, dtype=object)

    # Transpose the array to get the desired shape: (num_features_incl_date, num_bars)
    final_array = data_array_bars_as_rows.T

    # Print success message with final array shape
    print(f"Successfully processed {final_array.shape[1]} bars for {symbol}. Final shape: {final_array.shape}")
    return final_array

# --- Helper Function: Generate Sequences from Data ---
def _generate_sequences(symbarray, seq_len, max_horizon, input_size):
    """
    Generates sequences suitable for Transformer training (encoder input,
    decoder input for teacher forcing, and target output) from normalized data.
    """
    sequences_enc = [] # List for encoder input sequences
    sequences_dec_in = [] # List for decoder input sequences
    sequences_tgt = [] # List for target output sequences
    symbol_sequences = 0 # Counter for sequences generated for this symbol

    # Data array shape is expected to be (num_features + 1, num_bars), with date at index 0
    # Normalize based ONLY on the feature data (rows from index 1 onwards)
    norm_values = np.empty_like(symbarray[1:], dtype=float) # Array to hold normalized feature values
    # Need at least 2 data points to calculate standard deviation reliably
    if symbarray.shape[1] > 1:
        # Extract feature values and ensure they are float type for calculations
        data_values = symbarray[1:].astype(float)

        ## # --- Optional: Apply log1p transform to volume if needed ---
        ## # This block is commented out but shows how to handle volume transformation before normalization
        ## volume_index = 4 # Assuming volume is the 5th feature (index 4)
        ## if data_values.shape[0] > volume_index: # Check if the volume feature exists
        ##    # Apply log1p (log(1 + x)) transform, common for skewed data like volume
        ##    data_values[volume_index, :] = np.log1p(data_values[volume_index, :])

        # Calculate mean and standard deviation across time steps (columns) for each feature (row)
        mean = data_values.mean(axis=1, keepdims=True) # Shape (input_size, 1)
        std = data_values.std(axis=1, keepdims=True) + \
            1e-6  # Add epsilon to prevent division by zero if std is 0
        # Apply Z-score normalization: (data - mean) / std
        norm_values = (data_values - mean) / std # Shape (input_size, num_bars)
    elif symbarray.shape[1] == 1:
        # If only one data point, set normalized values to 0 (std dev is undefined)
        norm_values = np.zeros_like(symbarray[1:], dtype=float)
    else: # No data points available
        # Return empty lists if no data to process
        return [], [], [], 0

    # Determine the number of possible sequences that can be generated
    # Need seq_len bars for encoder input + max_horizon bars for target output
    if norm_values.shape[1] >= seq_len + max_horizon:
        num_sequences_possible = norm_values.shape[1] - seq_len - max_horizon + 1

        # Iterate to create sequences by sliding a window across the data
        for i in range(num_sequences_possible):
            start_idx = i # Start index of the sequence window
            encoder_end_idx = i + seq_len # End index for encoder input (exclusive)
            decoder_end_idx = encoder_end_idx + max_horizon # End index for decoder target (exclusive)

            # Encoder Input: Data from start_idx up to encoder_end_idx
            # Shape: (input_size, seq_len)
            encoder_input_data = norm_values[:, start_idx:encoder_end_idx]

            # Target Output: Data from encoder_end_idx up to decoder_end_idx
            # Shape: (input_size, max_horizon)
            target_data = norm_values[:, encoder_end_idx:decoder_end_idx]

            # Decoder Input (for Teacher Forcing during training):
            # Consists of the last point of the encoder input + the target shifted by one position.
            # Shape: (input_size, max_horizon)
            # Get the last data point from the encoder sequence
            last_encoder_val = norm_values[:,
                                           encoder_end_idx-1:encoder_end_idx] # Shape (input_size, 1)
            # Get the target sequence shifted left by one position
            target_shifted = norm_values[:, encoder_end_idx:decoder_end_idx-1] # Shape (input_size, max_horizon - 1)
            # Concatenate the last encoder value with the shifted target to form decoder input
            decoder_input_data = np.concatenate(
                (last_encoder_val, target_shifted), axis=1) # Shape (input_size, max_horizon)

            # Append the flattened sequences to their respective lists
            # Flattening converts shape (input_size, seq_len/horizon) to (input_size * seq_len/horizon,)
            sequences_enc.append(encoder_input_data.flatten())
            sequences_dec_in.append(decoder_input_data.flatten())
            sequences_tgt.append(target_data.flatten())
            symbol_sequences += 1 # Increment the count of sequences generated

    # Return the lists of sequences and the count
    return sequences_enc, sequences_dec_in, sequences_tgt, symbol_sequences


# --- Main Function: Get Training Data ---
def get_training_data():
    """
    Loads pre-generated training data if available, otherwise fetches data
    for all training symbols, generates sequences, splits into training
    and validation sets, and saves the result.
    """
    global skipped_symbols_list # Access the global list for skipped symbols
    skipped_symbols_list = [] # Reset the list at the start

    seq_len = config.SEQ_LEN # Sequence length from config
    max_horizon = config.N_PRED_TRAINING_MAX # Max prediction horizon for training from config
    input_size = config.INPUT_SIZE # Number of input features from config
    # Determine if data fetching is forced (ignores saved data) or not
    force_fetch = not config.LOAD_TRAINING_DATA

    # Check if loading from a saved file is attempted and if the file exists
    if not force_fetch and os.path.isfile(config.TRAINING_DATA_PATH):
        print(
            f"Loading pre-generated training data from: {config.TRAINING_DATA_PATH}")
        try:
            # Load data from the .npz file
            with np.load(config.TRAINING_DATA_PATH) as npfile:
                Xtrain_enc = npfile["Xtrain_enc"]
                Xtrain_dec_in = npfile["Xtrain_dec_in"]
                Ytrain_tgt = npfile["Ytrain_tgt"]
                Xval_enc = npfile["Xval_enc"]
                Xval_dec_in = npfile["Xval_dec_in"]
                Yval_tgt = npfile["Yval_tgt"]
            print("Data loaded successfully.")
            force_fetch = False  # Data loaded, no need to fetch
        except Exception as e:
            # If loading fails, print error and force fetching
            print(
                f"Error loading data from {config.TRAINING_DATA_PATH}: {e}. Regenerating.")
            force_fetch = True

    # If data fetching is required (either forced or load failed)
    if force_fetch:
        print("Fetching and processing training data...")
        all_sequences_encoder = [] # List to collect encoder sequences from all symbols
        all_sequences_decoder_input = [] # List to collect decoder input sequences from all symbols
        all_sequences_target = [] # List to collect target sequences from all symbols
        total_sequences_generated = 0 # Counter for total sequences
        Nticks = {} # Dictionary to store number of bars fetched per symbol

        # Iterate through each symbol specified in the configuration for training
        for i, symbol in enumerate(config.SYMBOLS):
            print(f"\n--- Fetching data for {symbol} ---")
            # Fetch historical bars for the current symbol
            bars = _get_historical_bars_threaded(
                symbol)

            # Check if bars were successfully fetched
            if bars.shape[1] > 0:
                Nticks[symbol] = bars.shape[1] # Record number of bars
                print(f"Generating sequences for {symbol}...")
                # Generate sequences from the fetched bars
                enc, dec_in, tgt, count = _generate_sequences(
                    bars, seq_len, max_horizon, input_size)
                # If sequences were generated for this symbol
                if count > 0:
                    # Extend the global lists with sequences from this symbol
                    all_sequences_encoder.extend(enc)
                    all_sequences_decoder_input.extend(dec_in)
                    all_sequences_target.extend(tgt)
                    total_sequences_generated += count # Update total count
                    print(f"Generated {count} sequences for {symbol}.")
                else:
                    # If insufficient data for sequences
                    print(
                        f"Insufficient data ({bars.shape[1]} bars) for sequences for {symbol}.")
                    skipped_symbols_list.append(symbol) # Add symbol to skipped list
            else:
                # If no bars were fetched at all for the symbol
                skipped_symbols_list.append(symbol) # Add symbol to skipped list
            time.sleep(0.5)  # Add a small delay between API calls

        # Print a summary of the data fetching process
        print("\n--- Data Fetch Summary ---")
        print("Fetched bars per symbol:", Nticks)
        print("Skipped symbols:", skipped_symbols_list)
        print(f"Total sequences generated: {total_sequences_generated}")

        # If no sequences were generated across all symbols, print error and return empty tensors
        if not all_sequences_encoder:
            print("ERROR: No sequences generated. Cannot proceed with training.")
            # Define shapes for empty tensors with correct flattened dimensions
            empty_shape_enc = (0, seq_len * input_size)
            empty_shape_dec = (0, max_horizon * input_size)
            # Return empty tensors wrapped in tuples
            return (torch.empty(empty_shape_enc), torch.empty(empty_shape_dec), torch.empty(empty_shape_dec)), \
                   (torch.empty(empty_shape_enc), torch.empty(
                       empty_shape_dec), torch.empty(empty_shape_dec))

        # Convert lists of sequences to NumPy arrays
        X_enc = np.array(all_sequences_encoder)
        X_dec_in = np.array(all_sequences_decoder_input)
        Y_tgt = np.array(all_sequences_target)

        # Shuffle the generated sequences
        indices = np.arange(len(X_enc)) # Get indices of the sequences
        np.random.shuffle(indices) # Shuffle the indices randomly
        # Reorder the sequence arrays using the shuffled indices
        X_enc, X_dec_in, Y_tgt = X_enc[indices], X_dec_in[indices], Y_tgt[indices]

        # Determine the index to split data into training and validation sets
        # Based on the validation split ratio from config
        valsplit_index = int(len(X_enc) * (1 - config.VAL_SPLIT_RATIO))
        # Split the data into training and validation sets
        Xtrain_enc, Xval_enc = X_enc[:valsplit_index], X_enc[valsplit_index:]
        Xtrain_dec_in, Xval_dec_in = X_dec_in[:
                                              valsplit_index], X_dec_in[valsplit_index:]
        Ytrain_tgt, Yval_tgt = Y_tgt[:valsplit_index], Y_tgt[valsplit_index:]

        # Print the size of the training and validation sets
        print(f"\n--- Data Split ---")
        print(f"Training sequences: {len(Xtrain_enc)}")
        print(f"Validation sequences: {len(Xval_enc)}")

        # Attempt to save the generated training and validation data to a file
        try:
            print(
                f"Saving generated training data to: {config.TRAINING_DATA_PATH}")
            # Save the arrays in a compressed .npz format
            np.savez(config.TRAINING_DATA_PATH,
                     Xtrain_enc=Xtrain_enc, Xtrain_dec_in=Xtrain_dec_in, Ytrain_tgt=Ytrain_tgt,
                     Xval_enc=Xval_enc, Xval_dec_in=Xval_dec_in, Yval_tgt=Yval_tgt)
        except Exception as e:
            # Print error if saving fails
            print(f"Error saving data to {config.TRAINING_DATA_PATH}: {e}")

    # Convert the NumPy arrays of training and validation data to PyTorch tensors
    # Ensure data type is float32
    Xtrain_enc_t = torch.tensor(Xtrain_enc, dtype=torch.float32)
    Xtrain_dec_in_t = torch.tensor(Xtrain_dec_in, dtype=torch.float32)
    Ytrain_tgt_t = torch.tensor(Ytrain_tgt, dtype=torch.float32)
    Xval_enc_t = torch.tensor(Xval_enc, dtype=torch.float32)
    Xval_dec_in_t = torch.tensor(Xval_dec_in, dtype=torch.float32)
    Yval_tgt_t = torch.tensor(Yval_tgt, dtype=torch.float32)

    # Return the training and validation data as tuples of tensors
    train_data = (Xtrain_enc_t, Xtrain_dec_in_t, Ytrain_tgt_t)
    val_data = (Xval_enc_t, Xval_dec_in_t, Yval_tgt_t)

    return train_data, val_data


# --- Main Function: Get Live Data for Prediction ---
def get_live_data(symbol):
    """
    Fetches the latest historical data required as input for the model's
    prediction (inference) step. Normalizes this data using its own stats.
    Returns the normalized input sequence, the full fetched bars, and the
    normalization statistics.
    """
    print(f"Fetching live data for prediction symbol: {symbol}")
    seq_len = config.SEQ_LEN # Sequence length required for model input
    # Define the path for caching the fetched live data using a template
    live_data_cache_path = config.LIVE_DATA_CACHE_PATH_TEMPLATE.format(
        symbol=symbol)

    # --- Caching Mechanism ---
    # Use the LOAD_TRAINING_DATA flag from config to also control live data caching
    load_from_cache = config.LOAD_TRAINING_DATA
    bars = None # Initialize bars variable

    # Check if loading from cache is enabled and if the cache file exists
    if load_from_cache and os.path.exists(live_data_cache_path):
        try:
            print(f"Loading cached live data from {live_data_cache_path}")
            # Load data from the cache file
            bars = np.load(live_data_cache_path, allow_pickle=True) # Allow_pickle=True needed for dtype=object array
        except Exception as e:
            # If loading fails, print error and set bars to None to force fetching
            print(
                f"Could not load cached live data: {e}. Fetching fresh data.")
            bars = None

    # If bars are not loaded from cache (cache disabled or load failed)
    if bars is None:
        print(
            f"Fetching live data using window '{config.FREQ}'...")
        # Fetch the latest historical bars using the helper function
        bars = _get_historical_bars_threaded(
            symbol)
        # Check if any bars were fetched
        if bars.shape[1] > 0:
            # Attempt to save the fetched data to the cache file
            try:
                print(
                    f"Saving fetched live data ({bars.shape[1]} bars) to {live_data_cache_path}")
                np.save(live_data_cache_path, bars)
            except Exception as e:
                print(f"Error saving live data cache: {e}")
        else:
            # If no bars were fetched, print warning and raise an error as prediction requires data
            print(f"Warning: Fetched 0 bars for {symbol}.")
            raise ValueError(f"Failed to fetch any live data for {symbol}.")

    # Print the total number of bars fetched for the prediction base
    print(f"Total bars fetched for live sequence base: {bars.shape[1]}")
    # Check if enough bars were fetched to form the required input sequence length
    if bars.shape[1] < seq_len:
        # Raise an error if insufficient data for the input sequence
        raise ValueError(
            f"Not enough live data for {symbol}. Need {seq_len}, got {bars.shape[1]}. "
            f"Check fetch window ('{config.FREQ}') or data availability.")

    # --- Prepare Input Sequence for Prediction ---
    # Take the MOST RECENT `seq_len` bars from the fetched data
    bars_input_sequence = bars[:, -seq_len:].copy() # Use .copy() to avoid modifying the original array
    print(
        f"Using the last {bars_input_sequence.shape[1]} bars for the input sequence.")

    # --- Normalize the Input Sequence ---
    # Normalize the feature data (excluding dates) using statistics calculated ONLY from this sequence.
    norm_bars_input = bars_input_sequence.copy() # Work on a copy
    # Extract feature values and ensure float type
    data_values = norm_bars_input[1:].astype(
        float)
    
    ## # --- Optional: Apply log1p transform to volume if needed ---
    ## volume_index = 4
    ## if data_values.shape[0] > volume_index:
    ##    data_values[volume_index, :] = np.log1p(data_values[volume_index, :])

    # Calculate mean and standard deviation for normalization
    mean = data_values.mean(axis=1, keepdims=True)
    std = data_values.std(axis=1, keepdims=True) + 1e-6
    # Apply Z-score normalization
    norm_values = (data_values - mean) / std  # Shape: (input_size, seq_len)

    # Transpose the normalized sequence to match the expected input format for the predictor function
    # Expected shape: (seq_len, input_size)
    norm_sequence_transposed = norm_values.T

    # Return the normalized input sequence, the full historical bars (for plotting context),
    # and the mean/std used for normalization (needed for denormalization of predictions).
    return norm_sequence_transposed, bars, (mean, std)