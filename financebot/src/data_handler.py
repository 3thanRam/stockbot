import threading
import numpy as np
import time
import os
from datetime import datetime, timezone
import torch
import matplotlib.dates as mdates
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
import financebot.src.config as config

# Global list to track skipped symbols (if needed elsewhere)
skipped_symbols_list = []

api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")


# Check if keys are loaded (important if using environment variables)
if not api_key or not secret_key:
    raise ValueError(
        "API keys not found. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")

# Inside data_handler.py

# --- Inside data_handler.py ---
from datetime import datetime, timezone # Make sure this import is correct at the top

def _formattime(rawdata):
    """Converts input to a timezone-aware datetime object (UTC)."""
    # --- START MODIFICATION ---
    # 1. Check if already a datetime object (THIS MUST BE FIRST)
    if isinstance(rawdata, datetime):
        # Ensure it's timezone-aware and set to UTC
        if rawdata.tzinfo is None:
            return rawdata.replace(tzinfo=timezone.utc)
        else:
            # If it's already aware, convert to UTC just in case it's different
            return rawdata.astimezone(timezone.utc)
    # --- END MODIFICATION ---

    # --- Fallback checks for other formats (e.g., from old data) ---
    str_data = str(rawdata)

    # 2. Check for YYYYMMDD format (length 8 string)
    if len(str_data) == 8 and str_data.isdigit():
        try:
            return datetime.strptime(str_data, '%Y%m%d').replace(tzinfo=timezone.utc)
        except ValueError:
            pass # Continue

    # 3. Check if it looks like a Unix timestamp (potentially float)
    try:
        epoch_seconds = int(float(rawdata))
        if epoch_seconds > 946684800: # Jan 1, 2000 UTC
            return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    except (ValueError, TypeError):
        pass # Continue

    # 4. Check for 'YYYYMMDD HH:MM:SS' format
    try:
        # Note: The format string provided by the error ('YYYY-MM-DD HH:MM:SS+ZZ:ZZ')
        # is *not* handled here, but it shouldn't be needed if the isinstance check works.
        return datetime.strptime(str_data, '%Y%m%d %H:%M:%S').replace(tzinfo=timezone.utc)
    except ValueError:
        pass # Continue

    # If none of the above matched:
    # This warning should NOT appear for datetime inputs if the isinstance check is working
    print(f"Warning: Unrecognized date format '{rawdata}' (type: {type(rawdata)}). Returning None.")
    return None
# --- START OF REVISED _get_historical_bars_threaded ---
import numpy as np
import time
import os
from datetime import datetime, timezone
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
import config  # Import configuration

# Ensure API keys are loaded (place this near the top of data_handler.py if not already there)
api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")
if not api_key or not secret_key:
    raise ValueError(
        "API keys not found. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")

# Define the expected number of features based on config.CATEGORIES
# This ensures the empty array returned on failure has the correct first dimension
EXPECTED_NUM_FEATURES = len(config.CATEGORIES)


def _get_historical_bars_threaded(symbol):
    """
    Internal function to fetch bars for one symbol using Alpaca.
    Returns a NumPy array of shape (num_features_incl_date, num_bars)
    ordered according to config.CATEGORIES.
    """
    print(f"--- Fetching Alpaca data for {symbol} ---")
    client = StockHistoricalDataClient(api_key, secret_key)

    # --- Define the request parameters using config ---
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=config.FREQ,
        start=config.START,
        end=config.END
    )

    # --- Attempt to fetch data ---
    try:
        stockbars = client.get_stock_bars(req)
        print(f"Alpaca API call successful for {symbol}.")
    except Exception as e:
        print(f"Error fetching data from Alpaca for {symbol}: {e}")
        # Return an empty array with the correct number of feature rows
        return np.empty((EXPECTED_NUM_FEATURES, 0), dtype=object)

    # --- Process the received data ---
    processed_data_rows = [] # To store data bar-by-bar initially

    if symbol in stockbars.data:
        bars = stockbars.data[symbol]
        print(f"Processing {len(bars)} bars returned for {symbol}...")

        if not bars: # Handle case where symbol exists but data list is empty
            print(f"Warning: No bars in the data list for symbol {symbol}.")
            return np.empty((EXPECTED_NUM_FEATURES, 0), dtype=object)

        for bar in bars:
            current_bar_data = []
            try:
                # Iterate through categories defined in config to ensure correct order
                for category in config.CATEGORIES:
                    if category == 'date':
                        # Get timestamp and ensure it's timezone-aware UTC
                        ts = bar.timestamp
                        if ts.tzinfo is None:
                            ts_aware = ts.replace(tzinfo=timezone.utc)
                        else:
                            ts_aware = ts.astimezone(timezone.utc)
                        current_bar_data.append(ts_aware)
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
                        print(f"Warning: Unhandled category '{category}' in config.CATEGORIES. Skipping.")
                        current_bar_data.append(None) # Append None as placeholder

                # Ensure the bar had the correct number of columns generated
                if len(current_bar_data) == EXPECTED_NUM_FEATURES:
                    processed_data_rows.append(current_bar_data)
                else:
                     print(f"Warning: Skipping bar due to mismatched feature count. Expected {EXPECTED_NUM_FEATURES}, got {len(current_bar_data)}. Bar: {bar}")

            except AttributeError as e:
                print(f"Error accessing attribute for bar data: {e}. Bar: {bar}. Skipping bar.")
            except Exception as e:
                print(f"Unexpected error processing bar: {e}. Bar: {bar}. Skipping bar.")

    else:
        print(f"Warning: No data key found for symbol '{symbol}' in Alpaca response.")
        return np.empty((EXPECTED_NUM_FEATURES, 0), dtype=object)

    # --- Final check and array conversion ---
    if not processed_data_rows:
        print(f"Warning: No valid bars were processed for {symbol} after fetching.")
        return np.empty((EXPECTED_NUM_FEATURES, 0), dtype=object)

    # Convert the list of rows (bars) into a NumPy array
    # Shape will be (num_bars, num_features_incl_date)
    data_array_bars_as_rows = np.array(processed_data_rows, dtype=object)

    # Transpose the array to get the desired shape: (num_features_incl_date, num_bars)
    final_array = data_array_bars_as_rows.T

    print(f"Successfully processed {final_array.shape[1]} bars for {symbol}. Final shape: {final_array.shape}")
    return final_array

# --- END OF REVISED _get_historical_bars_threaded ---

# Note: You will also need the _formattime function (simplified as suggested previously
# to handle datetime objects primarily) and the rest of the data_handler.py functions
# (_generate_sequences, get_training_data, get_live_data). Make sure get_training_data
# and get_live_data now call this revised _get_historical_bars_threaded function
# without the extra client_id, time_window, bar_size arguments.

def _generate_sequences(symbarray, seq_len, max_horizon, input_size):
    """Generates encoder, decoder input, and target sequences from normalized data."""
    sequences_enc = []
    sequences_dec_in = []
    sequences_tgt = []
    symbol_sequences = 0

    # Assumes symbarray shape is (num_features + 1, num_bars), with date at index 0
    # Normalize based on features only (index 1 onwards)
    norm_values = np.empty_like(symbarray[1:], dtype=float)
    if symbarray.shape[1] > 1:  # Need at least 2 points to calculate std dev
        # Ensure float type for calculations
        data_values = symbarray[1:].astype(float)

        ##volume_index = 4
        ##if data_values.shape[0] > volume_index: # Check if volume index exists
        ##    data_values[volume_index, :] = np.log1p(data_values[volume_index, :])

        mean = data_values.mean(axis=1, keepdims=True)
        std = data_values.std(axis=1, keepdims=True) + \
            1e-6  # Avoid division by zero
        norm_values = (data_values - mean) / std
    elif symbarray.shape[1] == 1:
        # Normalize single point to 0
        norm_values = np.zeros_like(symbarray[1:], dtype=float)
    else:  # No data points
        return [], [], [], 0  # Return empty lists if no data

    # Ensure enough data points: seq_len for input + horizon for output
    if norm_values.shape[1] >= seq_len + max_horizon:
        num_sequences_possible = norm_values.shape[1] - seq_len - max_horizon + 1

        for i in range(num_sequences_possible):
            start_idx = i
            encoder_end_idx = i + seq_len
            decoder_end_idx = encoder_end_idx + max_horizon

            # Encoder Input: Shape (input_size, seq_len)
            encoder_input_data = norm_values[:, start_idx:encoder_end_idx]

            # Target Output: Shape (input_size, horizon)
            target_data = norm_values[:, encoder_end_idx:decoder_end_idx]

            # Decoder Input (Teacher Forcing): Shape (input_size, horizon)
            last_encoder_val = norm_values[:,
                                           encoder_end_idx-1:encoder_end_idx]
            target_shifted = norm_values[:, encoder_end_idx:decoder_end_idx-1]
            decoder_input_data = np.concatenate(
                (last_encoder_val, target_shifted), axis=1)

            # Append flattened versions
            sequences_enc.append(encoder_input_data.flatten())
            sequences_dec_in.append(decoder_input_data.flatten())
            sequences_tgt.append(target_data.flatten())
            symbol_sequences += 1

    return sequences_enc, sequences_dec_in, sequences_tgt, symbol_sequences


def get_training_data():
    """
    Loads or fetches training data for all symbols, processes into sequences,
    and splits into training and validation sets.
    Returns (train_data_tuple, val_data_tuple), where each tuple is
    (X_enc_tensor, X_dec_in_tensor, Y_tgt_tensor).
    """
    global skipped_symbols_list
    skipped_symbols_list = []

    seq_len = config.SEQ_LEN
    max_horizon = config.N_PRED_TRAINING_MAX
    input_size = config.INPUT_SIZE
    force_fetch = not config.LOAD_TRAINING_DATA

    if not force_fetch and os.path.isfile(config.TRAINING_DATA_PATH):
        print(
            f"Loading pre-generated training data from: {config.TRAINING_DATA_PATH}")
        try:
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
            print(
                f"Error loading data from {config.TRAINING_DATA_PATH}: {e}. Regenerating.")
            force_fetch = True

    if force_fetch:
        print("Fetching and processing training data...")
        all_sequences_encoder = []
        all_sequences_decoder_input = []
        all_sequences_target = []
        total_sequences_generated = 0
        Nticks = {}

        for i, symbol in enumerate(config.SYMBOLS):
            #client_id = config.BASE_CLIENT_ID_TRAIN + i
            print(f"\n--- Fetching data for {symbol} ---")
            bars = _get_historical_bars_threaded(
                symbol)

            if bars.shape[1] > 0:
                Nticks[symbol] = bars.shape[1]
                print(f"Generating sequences for {symbol}...")
                enc, dec_in, tgt, count = _generate_sequences(
                    bars, seq_len, max_horizon, input_size)
                if count > 0:
                    all_sequences_encoder.extend(enc)
                    all_sequences_decoder_input.extend(dec_in)
                    all_sequences_target.extend(tgt)
                    total_sequences_generated += count
                    print(f"Generated {count} sequences for {symbol}.")
                else:
                    print(
                        f"Insufficient data ({bars.shape[1]} bars) for sequences for {symbol}.")
                    skipped_symbols_list.append(symbol)
            else:
                skipped_symbols_list.append(symbol)
            time.sleep(0.5)  # Small delay

        print("\n--- Data Fetch Summary ---")
        print("Fetched bars per symbol:", Nticks)
        print("Skipped symbols:", skipped_symbols_list)
        print(f"Total sequences generated: {total_sequences_generated}")

        if not all_sequences_encoder:
            print("ERROR: No sequences generated. Cannot proceed with training.")
            # Return empty tensors with correct final dimension
            empty_shape_enc = (0, seq_len * input_size)
            empty_shape_dec = (0, max_horizon * input_size)
            return (torch.empty(empty_shape_enc), torch.empty(empty_shape_dec), torch.empty(empty_shape_dec)), \
                   (torch.empty(empty_shape_enc), torch.empty(
                       empty_shape_dec), torch.empty(empty_shape_dec))

        X_enc = np.array(all_sequences_encoder)
        X_dec_in = np.array(all_sequences_decoder_input)
        Y_tgt = np.array(all_sequences_target)

        indices = np.arange(len(X_enc))
        np.random.shuffle(indices)
        X_enc, X_dec_in, Y_tgt = X_enc[indices], X_dec_in[indices], Y_tgt[indices]

        valsplit_index = int(len(X_enc) * (1 - config.VAL_SPLIT_RATIO))
        Xtrain_enc, Xval_enc = X_enc[:valsplit_index], X_enc[valsplit_index:]
        Xtrain_dec_in, Xval_dec_in = X_dec_in[:
                                              valsplit_index], X_dec_in[valsplit_index:]
        Ytrain_tgt, Yval_tgt = Y_tgt[:valsplit_index], Y_tgt[valsplit_index:]

        print(f"\n--- Data Split ---")
        print(f"Training sequences: {len(Xtrain_enc)}")
        print(f"Validation sequences: {len(Xval_enc)}")

        try:
            print(
                f"Saving generated training data to: {config.TRAINING_DATA_PATH}")
            np.savez(config.TRAINING_DATA_PATH,
                     Xtrain_enc=Xtrain_enc, Xtrain_dec_in=Xtrain_dec_in, Ytrain_tgt=Ytrain_tgt,
                     Xval_enc=Xval_enc, Xval_dec_in=Xval_dec_in, Yval_tgt=Yval_tgt)
        except Exception as e:
            print(f"Error saving data to {config.TRAINING_DATA_PATH}: {e}")

    # Convert to tensors
    Xtrain_enc_t = torch.tensor(Xtrain_enc, dtype=torch.float32)
    Xtrain_dec_in_t = torch.tensor(Xtrain_dec_in, dtype=torch.float32)
    Ytrain_tgt_t = torch.tensor(Ytrain_tgt, dtype=torch.float32)
    Xval_enc_t = torch.tensor(Xval_enc, dtype=torch.float32)
    Xval_dec_in_t = torch.tensor(Xval_dec_in, dtype=torch.float32)
    Yval_tgt_t = torch.tensor(Yval_tgt, dtype=torch.float32)

    train_data = (Xtrain_enc_t, Xtrain_dec_in_t, Ytrain_tgt_t)
    val_data = (Xval_enc_t, Xval_dec_in_t, Yval_tgt_t)

    return train_data, val_data


def get_live_data(symbol):
    """
    Fetches the latest data for prediction, normalizes it, and returns
    the normalized sequence, full bars, and normalization stats.
    """
    print(f"Fetching live data for prediction symbol: {symbol}")
    seq_len = config.SEQ_LEN
    live_data_cache_path = config.LIVE_DATA_CACHE_PATH_TEMPLATE.format(
        symbol=symbol)

    # Simple cache check (can be disabled by setting LOAD_TRAINING_DATA=False in config)
    # Reuse flag for caching live data fetch
    load_from_cache = config.LOAD_TRAINING_DATA
    bars = None

    if load_from_cache and os.path.exists(live_data_cache_path):
        try:
            print(f"Loading cached live data from {live_data_cache_path}")
            bars = np.load(live_data_cache_path)
        except Exception as e:
            print(
                f"Could not load cached live data: {e}. Fetching fresh data.")
            bars = None  # Ensure fetch if load fails

    if bars is None:
        print(
            f"Fetching live data using window '{config.FREQ}'...")
        bars = _get_historical_bars_threaded(
            symbol)
        if bars.shape[1] > 0:
            try:
                print(
                    f"Saving fetched live data ({bars.shape[1]} bars) to {live_data_cache_path}")
                np.save(live_data_cache_path, bars)
            except Exception as e:
                print(f"Error saving live data cache: {e}")
        else:
            print(f"Warning: Fetched 0 bars for {symbol}.")
            # Cannot proceed without data
            raise ValueError(f"Failed to fetch any live data for {symbol}.")

    print(f"Total bars fetched for live sequence base: {bars.shape[1]}")
    if bars.shape[1] < seq_len:
        raise ValueError(
            f"Not enough live data for {symbol}. Need {seq_len}, got {bars.shape[1]}. "
            f"Check fetch window ('{config.FREQ}') or data availability.")

    # Take MOST RECENT seq_len bars
    bars_input_sequence = bars[:, -seq_len:]
    print(
        f"Using the last {bars_input_sequence.shape[1]} bars for the input sequence.")

    # Normalize using stats ONLY from this input sequence
    norm_bars_input = bars_input_sequence.copy()
    data_values = norm_bars_input[1:].astype(
        float)  # Features only, ensure float
    
    ##volume_index = 4
    ##if data_values.shape[0] > volume_index: # Check if volume index exists
    ##    data_values[volume_index, :] = np.log1p(data_values[volume_index, :])

    mean = data_values.mean(axis=1, keepdims=True)
    std = data_values.std(axis=1, keepdims=True) + 1e-6
    norm_values = (data_values - mean) / std  # Shape: (input_size, seq_len)

    # Transpose to (seq_len, input_size) for predictor input format
    norm_sequence_transposed = norm_values.T

    return norm_sequence_transposed, bars, (mean, std)
