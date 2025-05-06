# This script orchestrates the main workflow for the finance bot,
# including model training/loading, live data fetching, prediction, and plotting.

import numpy as np # For numerical operations, especially with data manipulation and predictions
from datetime import timedelta,datetime # For handling time and date calculations
import os # For interacting with the operating system (path manipulation)
import sys # For system-specific parameters and functions

# Determine the script's directory and navigate up to the project root
# Then add the project root directory to the system path to allow importing modules
srcpath=os.path.dirname(os.path.realpath(__file__))
botdir="/".join(srcpath.split("/")[:-2]) # Go up two directories from the current file's directory ('financebot/src')
sys.path.append(botdir)

# Import application-specific configuration and modules from financebot.src
import financebot.src.config as config # Application configuration settings
from financebot.src.model import TransformerModel # The Transformer model definition
from financebot.src.data_handler import get_training_data, get_live_data # Functions for data loading
from financebot.src.trainer import batchify_data, fit # Functions for data preparation and model training
from financebot.src.predictor import predict_autoregressive # Function for generating predictions
from financebot.src.plotting_utils import plot_loss, plot_prediction # Functions for plotting results


# Main function to run the complete workflow
def main_workflow():
    """ Main orchestration function """
    print("--- Starting Main Workflow ---")

    # --- 1. Initialization ---
    print("Initializing model and optimizer...")
    # Instantiate the Transformer model and move it to the configured device (CPU/GPU)
    model = TransformerModel().to(config.DEVICE)
    # Initialize the optimizer specified in the configuration
    optimizer = config.get_optimizer(model.parameters())
    # Get the loss function from configuration
    loss_fn = config.LOSS_FN

    # --- 2. Training or Loading ---
    # Check configuration if a saved model should be loaded instead of training
    if config.LOAD_SAVED_MODEL:
        print("Attempting to load saved model...")
        # Attempt to load the model state; exit if loading fails
        if not model.load_model():
            print("ERROR: Failed to load model. Exiting.")
            return
    else:
        print("Starting training process...")
        # Fetch training and validation data
        train_data, val_data = get_training_data()
        # Check if fetched data contains actual sequences
        if train_data[0].nelement() == 0 or val_data[0].nelement() == 0:
            print(
                "ERROR: No training or validation data sequences available. Cannot train. Exiting.")
            return

        # Prepare data into batches for training and validation
        train_dataloader = batchify_data(train_data)
        val_dataloader = batchify_data(val_data)

        # Check if dataloaders were created successfully
        if not train_dataloader or not val_dataloader:
            print("ERROR: Failed to create dataloaders. Cannot train. Exiting.")
            return

        # Train the model using the fit function
        train_loss_list, validation_loss_list = fit(
            model, train_dataloader, val_dataloader, optimizer, loss_fn)

        # Save the trained model state
        model.save_model()

        # Plot the training and validation loss curves
        plot_loss(train_loss_list, validation_loss_list)

    # --- 3. Prediction ---
    # Section for fetching live data and performing prediction
    print(f"\n--- Starting Prediction for {config.PREDICTION_SYMBOL} ---")
    try:
        # Fetch the latest live data sequence required for the model input
        # input_norm: Normalized input data sequence (SEQ_LEN, input_size)
        # bars_full: Raw historical bars fetched (n_features_incl_date, n_bars_fetched)
        # (mean, std): Normalization statistics used
        input_norm, bars_full, (mean, std) = get_live_data(
            config.PREDICTION_SYMBOL)

        # Generate future predictions using the trained model in an autoregressive manner
        # predictions_norm_list: List containing normalized predictions for N_PRED_INFERENCE steps
        predictions_norm_list = predict_autoregressive(model, input_norm)

        # Check if any predictions were returned
        if not predictions_norm_list:
            print("Prediction step returned no results. Exiting.")
            return

        # Convert the list of prediction arrays into a single numpy array
        predictions_norm_array = np.array(
            predictions_norm_list)  # Shape: (N_pred, input_size)
        # Denormalize the predictions back to original scale using the saved mean and std
        # Broadcasting mean.T/std.T (shape 1, input_size) with predictions_norm_array (shape N_pred, input_size)
        predictions_denorm = predictions_norm_array * \
            std.T + mean.T  # Final shape: (N_pred, input_size)
        print("Predictions generated and denormalized.")

        ## # --- Optional: Handle specific feature denormalization (e.g., volume) ---
        ## # This block is commented out but shows how to handle features that might require different denormalization (like log transform)
        ## volume_index = 4 # Assuming volume is the 5th feature (index 4)
        ## if predictions_denorm.shape[1] > volume_index: # Check if the volume column exists in the prediction output
        ##    # Apply inverse log transform (e.g., expm1 for log1p)
        ##    predictions_denorm[:, volume_index] = np.expm1(predictions_denorm[:, volume_index])
        ##    # Ensure volume predictions are non-negative after inverse transform
        ##    predictions_denorm[:, volume_index] = np.maximum(predictions_denorm[:, volume_index], 0)

    # Catch specific ValueError during data fetching
    except ValueError as e:
        print(f"ERROR during prediction data fetching: {e}")
        return
    # Catch any other unexpected errors during the prediction process
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        # import traceback; traceback.print_exc() # Uncomment this line to print a detailed error trace
        return


    # --- 4. Plotting Prediction ---
    # Section for preparing data and plotting the results
    print("Preparing data for plotting predictions...")
    # bars_full has shape (features_incl_date, num_bars); dates are in the first row (index 0)
    # Extract the date row and get the last SEQ_LEN entries (corresponding to the input sequence)
    input_dates_raw = bars_full[0, -config.SEQ_LEN:]

    # Filter out any non-datetime objects from the extracted input dates
    input_dates = np.array([d for d in input_dates_raw if isinstance(d, datetime)])

    # Check if we successfully extracted any valid datetime objects for the input sequence
    if len(input_dates) == 0:
        print("Error: No valid datetime objects found in the input sequence dates.")
        # If no valid input dates, we cannot estimate prediction dates
        pred_dates = [] # Ensure the prediction dates list is empty
    else:
        print(f"Successfully extracted {len(input_dates)} datetime objects for input sequence.")
        # --- Estimate future dates for the prediction line ---
        pred_dates = [] # Initialize prediction dates list
        if len(input_dates) >= 2:
            # If at least two input dates are available, calculate the time delta between the last two
            time_delta = input_dates[-1] - input_dates[-2]
            # Check if the calculated time delta is positive (valid interval)
            if time_delta.total_seconds() > 0:
                 # Generate prediction dates by adding the time delta sequentially to the last input date
                 pred_dates = [input_dates[-1] + time_delta * (i + 1) for i in range(config.N_PRED_INFERENCE)]
            else:
                 # Warning if the time delta is invalid (e.g., zero or negative)
                 print(f"Warning: Time delta between last two dates is zero or negative ({time_delta}). Cannot reliably estimate prediction dates.")
                 # If delta is invalid, pred_dates remains empty for now.

        elif len(input_dates) == 1: # Fallback: If only one input date is available
            print("Warning: Only one valid input date found. Estimating future dates using config.FREQ.")
            try:
                # Attempt to get a timedelta from the configured frequency (e.g., TimeFrame.Hour)
                time_delta = config.FREQ.to_timedelta()
                # Generate prediction dates using the configured frequency timedelta
                pred_dates = [input_dates[-1] + time_delta * (i + 1) for i in range(config.N_PRED_INFERENCE)]
            except Exception as e:
                # Fallback if converting config.FREQ to timedelta fails
                print(f"Error creating timedelta from config.FREQ: {e}. Falling back to hourly.")
                time_delta = timedelta(hours=1) # Use a fixed 1-hour delta as a final fallback
                pred_dates = [input_dates[-1] + time_delta * (i + 1) for i in range(config.N_PRED_INFERENCE)]


    # Proceed with plotting only if prediction dates were successfully generated
    if pred_dates: # This check implicitly covers the case where len(input_dates) was 0 or date estimation failed
        # Call the plotting utility function to visualize the historical data and the predictions
        # Pass the full raw historical data, the denormalized predictions, the input dates, and the estimated prediction dates
        plot_prediction(bars_full, predictions_denorm, input_dates, pred_dates)
    else:
        # Print a message indicating plotting was skipped due to lack of valid date information
        print("Skipping prediction plot due to missing date information or estimation failure.")

    print("\n--- Main Workflow Finished ---")


# Standard Python entry point
# This block ensures main_workflow() is called when the script is executed directly
if __name__ == "__main__":
    main_workflow()