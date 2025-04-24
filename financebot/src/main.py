import numpy as np
from datetime import timedelta,datetime
import os 
import sys
srcpath=os.path.dirname(os.path.realpath(__file__))
botdir="/".join(srcpath.split("/")[:-2])
sys.path.append(botdir)

# Import configuration and modules
import financebot.src.config as config
from financebot.src.model import TransformerModel
from financebot.src.data_handler import get_training_data, get_live_data
from financebot.src.trainer import batchify_data, fit
from financebot.src.predictor import predict_autoregressive
from financebot.src.plotting_utils import plot_loss, plot_prediction


def main_workflow():
    """ Main orchestration function """
    print("--- Starting Main Workflow ---")

    # --- 1. Initialization ---
    print("Initializing model and optimizer...")
    model = TransformerModel().to(config.DEVICE)
    optimizer = config.get_optimizer(model.parameters())
    loss_fn = config.LOSS_FN  # Using loss_fn from config

    # --- 2. Training or Loading ---
    if config.LOAD_SAVED_MODEL:
        print("Attempting to load saved model...")
        if not model.load_model():
            print("ERROR: Failed to load model. Exiting.")
            return  # Exit if loading failed when requested
    else:
        print("Starting training process...")
        # Get training and validation data
        train_data, val_data = get_training_data()
        # Check if data is available
        if train_data[0].nelement() == 0 or val_data[0].nelement() == 0:
            print(
                "ERROR: No training or validation data sequences available. Cannot train. Exiting.")
            return  # Exit if no data to train on

        # Create dataloaders
        train_dataloader = batchify_data(train_data)
        val_dataloader = batchify_data(val_data)

        if not train_dataloader or not val_dataloader:
            print("ERROR: Failed to create dataloaders. Cannot train. Exiting.")
            return  # Exit if loaders are empty

        # Train the model
        train_loss_list, validation_loss_list = fit(
            model, train_dataloader, val_dataloader, optimizer, loss_fn)

        # Save the trained model
        model.save_model()

        # Plot training loss
        plot_loss(train_loss_list, validation_loss_list)

    # --- 3. Prediction ---
    print(f"\n--- Starting Prediction for {config.PREDICTION_SYMBOL} ---")
    try:
        # Get the latest data sequence for prediction
        # input_norm shape: (seq_len, input_size)
        # bars_full shape: (n_features_incl_date, n_bars_fetched)
        # norm_stats: (mean, std) shapes: (input_size, 1)
        input_norm, bars_full, (mean, std) = get_live_data(
            config.PREDICTION_SYMBOL)

        # Perform autoregressive prediction
        # predictions_norm_list: List of N arrays, each (input_size,)
        predictions_norm_list = predict_autoregressive(model, input_norm)

        if not predictions_norm_list:
            print("Prediction step returned no results. Exiting.")
            return

        # Denormalize predictions
        predictions_norm_array = np.array(
            predictions_norm_list)  # Shape: (N_pred, input_size)
        # Ensure mean/std are broadcastable: mean.T/std.T shape (1, input_size)
        predictions_denorm = predictions_norm_array * \
            std.T + mean.T  # Shape: (N_pred, input_size)
        print("Predictions generated and denormalized.")

        ##volume_index = 4
        ##if predictions_denorm.shape[1] > volume_index: # Check if volume column exists
        ##    predictions_denorm[:, volume_index] = np.expm1(predictions_denorm[:, volume_index])
        ##    # Optional: Clip volume at zero AFTER inverse transform just in case of tiny negatives
        ##    predictions_denorm[:, volume_index] = np.maximum(predictions_denorm[:, volume_index], 0)
    except ValueError as e:
        print(f"ERROR during prediction data fetching: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed trace
        return


    # --- 4. Plotting Prediction ---
    print("Preparing data for plotting predictions...")
    # bars_full has shape (features_incl_date, num_bars)
    # Get the date row (index 0) for the last seq_len entries
    input_dates_raw = bars_full[0, -config.SEQ_LEN:]

    # Directly use the datetime objects from the raw data, as they should be correct.
    # Filter out any potential None values just in case something went wrong earlier.
    input_dates = np.array([d for d in input_dates_raw if isinstance(d, datetime)])

    # Check if we actually have dates after filtering
    if len(input_dates) == 0:
        print("Error: No valid datetime objects found in the input sequence dates.")
        # Cannot proceed with plotting or date estimation
        pred_dates = [] # Ensure pred_dates is empty
    else:
        print(f"Successfully extracted {len(input_dates)} datetime objects for input sequence.")
        # Estimate future dates *only if* we have input_dates
        pred_dates = []
        if len(input_dates) >= 2:
            # Use the interval between the last two valid input dates
            time_delta = input_dates[-1] - input_dates[-2]
            # Basic check for non-zero interval
            if time_delta.total_seconds() > 0:
                 pred_dates = [input_dates[-1] + time_delta * (i + 1) for i in range(config.N_PRED_INFERENCE)]
            else:
                 print(f"Warning: Time delta between last two dates is zero or negative ({time_delta}). Cannot reliably estimate prediction dates.")
                 # Fallback? Or just leave pred_dates empty? Leaving empty for now.

        elif len(input_dates) == 1: # Fallback: Use configured bar size
            print("Warning: Only one valid input date found. Estimating future dates using config.FREQ.")
            try:
                # Use the TimeFrame object directly from config
                time_delta = config.FREQ.to_timedelta()
                pred_dates = [input_dates[-1] + time_delta * (i + 1) for i in range(config.N_PRED_INFERENCE)]
            except Exception as e:
                print(f"Error creating timedelta from config.FREQ: {e}. Falling back to hourly.")
                time_delta = timedelta(hours=1) # Final fallback
                pred_dates = [input_dates[-1] + time_delta * (i + 1) for i in range(config.N_PRED_INFERENCE)]


    # Plot if we have dates for predictions
    if pred_dates: # This implicitly checks if len(input_dates) was > 0
        # Pass bars_full (transposed), input_dates (filtered), pred_dates (estimated)
        plot_prediction(bars_full, predictions_denorm, input_dates, pred_dates)
    else:
        # This message will now trigger if len(input_dates) == 0 OR if date estimation failed
        print("Skipping prediction plot due to missing date information or estimation failure.")

    print("\n--- Main Workflow Finished ---")


if __name__ == "__main__":
    main_workflow()
