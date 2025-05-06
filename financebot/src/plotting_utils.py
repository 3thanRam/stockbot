# This script contains utility functions for plotting results,
# including training/validation loss and stock data with predictions.

import matplotlib.pyplot as plt # For creating plots
import matplotlib.dates as mdates # For formatting dates on plot axes
from matplotlib.patches import Rectangle # For drawing shapes like rectangles on plots
import numpy as np # For numerical operations and array handling
import financebot.src.config as config # Import application configuration settings

# Import the time formatting helper function from data_handler
# Renamed locally for convenience
from financebot.src.data_handler import _formattime as formattime


# Function to plot and save the training and validation loss over epochs
def plot_loss(train_loss_list, validation_loss_list, path=config.LOSS_PLOT_PATH):
    """Plots and saves the training and validation loss."""
    # Create a new figure for the plot
    plt.figure(figsize=(10, 5))
    # Plot the training loss values
    plt.plot(train_loss_list, label="Training Loss")
    # Plot the validation loss values
    plt.plot(validation_loss_list, label="Validation Loss")
    # Set axis labels and title
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend() # Add a legend to identify lines
    plt.title("Training and Validation Loss")
    plt.grid(True, linestyle=':') # Add a grid
    # Attempt to save the figure to the specified path
    try:
        plt.savefig(path)
        print(f"Loss plot saved to {path}")
    except Exception as e:
        print(f"Error saving loss plot to {path}: {e}")
    # plt.show() # Optional: Uncomment to display the plot interactively
    plt.close() # Close the figure to free up memory


# Function to plot historical stock data and the model's predictions
def plot_prediction(bars_full, predictions_denorm, input_dates, pred_dates, path=config.PREDICTION_PLOT_PATH):
    """Plots historical data and predictions."""
    # Retrieve necessary parameters from configuration
    symbol = config.PREDICTION_SYMBOL
    categories = config.CATEGORIES
    seq_len = config.SEQ_LEN
    n_pred = config.N_PRED_INFERENCE

    # Expected data shapes:
    # predictions_denorm: NumPy array, shape (n_pred, input_size)
    # bars_full: NumPy array, shape (n_features_incl_date, n_bars)

    # Create a figure with two subplots stacked vertically, sharing the x-axis scale initially
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False) # sharex=False because date ranges are different
    # Define a list of colors to cycle through for plotting different features
    colors = ["cyan", "red", "green", "magenta", "yellow",
              "black", "blue", "orange"]
    # Define line styles for historical and predicted data
    reallinestyle = "-"
    predlinestyle = "--"

    # --- Plot 1: Zoomed view ---
    # Focus on the input sequence and prediction horizon
    # Determine the window size for the zoomed plot (half of seq_len, but not more than available input dates)
    zoom_window = min(seq_len // 2, len(input_dates))
    # Get the input dates for the zoomed window
    zoom_input_dates = input_dates[-zoom_window:]
    # Concatenate zoomed input dates with prediction dates for setting x-limits later
    zoom_all_dates = np.concatenate(
        (zoom_input_dates, pred_dates)) if pred_dates else zoom_input_dates

    # Set the title for the zoomed subplot
    axes[0].set_title(f"{symbol} - Zoomed Prediction Horizon ({n_pred} steps)")
    min_vals_zoom, max_vals_zoom = [], [] # Lists to track min/max y-values for setting y-limits

    # Iterate through each data category (feature) to plot
    for s, categ in enumerate(categories):
        if categ == "date": # Skip the date category
            continue
        # Select a color for the current feature, cycling through the colors list
        color = colors[(s - 1) % len(colors)] # s-1 because date is skipped

        # --- Plot Real Data (Historical) for Zoom ---
        real_data_full = bars_full[s]  # Get the full historical data for this category (shape n_bars,)
        # Determine the start index in the full data corresponding to the start of the input sequence
        start_idx_full = max(0, bars_full.shape[1] - seq_len)
        # Extract the real data corresponding to the input sequence dates
        real_data_input = real_data_full[start_idx_full:]
        # Extract the real data for the specific zoom window duration
        real_data_zoom = real_data_input[-zoom_window:]

        # Plot the historical data within the zoom window if dates and data lengths match
        if len(zoom_input_dates) == len(real_data_zoom):
            axes[0].plot(zoom_input_dates, real_data_zoom, label=f"History: {categ}",
                         linestyle=reallinestyle, color=color, alpha=0.7)
            # Track min/max values for y-limit setting if data exists
            if real_data_zoom.size > 0:
                min_vals_zoom.append(np.min(real_data_zoom))
                max_vals_zoom.append(np.max(real_data_zoom))

        # --- Plot Predicted Data for Zoom ---
        # Check if predictions exist and have data for the current category
        if predictions_denorm.size > 0 and predictions_denorm.shape[1] > (s-1):
            pred_data_categ = predictions_denorm[:, s-1]  # Get predictions for this category (shape n_pred,)
            # Plot the predicted data if prediction dates and data lengths match
            if len(pred_dates) == len(pred_data_categ):
                axes[0].plot(pred_dates, pred_data_categ, label=f"Predicted: {categ}",
                             linestyle=predlinestyle, color=color, alpha=0.9)
                # Track min/max values for y-limit setting
                min_vals_zoom.append(np.min(pred_data_categ))
                max_vals_zoom.append(np.max(pred_data_categ))

    axes[0].legend(fontsize='small') # Add legend to the zoomed plot
    axes[0].grid(True, linestyle=':') # Add grid to the zoomed plot
    axes[0].set_ylabel("Price") # Set y-axis label
    # Set y-axis limits based on min/max values in the zoomed data with small padding
    if min_vals_zoom and max_vals_zoom:
        axes[0].set_ylim(np.min(min_vals_zoom)*0.99,
                         np.max(max_vals_zoom)*1.01)
    # Set date formatter for the x-axis to show only time (assuming hourly focus)
    axes[0].xaxis.set_major_formatter(
        mdates.DateFormatter('%H:%M'))
    # Use an auto locator for major ticks on the x-axis
    axes[0].xaxis.set_major_locator(
        mdates.AutoDateLocator(minticks=3, maxticks=7))

    # --- Plot 2: Full Context ---
    # Show the entire historical data range and the prediction
    axes[1].set_title(f"{symbol} - Full History and Prediction")
    min_vals_full, max_vals_full = [], [] # Lists to track min/max y-values for the full plot

    # Get ALL dates from the full historical data using the formatting helper
    all_dates = np.array([formattime(d) for d in bars_full[0]], dtype=object) # Ensure dtype object for datetime/None
    # Create a mask to filter out any potential None values from formatting errors
    valid_dates_mask = all_dates != None
    # Apply the mask to get only valid dates
    all_dates = all_dates[valid_dates_mask]

    # Iterate through each data category (feature) to plot on the full context axes
    for s, categ in enumerate(categories):
        if categ == "date": # Skip the date category
            continue
        # Select a color for the current feature
        color = colors[(s - 1) % len(colors)]

        # --- Plot Full Historical Data ---
        # Get the full historical data for this category and apply the valid dates mask
        real_data_full = bars_full[s][valid_dates_mask]
        # Plot the full historical data if dates and data lengths match
        if len(all_dates) == len(real_data_full):
            axes[1].plot(all_dates, real_data_full, label=f"History: {categ}",
                         linestyle=reallinestyle, color=color, alpha=0.6)
            # Track min/max values for y-limit setting
            if real_data_full.size > 0:
                min_vals_full.append(np.min(real_data_full))
                max_vals_full.append(np.max(real_data_full))

        # --- Plot Predicted Data (again on full plot) ---
        # Plot the same predicted data as in the zoomed plot
        if predictions_denorm.size > 0 and predictions_denorm.shape[1] > (s-1):
            pred_data_categ = predictions_denorm[:, s-1]
            if len(pred_dates) == len(pred_data_categ):
                axes[1].plot(pred_dates, pred_data_categ, label=f"Predicted: {categ}",
                             linestyle=predlinestyle, color=color, alpha=0.9, linewidth=1.5)
                # Track min/max values for y-limit setting
                min_vals_full.append(np.min(pred_data_categ))
                max_vals_full.append(np.max(pred_data_categ))

    # --- Add Context Rectangle ---
    # Draw a rectangle on the full plot to indicate the area shown in the zoomed plot
    if (zoom_input_dates.size > 0  # Check if numpy array (input dates) is not empty
        and len(pred_dates) > 0      # Check if list (prediction dates) is not empty
        and min_vals_zoom # Check if zoom y-values were tracked
        and max_vals_zoom): # Check if zoom y-values were tracked

        # Determine the start and end dates for the rectangle using zoomed dates and prediction dates
        rect_start_date = zoom_input_dates[0]
        rect_end_date = pred_dates[-1]
        # Determine the y-limits for the rectangle based on the zoomed plot's data range
        rect_ymin = np.min(min_vals_zoom) * 0.99
        rect_ymax = np.max(max_vals_zoom) * 1.01
        try: # Use a try-except block as date2num conversion can sometimes fail
            # Add the rectangle patch to the full context axes
            axes[1].add_patch(Rectangle((mdates.date2num(rect_start_date), rect_ymin), # (x, y) bottom-left corner
                                        mdates.date2num(
                                            rect_end_date) - mdates.date2num(rect_start_date), # width
                                        rect_ymax - rect_ymin, # height
                                        edgecolor='grey', facecolor='none', lw=1, ls='--')) # Style
        except Exception as e:
            print(f"Could not draw context rectangle: {e}")

    axes[1].legend(fontsize='small') # Add legend to the full plot
    axes[1].grid(True, linestyle=':') # Add grid to the full plot
    axes[1].set_ylabel("Price") # Set y-axis label
    # Set y-axis limits based on min/max values in the full data with small padding
    if min_vals_full and max_vals_full:
        axes[1].set_ylim(np.min(min_vals_full)*0.98,
                         np.max(max_vals_full)*1.02)

    # Use auto locator and concise formatter for better date display on the full x-axis
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(formatter)

    fig.autofmt_xdate() # Automatically format dates on x-axis to prevent overlap
    plt.tight_layout() # Adjust subplot parameters for a tight layout
    # Attempt to save the entire figure
    try:
        plt.savefig(path)
        print(f"Prediction plot saved to {path}")
    except Exception as e:
        print(f"Error saving prediction plot to {path}: {e}")
    plt.show()  # Optional: Uncomment to display the plot interactively
    plt.close() # Close the figure