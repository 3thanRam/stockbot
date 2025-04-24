import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
import financebot.src.config as config

# Helper to reuse date formatting from data_handler
from financebot.src.data_handler import _formattime as formattime


def plot_loss(train_loss_list, validation_loss_list, path=config.LOSS_PLOT_PATH):
    """Plots and saves the training and validation loss."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label="Training Loss")
    plt.plot(validation_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, linestyle=':')
    try:
        plt.savefig(path)
        print(f"Loss plot saved to {path}")
    except Exception as e:
        print(f"Error saving loss plot to {path}: {e}")
    # plt.show() # Optional: Show plot interactively
    plt.close()


def plot_prediction(bars_full, predictions_denorm, input_dates, pred_dates, path=config.PREDICTION_PLOT_PATH):
    """Plots historical data and predictions."""
    symbol = config.PREDICTION_SYMBOL
    categories = config.CATEGORIES
    seq_len = config.SEQ_LEN
    n_pred = config.N_PRED_INFERENCE

    # predictions_denorm expected shape: (n_pred, input_size)
    # bars_full expected shape: (n_features_incl_date, n_bars)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    colors = ["cyan", "red", "green", "magenta", "yellow",
              "black", "blue", "orange"]  # Extended colors
    reallinestyle = "-"
    predlinestyle = "--"

    # --- Plot 1: Zoomed view ---
    # Show half or less if input is short
    zoom_window = min(seq_len // 2, len(input_dates))
    zoom_input_dates = input_dates[-zoom_window:]
    zoom_all_dates = np.concatenate(
        (zoom_input_dates, pred_dates)) if pred_dates else zoom_input_dates

    axes[0].set_title(f"{symbol} - Zoomed Prediction Horizon ({n_pred} steps)")
    min_vals_zoom, max_vals_zoom = [], []

    for s, categ in enumerate(categories):
        if categ == "date":
            continue
        # Cycle through colors for features
        color = colors[(s - 1) % len(colors)]

        # Real data for zoom plot
        real_data_full = bars_full[s]  # Data for this category
        # Ensure indices are valid
        # Start index for the input seq in full data
        start_idx_full = max(0, bars_full.shape[1] - seq_len)
        # Data corresponding to input_dates
        real_data_input = real_data_full[start_idx_full:]
        real_data_zoom = real_data_input[-zoom_window:]

        if len(zoom_input_dates) == len(real_data_zoom):
            axes[0].plot(zoom_input_dates, real_data_zoom, label=f"History: {categ}",
                         linestyle=reallinestyle, color=color, alpha=0.7)
            if real_data_zoom.size > 0:
                min_vals_zoom.append(np.min(real_data_zoom))
                max_vals_zoom.append(np.max(real_data_zoom))

        # Predicted data for zoom plot
        if predictions_denorm.size > 0 and predictions_denorm.shape[1] > (s-1):
            pred_data_categ = predictions_denorm[:, s-1]  # shape (n_pred,)
            if len(pred_dates) == len(pred_data_categ):
                axes[0].plot(pred_dates, pred_data_categ, label=f"Predicted: {categ}",
                             linestyle=predlinestyle, color=color, alpha=0.9)
                min_vals_zoom.append(np.min(pred_data_categ))
                max_vals_zoom.append(np.max(pred_data_categ))

    axes[0].legend(fontsize='small')
    axes[0].grid(True, linestyle=':')
    axes[0].set_ylabel("Price")
    if min_vals_zoom and max_vals_zoom:
        axes[0].set_ylim(np.min(min_vals_zoom)*0.99,
                         np.max(max_vals_zoom)*1.01)
    axes[0].xaxis.set_major_formatter(
        mdates.DateFormatter('%H:%M'))  # Show time for hourly focus
    axes[0].xaxis.set_major_locator(
        mdates.AutoDateLocator(minticks=3, maxticks=7))

    # --- Plot 2: Full Context ---
    axes[1].set_title(f"{symbol} - Full History and Prediction")
    min_vals_full, max_vals_full = [], []

    # Dates for the *entire* bars_full sequence
    all_dates = np.array([formattime(d) for d in bars_full[0]])
    # Filter out potential None values from formattime
    valid_dates_mask = all_dates != None
    all_dates = all_dates[valid_dates_mask]

    for s, categ in enumerate(categories):
        if categ == "date":
            continue
        color = colors[(s - 1) % len(colors)]

        # Full historical data
        real_data_full = bars_full[s][valid_dates_mask]  # Use mask
        if len(all_dates) == len(real_data_full):
            axes[1].plot(all_dates, real_data_full, label=f"History: {categ}",
                         linestyle=reallinestyle, color=color, alpha=0.6)
            if real_data_full.size > 0:
                min_vals_full.append(np.min(real_data_full))
                max_vals_full.append(np.max(real_data_full))

        # Predicted data (same as above)
        if predictions_denorm.size > 0 and predictions_denorm.shape[1] > (s-1):
            pred_data_categ = predictions_denorm[:, s-1]
            if len(pred_dates) == len(pred_data_categ):
                axes[1].plot(pred_dates, pred_data_categ, label=f"Predicted: {categ}",
                             linestyle=predlinestyle, color=color, alpha=0.9, linewidth=1.5)
                min_vals_full.append(np.min(pred_data_categ))
                max_vals_full.append(np.max(pred_data_categ))

    # Add rectangle for context if possible
    if (zoom_input_dates.size > 0  # Check if numpy array is not empty
        and len(pred_dates) > 0      # Check if list is not empty
        # Check if list is not empty (standard python bool check)
        and min_vals_zoom
            # Check if list is not empty (standard python bool check)
            and max_vals_zoom):

        rect_start_date = zoom_input_dates[0]
        rect_end_date = pred_dates[-1]
        rect_ymin = np.min(min_vals_zoom) * 0.99
        rect_ymax = np.max(max_vals_zoom) * 1.01
        try:  # date2num can fail
            axes[1].add_patch(Rectangle((mdates.date2num(rect_start_date), rect_ymin),
                                        mdates.date2num(
                                            rect_end_date) - mdates.date2num(rect_start_date),
                                        rect_ymax - rect_ymin,
                                        edgecolor='grey', facecolor='none', lw=1, ls='--'))
        except Exception as e:
            print(f"Could not draw context rectangle: {e}")

    axes[1].legend(fontsize='small')
    axes[1].grid(True, linestyle=':')
    axes[1].set_ylabel("Price")
    if min_vals_full and max_vals_full:
        axes[1].set_ylim(np.min(min_vals_full)*0.98,
                         np.max(max_vals_full)*1.02)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(formatter)

    fig.autofmt_xdate()
    plt.tight_layout()
    try:
        plt.savefig(path)
        print(f"Prediction plot saved to {path}")
    except Exception as e:
        print(f"Error saving prediction plot to {path}: {e}")
    plt.show()  # Optional: Show plot interactively
    plt.close()
