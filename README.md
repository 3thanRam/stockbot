# AlpacaBot: AI-Powered Stock Forecasting & Visualization

## Project Overview

This project combines a Transformer-based financial forecasting engine (`financebot`) with an interactive visualization and potential trading interface (`alpacabot`). It fetches historical and live stock market data via the Alpaca API, trains a Transformer model to predict future price movements, and visualizes these predictions alongside the actual data in real-time or simulation. The `alpacabot` component provides an interactive matplotlib plot with controls for pausing, shifting views, and adjusting prediction length, and includes basic integration with an Alpaca trading account.

**Key Features:**

* **Transformer Model:** Utilizes a modern Transformer architecture (`financebot/src/model.py`) for sequential data forecasting.
* **Data Handling:** Fetches historical and live data from Alpaca (`financebot/src/data_handler.py`, `livedata.py`).
* **Training & Inference:** Supports training the model from scratch or loading a pre-trained model (`financebot/src/trainer.py`, `financebot/src/predictor.py`).
* **Curriculum Learning:** Implements a curriculum learning strategy during training to gradually increase the prediction horizon.
* **Interactive Visualization:** Real-time or simulated plotting of stock data and predictions using Matplotlib animation (`alpacabot/plot.py`, `livedata.py`, `simulatedata.py`, `plotdata.py`).
* **Plot Controls:** Buttons for pausing the stream, shifting the visible data window, and adjusting the prediction length (`alpacabot/plot.py`).
* **Alpaca Account Integration:** Connects to an Alpaca trading account (supports paper trading) to display account details and potentially execute orders (`account.py`).
* **Configurable:** Parameters for data fetching, model architecture, training, and file paths are centrally managed (`financebot/src/config.py`).


### Prerequisites

* Python
* An Alpaca account (paper trading recommended for testing)
* Your Alpaca API Key ID and Secret Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Environment Variables:**
    Set your Alpaca API keys as environment variables. Replace `YOUR_API_KEY_ID` and `YOUR_API_SECRET_KEY` with your actual keys.

    **Linux/macOS:**
    ```bash
    export APCA_API_KEY_ID="YOUR_API_KEY_ID"
    export APCA_API_SECRET_KEY="YOUR_API_SECRET_KEY"
    ```
    **Windows (Command Prompt):**
    ```bash
    set APCA_API_KEY_ID="YOUR_API_KEY_ID"
    set APCA_API_SECRET_KEY="YOUR_API_SECRET_KEY"
    ```
    **Windows (PowerShell):**
    ```powershell
    $env:APCA_API_KEY_ID="YOUR_API_KEY_ID"
    $env:APCA_API_SECRET_KEY="YOUR_API_SECRET_KEY"
    ```


## Configuration

All main parameters are defined in `financebot/src/config.py`. You can modify this file to:

* Change training symbols (`SYMBOLS`).
* Set the prediction symbol (`PREDICTION_SYMBOL`).
* Adjust data fetching ranges (`START`, `END`, `FREQ`).
* Modify model architecture parameters (`DIM_MODEL`, `NUM_HEADS`, etc.).
* Tune training hyperparameters (`EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`).
* Configure sequence lengths and prediction horizons (`SEQ_LEN`, `N_PRED_TRAINING_MAX`, `N_PRED_INFERENCE`).
* Enable/disable loading saved models or training data (`LOAD_SAVED_MODEL`, `LOAD_TRAINING_DATA`).
* Specify file paths for data, models, and graphs.
* Adjust the curriculum learning schedule (`CURRICULUM_SCHEDULE`).

## How to Run

The main entry point is `alpacabot/main.py`.

1.  **Ensure your virtual environment is active** (if you created one) and environment variables are set.
2.  **Run the main script:**
    ```bash
    python alpacabot/main.py
    ```
    By default, this will attempt to load a model (if `LOAD_SAVED_MODEL` is True in `config.py`) or train one (if False), then fetch live data for the `PREDICTION_SYMBOL` and start the interactive visualization.

The plot window will appear, displaying the historical data, predicted future values, and interactive buttons.

*Note: If you run for the first time with `LOAD_SAVED_MODEL = False`, the script will first fetch training data (which can take some time depending on the date range and number of symbols), train the model, save it, and then proceed to the visualization.*

## Project Structure
.
├── alpacabot/
│   ├── main.py           # Main entry point, orchestrates setup and animation
│   ├── plot.py           # Matplotlib figure setup, plot artists, and interactive buttons
│   ├── livedata.py       # Handles fetching and integrating live Alpaca stream data
│   ├── simulatedata.py   # Handles fetching historical data and simulating a stream
│   └── account.py        # WIP: Interfaces with Alpaca trading account (details, orders)
├── financebot/
│   └── src/
│       ├── config.py       # Application configuration parameters
│       ├── data_handler.py # Fetches, formats, normalizes, and sequences data
│       ├── model.py        # Defines the Transformer model architecture and PE
│       ├── predictor.py    # Handles model inference (autoregressive prediction)
│       ├── trainer.py      # Implements the training and validation loops
│       └── plotting_utils.py# Utility functions for plotting loss and predictions
├── data/                 # Directory for data caches and saved models (created based on config)
└── graphs/               # Directory for saving plots (created based on config)



## Important Considerations

* **Financial Risk:** This is a DEMONSTRATION project. Using any part of this code for actual trading carries significant financial RISK. The model's predictions are not guaranteed to be accurate.
* **Alpaca API Limits:** Be mindful of Alpaca's API rate limits when fetching data or using the stream.
* **Data Quality:** The performance of the model is highly dependent on the quality and relevance of the training data.
* **Model Limitations:** Transformer models have limitations, especially when predicting highly volatile or unprecedented market movements.