# This script contains configuration settings and parameters for the finance bot,
# including file paths, data parameters, model architecture, training settings,
# and utility functions.

import os # For path manipulation and environment variables
import torch # PyTorch library (used for device and optimizer)
import torch.nn as nn # Neural network modules (used for loss function)
# For handling time and date calculations
from datetime import timedelta, datetime
# Alpaca library components for defining timeframes
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import math
# --- Core Parameters ---
# Define base file paths relative to the script's location
FILEPATH = os.path.dirname(os.path.realpath(__file__)) # Directory of the current script
FINANCEDIR=os.path.join(FILEPATH, "..") # Parent directory of the script (financebot root)

# Define specific directories based on the root financebot directory
DATADIR=os.path.join(FINANCEDIR, "data") # Directory for storing data files (training, live cache, model)
GRAPHSDIR=os.path.join(FINANCEDIR, "graphs") # Directory for saving plots
SRCDIR=os.path.join(FINANCEDIR, "src") # Source code directory

# Define the device to use for computations ("cuda" for GPU if available, "cpu" otherwise)
DEVICE = "cpu" # Explicitly set to CPU here, but often dynamically determined

# --- Data Parameters ---
# List of stock symbols to be used for training the model
SYMBOLS = ["AMZN", "AAPL", "META", "NVDA",
           "MSFT", "TSLA"]
# The specific stock symbol to use for live prediction/inference
PREDICTION_SYMBOL = "VOO"
# Definition of the data structure for each bar: feature name and numpy dtype
DATA_TYPE = [('date', 'i8'), ('open', 'f8'), ('close', 'f8'),
             ('high', 'f8'), ('low', 'f8')]
# List of category names derived from DATA_TYPE
CATEGORIES = [categ for (categ, categtype) in DATA_TYPE]
# Number of features per time step, excluding the 'date'
INPUT_SIZE = len(DATA_TYPE) - 1
# The length of the input sequence (number of historical time steps) provided to the model encoder
SEQ_LEN = 100

# The number of future steps to predict during live inference (using the trained model)
N_PRED_INFERENCE = 10

# Client IDs potentially used for distinguishing API connections (e.g., in logs)
BASE_CLIENT_ID_TRAIN = 100
CLIENT_ID_LIVE = 99
# The timeframe unit for historical data bars (e.g., 1 hour bars)
FREQ = TimeFrame(1, TimeFrameUnit.Hour)
# Define the time window for fetching historical data
NOW = datetime.now().astimezone() # Get the current time with timezone
N_DAYS=365 # Number of days of historical data to fetch
END = (NOW - timedelta(hours=1)) # End time for data fetching (1 hour ago)
START = END - timedelta(days=N_DAYS) # Start time for data fetching (N_DAYS before end)

# --- Training Parameters ---
EPOCHS = 50 # Total number of training epochs
BATCH_SIZE = 32 # Number of sequences per training batch
LEARNING_RATE = 0.001 # Learning rate for the optimizer
DATA_SPLIT_RATIO = 0.9 # NOTE: This variable appears unused; VAL_SPLIT_RATIO is used for splitting
VAL_SPLIT_RATIO = 0.1  # Ratio of generated sequences to use for validation (e.g., 10% validation, 90% training)

# The maximum number of future steps the model will be trained to predict in one go (decoder length)
N_PRED_TRAINING_MAX=int(1.5*SEQ_LEN)
# Schedule for curriculum learning during training: defines the prediction horizon
# at different training progress points (e.g., epoch ranges)
#CURRICULUM_SCHEDULE = {
#     0: 15,   # From start (epoch 0), predict 15 steps
#     15: 30,  # After 15 epochs, predict 30 steps
#    30: 45,  # After 30 epochs, predict 45 steps
#    EPOCHS: N_PRED_TRAINING_MAX # After EPOCHS epochs, predict up to max_horizon N_PRED_TRAINING_MAX
#}
CURRICULUM_SCHEDULE={}
NC=8
Cstep=EPOCHS/NC
for c in range(NC):
    CURRICULUM_SCHEDULE[int(c*Cstep)]=int(math.sin(0.5*math.pi*(c+1)/NC)*N_PRED_TRAINING_MAX)


# --- Model Architecture Parameters ---
DIM_MODEL = 64 # Dimension of the internal model representation (embeddings, feedforward)
NUM_HEADS = 4 # Number of attention heads in the Transformer
NUM_ENCODER_LAYERS = 3 # Number of layers in the Transformer encoder
NUM_DECODER_LAYERS = 3 # Number of layers in the Transformer decoder
DROPOUT_P = 0.1 # Dropout probability used in the model
# Maximum length for positional encoding; should be at least the max sequence lengths used
MAX_LEN_POSITIONAL_ENCODING =  max(5000, SEQ_LEN, N_PRED_TRAINING_MAX)

# --- File Paths for Data and Model ---
# Path to save/load the pre-generated training data
TRAINING_DATA_PATH = os.path.join(DATADIR, "trainingdata_multistep.npz")
# Template path for caching live data fetched for prediction (symbol will be formatted in)
LIVE_DATA_CACHE_PATH_TEMPLATE = os.path.join(DATADIR, "livedata_{symbol}.npy")
# Path to save/load the trained model state dictionary
MODEL_SAVE_PATH = os.path.join(DATADIR, "modelsave_multistep.pth")
# Path for a log file (purpose not shown in provided snippets)
LOG_FILE_PATH = os.path.join(DATADIR, "logfile.txt")
# Path to save the training/validation loss plot
LOSS_PLOT_PATH = os.path.join(GRAPHSDIR, "trainval_loss.pdf")
# Path to save the prediction plot
PREDICTION_PLOT_PATH = os.path.join(GRAPHSDIR, "predtest.pdf")

# --- Workflow Control Flags ---
# If True, attempts to load a saved model instead of training
LOAD_SAVED_MODEL = False
# If True, attempts to load training data from file instead of fetching and processing
# This flag is also used to control live data caching.
LOAD_TRAINING_DATA = False

# --- Training Components ---
# The loss function used for training (Mean Squared Error Loss)
LOSS_FN = nn.MSELoss()


# --- Utility Function ---
def get_optimizer(model_parameters):
    """Helper function to create and return the optimizer instance."""
    # Uses the Adam optimizer with the specified learning rate
    return torch.optim.Adam(model_parameters, lr=LEARNING_RATE)


# Print confirmation that configuration has been loaded
print(f"Configuration loaded. Device: {DEVICE}")
print(f"Training Symbols: {SYMBOLS}")
print(f"Prediction Symbol: {PREDICTION_SYMBOL}")