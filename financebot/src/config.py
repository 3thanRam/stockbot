import os
import torch
import torch.nn as nn
from datetime import timedelta, datetime
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# --- Core Parameters ---
FILEPATH = os.path.dirname(os.path.realpath(__file__))
FINANCEDIR=os.path.join(FILEPATH, "..")

DATADIR=os.path.join(FINANCEDIR, "data")
GRAPHSDIR=os.path.join(FINANCEDIR, "graphs")
SRCDIR=os.path.join(FINANCEDIR, "src")

DEVICE = "cpu"  
# --- Data Parameters ---
SYMBOLS = ["AMZN", "AAPL", "META", "NVDA",
           "MSFT", "TSLA"]  # Symbols for training
PREDICTION_SYMBOL = "VOO"  # Symbol for prediction
DATA_TYPE = [('date', 'i8'), ('open', 'f8'), ('close', 'f8'),
             ('high', 'f8'), ('low', 'f8')]
CATEGORIES = [categ for (categ, categtype) in DATA_TYPE]
INPUT_SIZE = len(DATA_TYPE) - 1  # Features per time step (excluding date)
SEQ_LEN = 50           # Input sequence length for encoder
# How many steps to predict during training (decoder length)
N_PRED_TRAINING_MAX=60
CURRICULUM_SCHEDULE = {
     0: 15,   
     15: 30,  
    30: 45,  
    50: N_PRED_TRAINING_MAX 
}
N_PRED_INFERENCE = 10  # How many steps to predict during inference

BASE_CLIENT_ID_TRAIN = 100
CLIENT_ID_LIVE = 99
FREQ = TimeFrame(1, TimeFrameUnit.Hour)
NOW = datetime.now().astimezone()
N_DAYS=20
END = (NOW - timedelta(hours=1))
START = END - timedelta(days=N_DAYS)
# --- Training Parameters ---
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DATA_SPLIT_RATIO = 0.9
VAL_SPLIT_RATIO = 0.1  # Train/validation split ratio for generated sequences

DIM_MODEL = 64
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT_P = 0.1
MAX_LEN_POSITIONAL_ENCODING =  max(5000, SEQ_LEN, N_PRED_TRAINING_MAX)

TRAINING_DATA_PATH = os.path.join(DATADIR, "trainingdata_multistep.npz")
LIVE_DATA_CACHE_PATH_TEMPLATE = os.path.join(DATADIR, "livedata_{symbol}.npy")
MODEL_SAVE_PATH = os.path.join(DATADIR, "modelsave_multistep.pth")
LOG_FILE_PATH = os.path.join(DATADIR, "logfile.txt")
LOSS_PLOT_PATH = os.path.join(GRAPHSDIR, "trainval_loss.pdf")
PREDICTION_PLOT_PATH = os.path.join(GRAPHSDIR, "predtest.pdf")

LOAD_SAVED_MODEL = False
LOAD_TRAINING_DATA = False

LOSS_FN = nn.MSELoss()


def get_optimizer(model_parameters):
    """Helper to create optimizer"""
    return torch.optim.Adam(model_parameters, lr=LEARNING_RATE)


print(f"Configuration loaded. Device: {DEVICE}")
print(f"Training Symbols: {SYMBOLS}")
print(f"Prediction Symbol: {PREDICTION_SYMBOL}")

