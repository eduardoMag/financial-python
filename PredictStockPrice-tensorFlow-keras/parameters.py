import os
import time
from tensorflow.keras.layers import LSTM

# windows size or the sequence length
N_STEPS = 50
# lookup step, 1 is the next day
LOOKP_STEP = 15

# whether to scale feature columns & ouput price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"

# whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"

# wether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sdb-{int(SPLIT_BY_DATE)}"

# test ratio size
TEST_SIZE = 0.2

# features to use
FEATURES_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

# date now
date_now = time.strftime("%Y-%m-%d")

# model parameters
N_LAYERS = 2
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

# training parameters
# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "admin"
BATCH_SIZE = 64
EPOCHS = 500

# amazon stock market
ticker = "AMZN"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-" \ 
             f"{N_STEPS}-step-{LOOKP_STEP}-layers-{N_LAYERS}-units-{UNITS} "
if BIDIRECTIONAL:
    model_name += "-b"
