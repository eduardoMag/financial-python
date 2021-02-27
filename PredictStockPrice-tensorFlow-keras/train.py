from stock-prediction import create_model, load_data
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import pandas as pd
from parameters import *


# create these folders if they do not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")


# load the data
data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, shuffle=SHUFFLE, lookup_step=LOOKP_STEP, test_size=TEST_SIZE, feature_columns=FEATURES_COLUMNS)

#save the dataframe
data["df"].to_csv(ticker_data_filename)

#construct the model
model = crate_model(N_STEPS, len(FEATURES_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

#tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weight_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

#train the model
history = model.fit(data["x_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["x_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard], verbose=1)