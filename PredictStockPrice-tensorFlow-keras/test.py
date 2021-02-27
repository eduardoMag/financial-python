import numpy as np
import matplotlib.pyplot as plt
from stock_prediction import create_model, load_data
from parameters import *


def plot_graph(test_df):
    plt.plot(test_df[f'true_adjclose_{LOOKP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def get_final_df(model, data):
    # if predicted future price is higher than the current,
    # then calculate the true future price minus the current price, to get the buy profit
    buy_profit = lambda current, true_future, pred_future: true_future - current if pred_future > current else 0

    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price
    sell_profit = lambda current, true_future, pred_future: current - true_future if pred_future < current else 0
    x_test = data["x_test"]
    y_test = data["y_test"]

    #perform prediction and get proces
    y_pred = model.predict(x_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"]

    # add predicted future proces to the dataframe
    test_df[f"adjclose_{LOOKP_STEP}"] = y_pred

    #add true future prices to the dataframe
    test_df[f"true_adjclose_{LOOKP_STEP}"] = y_test

    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df

    #add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit,
                                      final_df["adjclose"],
                                      final_df[f"adjclose_{LOOKP_STEP}"],
                                      final_df[f"true_adjclose_{LOOKP_STEP}"])
                                  )

    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit,
                                       final_df["adjclose"],
                                       final_df[f"adjclose_{LOOKP_STEP}"],
                                       final_df[f"true_adjclose_{LOOKP_STEP}"])
                                   )
    return  final_df


def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_squence"][N_STEPS:]

    #expand dimension
    last_sequence =np.expand_dims(last_sequence, axis=0)

    #get the prediction
    prediction = model.predict(last_sequence)

    # get the price
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


# load the data
data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, shuffle=SHUFFLE, lookup_step=LOOKP_STEP, test_size=TEST_SIZE, feature_columns=FEATURES_COLUMNS)

#construct the model
model = create_model(N_STEPS, len(FEATURES_COLUMNS), loss=LOSS, units=UNITS, cell=CELLS, n_layers=N_LAYERS, dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

# load optimal model weights from results folder
model_path =os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

#evaluate the model
loss, mae = model.evaluate(data["x_test"], data["y_test"], verbose=0)

# calculate the mean absolute error (inverse scaling)
if SCALE:
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae

# get final dataframe for the testting set
final_df = get_final_df(model, data)

#predict the future price
future_price = predict(model, data)

# calculate accurracy by counting no. of positive profits
accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buyy_profit'] > 0])) / len(final_df)

#calculating total buy and sell profit
total_buy_profit = final_df["buy_profit"].sum()
total_sell_profit = final_df["sell_profit"].sum()
# total profit from both (sell & buy)
total_profit = total_sell_profit + total_sell_profit

#divide total profit by number of testing samples
profit_per_trade = total_profit / len(final_df)

#printing metrics
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
print(f"{LOSS} loss:", loss)
print("Mean Absolute Error:", mean_absolute_error)
print("Accuracy score:", accuracy_score)
print("Total buy profit:", total_buy_profit)
print("Total sell profit:", total_sell_profit)
print("Total profit:", total_profit)
print("Profit per trade:", profit_per_trade)

# plot true/pred prices graph
plot_graph(final_df)
print(final_df.tail(10))

#save the final dataframe to csv-results folder
csv_results_folder = "csv-results"
if not os.path.isdir(csv_results_folder):
    os.mkdir(csv_results_folder)
csv_filename = os.path.join(csv_results_folder, model_name + ".csv")
final_df.to_csv(csv_filename)