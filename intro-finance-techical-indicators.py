import yfinance as yf
import pandas_datareader as pdr
from mplfinance import candlestick_ohlc
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from stockstats import StockDataFrame


# import AAPL stock price
df_aapl = pdr.get_data_yahoo("AAPL", start="2019-01-01", end="2019-09-30")

# import SPY stock price
df_spy = pdr.get_data_yahoo("SPY", start="2019-01-01", end="2019-09-30")

print(df_spy.head)
print("*" * 30)
print(df_aapl)

# plotting stock prices
df_aapl[["Open", "High", "Low", "Close"]].plot()
plt.show()

# change the view to candlestick
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot()

plot_data = []
for i in range(150, len(df_aapl)):
    row = [
        i,
        df_aapl.Open.iloc[i],
        df_aapl.High.iloc[i],
        df_aapl.Low.iloc[i],
        df_aapl.Close.iloc[i],
    ]
    plot_data.append(row)
candlestick_ohlc(ax, plot_data)
plt.show()

# financial tecnical indicators
stocks = StockDataFrame.retype(df_aapl[["Open", "Close", "High", "Low", "Volume"]])

# simple moving average (SMA)
plt.plot(stocks["close_10_sma"], color="b", label="SMA")
plt.plot(df_aapl.Close, color="g", label="Close prices")
plt.legend(loc="lower right")
plt.show()

#exponential moving average (EMA)
plt.plot(stocks["close_10_sma"], color="b", label="SMA")  # plotting SMA
plt.plot(stocks["close_10_ema"], color="k", label="EMA")
plt.plot(df_aapl.Close, color="g", label="Close prices")  # plotting close prices
plt.legend(loc="lower right")
plt.show()

# moving average convergence/divergence (MACD Line)
plt.plot(stocks["macd"], color="b", label="MACD")
plt.plot(stocks["macds"], color="g", label="Signal Line")
plt.legend(loc="lower right")
plt.show()
