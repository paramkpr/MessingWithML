import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import math
import quandl
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

# noinspection SpellCheckingInspection
quandl.ApiConfig.api_key = 'mkSJc3cDJbVVxKD9vbei'
df = quandl.get("WIKI/GOOGL")
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

high = 'Adj. High'
low = 'Adj. Low'
start = 'Adj. Open'
close = 'Adj. Close'
volume = 'Adj. Volume'

df["HL_PCT"] = (df[high] - df[low]) / df[close] * 100  # Volatility
df["change_PCT"] = (df[close] - df[start]) / df[close] * 100  # Daily Percent Change

df = df[[close, 'HL_PCT', 'change_PCT', volume]]

forecast_col = close
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)  # Creates new col for label by shifting forecast col by 10% of df

X = preprocessing.scale(np.array(df.drop(['label'], 1)))  # Features get converted to numpy array
X_recent = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)  # Removes nan(s) created due to 35 extra row positions

y = np.array(df['label'])  # Label gets converted to numpy array

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_recent)
# print(forecast_set, confidence, forecast_out)

style.use('ggplot')

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
next_unix = last_unix + 86400

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()