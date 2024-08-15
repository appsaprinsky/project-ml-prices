#https://www.geeksforgeeks.org/get-financial-data-from-yahoo-finance-with-python/
# https://lightgbm.readthedocs.io/en/stable/Python-Intro.html
import yfinance as yahooFinance
import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint

import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score

startDate = datetime.datetime(2018, 1, 1)
endDate = datetime.datetime(2022, 1, 1)
tickers = ["META", "AAPL", "GOOGL", "AMZN", "MSFT", "TSLA"]
# tickers = ["META", "AAPL"]
data = {}

for ticker in tickers:
    company_info = yahooFinance.Ticker(ticker)
    prices = company_info.history(start=startDate, end=endDate, interval="1wk")
    prices.reset_index(inplace=True)
    data[ticker] = prices[["Date", "Open"]].rename(columns={"Open": f'{ticker}'})

merged_df = None

for ticker in tickers:
    if merged_df is None:
        merged_df = data[ticker]
    else:
        merged_df = pd.merge(merged_df, data[ticker], on="Date", how="inner")

merged_df.dropna(inplace=True)
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
window_size = 10
rolling_corr = merged_df[tickers].rolling(window=window_size, min_periods=1).corr().unstack()#.iloc[:, 3]
for ticker1 in tickers:
    for ticker2 in tickers:
        merged_df[f'{ticker1}_{ticker2}_Corr'] = rolling_corr[ticker1][ticker2].reset_index(level=0, drop=True)


merged_df.dropna(inplace=True)
window_size = len(merged_df)
rolling_corr = merged_df[tickers].rolling(window=window_size, min_periods=1).corr().unstack()#.iloc[:, 3]
for ticker1 in tickers:
    for ticker2 in tickers:
        merged_df[f'{ticker1}_{ticker2}_Corr_Long'] = rolling_corr[ticker1][ticker2].reset_index(level=0, drop=True)

merged_df.dropna(inplace=True)

def rolling_coint(series1, series2, window):
    coint_values = np.full(len(series1), np.nan)  
    for i in range(window, len(series1)):
        s1_window = series1[i-window:i]
        s2_window = series2[i-window:i]
        coint_t, p_value, _ = coint(s1_window, s2_window)
        coint_values[i] = p_value  
    return coint_values
window_size = 5

for ticker1 in tickers:
    for ticker2 in tickers:
        merged_df[f'{ticker1}_{ticker2}_Coint'] = rolling_coint(merged_df[ticker1], merged_df[ticker2], window_size)

merged_df.dropna(inplace=True)
print(merged_df)
merged_df['Y'] = np.where(merged_df['META'].shift(-1) > merged_df['META'], 1, 0)
merged_df = merged_df[:-1]
X_train = merged_df.loc[merged_df["Date"]<"2021-01-01"].loc[:, (merged_df.columns != 'Y') & (merged_df.columns != 'Date')].reset_index(drop=True)
X_test = merged_df.loc[merged_df["Date"]>="2021-01-01"].loc[:, (merged_df.columns != 'Y') & (merged_df.columns != 'Date')].reset_index(drop=True)
y_train = merged_df.loc[merged_df["Date"]<"2021-01-01"]['Y'].reset_index(drop=True)
y_test = merged_df.loc[merged_df["Date"]>="2021-01-01"]['Y'].reset_index(drop=True)



dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
# dtrain = lgb.Dataset(X_train, label=y_train)
# dtest = lgb.Dataset(X_test, label=y_test)
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}
# params = {
#     'max_depth': 3,
#     'eta': 0.1,
#     'objective': 'binary',
#     'eval_metric': 'logloss'
# }
bst = xgb.train(params, dtrain, num_boost_round=100)
y_pred = bst.predict(dtest)
# bst = lgb.train(params, dtrain, num_boost_round=100)
# y_pred = bst.predict(X_test)
y_pred_class = np.round(y_pred).astype(int)  # Convert probabilities to class labels
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Accuracy: {accuracy:.2f}")
print(y_pred_class)

# print(y_train)
# print(y_test)


'''
'''