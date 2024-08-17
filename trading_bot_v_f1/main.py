#https://www.geeksforgeeks.org/get-financial-data-from-yahoo-finance-with-python/
from parameters import *
from functions import *

import yfinance as yahooFinance
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import coint
from sklearn.metrics import accuracy_score
import xgboost as xgb

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
rolling_corr = merged_df[tickers].rolling(window=window_size_corr_short, min_periods=1).corr().unstack()
for ticker1 in tickers:
    for ticker2 in tickers:
        merged_df[f'{ticker1}_{ticker2}_Corr'] = rolling_corr[ticker1][ticker2].reset_index(level=0, drop=True)


merged_df.dropna(inplace=True)

window_size_corr_long = len(merged_df)
rolling_corr = merged_df[tickers].rolling(window=window_size_corr_long, min_periods=1).corr().unstack()
for ticker1 in tickers:
    for ticker2 in tickers:
        merged_df[f'{ticker1}_{ticker2}_Corr_Long'] = rolling_corr[ticker1][ticker2].reset_index(level=0, drop=True)
merged_df.fillna(method='ffill', inplace=True)
merged_df.dropna(inplace=True)
print(merged_df)

for ticker1 in tickers:
    for ticker2 in tickers:
        merged_df[f'{ticker1}_{ticker2}_Coint'] = rolling_coint(merged_df[ticker1], merged_df[ticker2], window_size_coint)

merged_df.dropna(inplace=True)
merged_df['Y'] = np.where(merged_df[SELECTED_Y].shift(-1) > merged_df[SELECTED_Y], 1, 0)
merged_df = merged_df[:-1]
X_train = merged_df.loc[merged_df["Date"]<TRAIN_TEST_DATE_DIVISION].loc[:, (merged_df.columns != 'Y') & (merged_df.columns != 'Date')].reset_index(drop=True)
X_test = merged_df.loc[merged_df["Date"]>=TRAIN_TEST_DATE_DIVISION].loc[:, (merged_df.columns != 'Y') & (merged_df.columns != 'Date')].reset_index(drop=True)
y_train = merged_df.loc[merged_df["Date"]<TRAIN_TEST_DATE_DIVISION]['Y'].reset_index(drop=True)
y_test = merged_df.loc[merged_df["Date"]>=TRAIN_TEST_DATE_DIVISION]['Y'].reset_index(drop=True)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
bst = xgb.train(params, dtrain, num_boost_round=100)
y_pred = bst.predict(dtest)
y_pred_class = np.round(y_pred).astype(int)  # convert probabilities to class labels
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Accuracy: {accuracy:.2f}")
print(y_pred_class)
# print(y_train)
# print(y_test)
