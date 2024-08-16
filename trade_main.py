import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yahooFinance
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import coint
from sklearn.metrics import accuracy_score
import xgboost as xgb


def rolling_coint(series1, series2, window):
    coint_values = np.full(len(series1), np.nan)  
    for i in range(window, len(series1)):
        s1_window = series1[i-window:i]
        s2_window = series2[i-window:i]
        coint_t, p_value, _ = coint(s1_window, s2_window)
        coint_values[i] = p_value  
    return coint_values


endDate = datetime.datetime.today()
startDate = endDate - relativedelta(years=3)
tickers = ["META", "AAPL", "GOOGL", "AMZN", "MSFT", "TSLA"]
window_size_corr_short = 10
window_size_coint = 5

params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}



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

for ticker1 in tickers:
    for ticker2 in tickers:
        merged_df[f'{ticker1}_{ticker2}_Coint'] = rolling_coint(merged_df[ticker1], merged_df[ticker2], window_size_coint)

merged_df.dropna(inplace=True)
print(merged_df)
merged_df['Y'] = np.where(merged_df['META'].shift(-1) > merged_df['META'], 1, 0)
# merged_df = merged_df[:-1]
# PREDICTION_DATE = endDate.strftime("%Y-%m-%d")
PREDICTION_DATE = list(merged_df['Date'])[-1]
print(PREDICTION_DATE)
print(merged_df['Y'] )

X_train = merged_df.loc[merged_df["Date"]<PREDICTION_DATE].loc[:, (merged_df.columns != 'Y') & (merged_df.columns != 'Date')].reset_index(drop=True)
X_predict_money = merged_df.loc[merged_df["Date"]>=PREDICTION_DATE].loc[:, (merged_df.columns != 'Y') & (merged_df.columns != 'Date')].reset_index(drop=True)
y_train = merged_df.loc[merged_df["Date"]<PREDICTION_DATE]['Y'].reset_index(drop=True)

dtrain = xgb.DMatrix(X_train, label=y_train)
dpredict_money = xgb.DMatrix(X_predict_money)
bst = xgb.train(params, dtrain, num_boost_round=100)
y_pred = bst.predict(dtrain)
y_pred_class = np.round(y_pred).astype(int)  # convert probabilities to class labels
accuracy = accuracy_score(y_train, y_pred_class)
print(f"Accuracy: {accuracy:.2f}")
print(y_pred_class)
# print(y_train)
# print(y_test)

y_pred = bst.predict(dpredict_money)
y_pred_class = np.round(y_pred).astype(int) 
print(y_pred_class)

