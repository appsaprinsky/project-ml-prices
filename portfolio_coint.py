from statsmodels.tsa.stattools import coint
import itertools
from tickers.EU import *
# from tickers.USA import *
from functions.objective_functions import *

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

def find_cointegrated_pairs(data, critical_value=0.05):
    n = data.shape[1]
    tickers = data.columns
    pairs = []
    pvalues = np.ones((n, n))
    for i, j in itertools.combinations(range(n), 2):
        score, p_value, _ = coint(data.iloc[:, i], data.iloc[:, j])
        pvalues[i, j] = p_value
        if p_value < critical_value:
            pairs.append((tickers[i], tickers[j]))
    return pairs, pvalues

DATE_START = '2020-01-01'
DATE_END = '2024-08-23'

data = yf.download(tickers, start=DATE_START, end=DATE_END)['Adj Close'] 
data = data.dropna()
cointegrated_pairs, pvalues = find_cointegrated_pairs(data)
print(f"Cointegrated Pairs: {cointegrated_pairs}")
cointegrated_assets = set([ticker for pair in cointegrated_pairs for ticker in pair])
filtered_data = data[list(cointegrated_assets)]
filtered_returns = filtered_data.pct_change().dropna()
weights = np.array([1/len(filtered_data.columns)] * len(filtered_data.columns))
portfolio_return = np.sum(filtered_returns.mean() * weights) * 252
portfolio_variance = np.dot(weights.T, np.dot(filtered_returns.cov() * 252, weights))
portfolio_volatility = np.sqrt(portfolio_variance)
print(f"Cointegrated Portfolio Return: {portfolio_return:.2%}")
print(f"Cointegrated Portfolio Volatility: {portfolio_volatility:.2%}")
portfolio_returns = filtered_returns.dot(weights)
cumulative_returns = (1 + portfolio_returns).cumprod()
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label='Cointegrated Portfolio')
plt.title('Cointegrated Portfolio Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
cointegration_portfolio = pd.DataFrame({"Tickets": list(filtered_data.columns), "Weights": weights})
cointegration_portfolio.to_csv("portfolio/output/cointegrated_portfolio.csv")


