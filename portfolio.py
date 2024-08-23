from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

DATE_START = '2020-01-01'
DATE_END = '2024-08-23'
tickers = [
    'AAPL',  # Apple Inc.
    'MSFT',  # Microsoft Corp.
    'GOOGL', # Alphabet Inc. (Google)
    'AMZN',  # Amazon.com Inc.
    'TSLA',  # Tesla Inc.
    'META',  # Meta Platforms Inc. (Facebook)
    'NVDA',  # NVIDIA Corporation
    'JNJ',   # Johnson & Johnson
    'V',     # Visa Inc.
    'PG',    # Procter & Gamble Co.
    'JPM',   # JPMorgan Chase & Co.
    'UNH',   # UnitedHealth Group Inc.
    'HD',    # Home Depot Inc.
    'MA',    # Mastercard Inc.
    'DIS',   # The Walt Disney Company
    'NFLX',  # Netflix Inc.
    'BABA',  # Alibaba Group Holding Ltd.
    'XOM',   # Exxon Mobil Corporation
    'BAC',   # Bank of America Corp.
    'KO',    # The Coca-Cola Company
    'PEP'    # PepsiCo Inc.
]
data = yf.download(tickers, start=DATE_START, end=DATE_END)['Adj Close']
print(data.head())
returns = data.pct_change().dropna() # calculate daily returns
print(returns.head())

weights = np.array([1/len(tickers)] * len(tickers)) # define portfolio weights (equal weighting)
portfolio_return = np.sum(returns.mean() * weights) * 252  # calculate the expected portfolio return # annualize return (252 trading days)


portfolio_variance = np.dot(weights.T, np.dot(returns.cov() * 252, weights))
portfolio_volatility = np.sqrt(portfolio_variance)
print(f"Expected Portfolio Return: {portfolio_return:.2%}")
print(f"Portfolio Volatility: {portfolio_volatility:.2%}")

def sharpe_ratio(weights, returns): 
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return -portfolio_return / portfolio_volatility  # we minimize

# Objective function to minimize volatility
def portfolio_volatility(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1} # weights sum to 1
bounds = tuple((0, 1) for _ in range(len(tickers))) #weight (between 0 and 1)
initial_weights = [1/len(tickers)] * len(tickers)# initial guess (equal weighting)

############# Optimize the portfolio Based on Sharpe Ratio#############
optimized = minimize(sharpe_ratio, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
optimized_weights = optimized.x
print(f"Shares: {data.columns}")
print(f"Optimized Weights: {optimized_weights}")
portfolio_returns = returns.dot(optimized_weights)
cumulative_returns = (1 + portfolio_returns).cumprod()
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label='Optimized Portfolio')
plt.title('Optimized Portfolio Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
optimized_return = np.sum(returns.mean() * optimized_weights) * 252
optimized_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(returns.cov() * 252, optimized_weights)))
optimized_sharpe_ratio = optimized_return / optimized_volatility

print(f"Optimized Portfolio Return: {optimized_return:.2%}")
print(f"Optimized Portfolio Volatility: {optimized_volatility:.2%}")
print(f"Optimized Sharpe Ratio: {optimized_sharpe_ratio:.2f}")


''' OUTPUT
Shares: Index(['AAPL', 'AMZN', 'BABA', 'BAC', 'DIS', 'GOOGL', 'HD', 'JNJ', 'JPM', 'KO',
       'MA', 'META', 'MSFT', 'NFLX', 'NVDA', 'PEP', 'PG', 'TSLA', 'UNH', 'V',
       'XOM'],
      dtype='object', name='Ticker')
Optimized Weights: [2.46056003e-17 0.00000000e+00 2.54785000e-16 0.00000000e+00
 0.00000000e+00 0.00000000e+00 1.52857093e-16 1.88556352e-16
 2.23407263e-16 9.78095214e-17 0.00000000e+00 6.76893180e-17
 0.00000000e+00 9.57046358e-17 5.23611926e-01 8.48722741e-17
 8.23132679e-02 1.10725656e-01 9.55664883e-02 1.37156800e-17
 1.87782662e-01]
Optimized Portfolio Return: 56.28%
Optimized Portfolio Volatility: 36.46%
Optimized Sharpe Ratio: 1.54
'''

############# Optimize the portfolio Based on Minimising Risk#############
optimized = minimize(portfolio_volatility, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
optimized_weights = optimized.x
print(f"Shares: {data.columns}")
print(f"Optimized MRisk Weights Risks: {optimized_weights}")
portfolio_returns = returns.dot(optimized_weights)
cumulative_returns = (1 + portfolio_returns).cumprod()
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label='Optimized MRisk Portfolio')
plt.title('Optimized MRisk Portfolio Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
optimized_return = np.sum(returns.mean() * optimized_weights) * 252
optimized_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(returns.cov() * 252, optimized_weights)))
optimized_sharpe_ratio = optimized_return / optimized_volatility

print(f"Optimized MRisk Portfolio Return: {optimized_return:.2%}")
print(f"Optimized MRisk Portfolio Volatility: {optimized_volatility:.2%}")
print(f"Optimized MRisk Sharpe Ratio: {optimized_sharpe_ratio:.2f}")


''' OUTPUT
Shares: Index(['AAPL', 'AMZN', 'BABA', 'BAC', 'DIS', 'GOOGL', 'HD', 'JNJ', 'JPM', 'KO',
       'MA', 'META', 'MSFT', 'NFLX', 'NVDA', 'PEP', 'PG', 'TSLA', 'UNH', 'V',
       'XOM'],
      dtype='object', name='Ticker')
Optimized MRisk Weights Risks: [1.39116279e-17 7.35610607e-02 4.13681710e-02 2.87280739e-17
 0.00000000e+00 0.00000000e+00 0.00000000e+00 3.86117884e-01
 1.58270079e-17 1.98006450e-01 2.01627767e-18 0.00000000e+00
 3.80326309e-18 3.29228697e-02 1.33557656e-17 1.75676360e-17
 2.04537222e-01 2.68067342e-18 0.00000000e+00 0.00000000e+00
 6.34863415e-02]
Optimized MRisk Portfolio Return: 10.54%
Optimized MRisk Portfolio Volatility: 17.13%
Optimized MRisk Sharpe Ratio: 0.62
'''
