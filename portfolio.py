from tickers.EU import *
# from tickers.USA import *
from functions.objective_functions import *

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

DATE_START = '2020-01-01'
DATE_END = '2024-08-23'

data = yf.download(tickers, start=DATE_START, end=DATE_END)['Adj Close'] 
returns = data.pct_change().dropna() # calculate daily returns
weights = np.array([1/len(tickers)] * len(tickers)) # define portfolio weights (equal weighting)
portfolio_return = np.sum(returns.mean() * weights) * 252  # calculate the expected portfolio return # annualize return (252 trading days)
portfolio_variance = np.dot(weights.T, np.dot(returns.cov() * 252, weights))
portfolio_volatility = np.sqrt(portfolio_variance)
print(f"Expected Portfolio Return: {portfolio_return:.2%}")
print(f"Portfolio Volatility: {portfolio_volatility:.2%}")
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
opt_portfolio = pd.DataFrame({"Tickets":list(data.columns), "Weights":optimized_weights})
opt_portfolio.to_csv("portfolio/output/optimised_portfolio.csv")

############# Optimize the portfolio Based on Minimising Risk#############
optimized = minimize(MR_portfolio_volatility, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
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
mrisk_portfolio = pd.DataFrame({"Tickets":list(data.columns), "Weights":optimized_weights})
mrisk_portfolio.to_csv("portfolio/output/MRisk_optimised_portfolio.csv")
