from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

DATE_START = '2020-01-01'
DATE_END = '2024-08-23'
tickers = [
    'SAP.DE',   # SAP SE (Germany, XETRA)
    'SIE.DE',   # Siemens AG (Germany, XETRA)
    'ASML.AS',  # ASML Holding N.V. (Netherlands, Euronext Amsterdam)
    'VOW3.DE',  # Volkswagen AG (Germany, XETRA)
    'AIR.PA',   # Airbus SE (France, Euronext Paris)
    'BNP.PA',   # BNP Paribas S.A. (France, Euronext Paris)
    'HSBA.L',   # HSBC Holdings Plc (UK, London Stock Exchange)
    'OR.PA',    # L'Oréal S.A. (France, Euronext Paris)
    'NESN.SW',  # Nestlé S.A. (Switzerland, SIX Swiss Exchange)
    'NOVN.SW',  # Novartis AG (Switzerland, SIX Swiss Exchange)
    'UNA.AS',   # Unilever N.V. (Netherlands, Euronext Amsterdam)
    'BAS.DE',   # BASF SE (Germany, XETRA)
    'SAN.PA',   # Sanofi (France, Euronext Paris)
    'BARC.L',   # Barclays Plc (UK, London Stock Exchange)
    'AZN.L',    # AstraZeneca Plc (UK, London Stock Exchange)
    'AD.AS',    # Koninklijke Ahold Delhaize N.V. (Netherlands, Euronext Amsterdam)
    'BMW.DE',   # Bayerische Motoren Werke AG (Germany, XETRA)
    'ENEL.MI'   # Enel S.p.A. (Italy, Borsa Italiana)
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

opt_portfolio = pd.DataFrame({"Tickets":list(data.columns), "Weights":optimized_weights})
opt_portfolio.to_csv("portfolio/output/optimised_portfolio.csv")

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

mrisk_portfolio = pd.DataFrame({"Tickets":list(data.columns), "Weights":optimized_weights})
mrisk_portfolio.to_csv("portfolio/output/MRisk_optimised_portfolio.csv")
