import numpy as np

# objective function to minimize negative sharpe ratio
def sharpe_ratio(weights, returns): 
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return -portfolio_return / portfolio_volatility  

# objective function to minimize volatility
def MR_portfolio_volatility(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))