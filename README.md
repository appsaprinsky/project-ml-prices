# project-ml-prices

# TO DO
## 1) Backtesting
Check the portfolio value if only whole shares are traded
## 2) Current Account trade info



# Portfolio Optimization using 2 methods

### Table: Shares and Optimized Weights for Minimized Risk

Here's the table with the tickers and the optimized weights:

| Ticker | Optimized Weight |
|--------|------------------|
| AAPL   | 0.00             |
| AMZN   | 0.00             |
| BABA   | 0.00             |
| BAC    | 0.00             |
| DIS    | 0.00             |
| GOOGL  | 0.00             |
| HD     | 0.00             |
| JNJ    | 0.00             |
| JPM    | 0.00             |
| KO     | 0.00             |
| MA     | 0.00             |
| META   | 0.00             |
| MSFT   | 0.00             |
| NFLX   | 0.00             |
| NVDA   | 0.52             |
| PEP    | 0.00             |
| PG     | 0.08             |
| TSLA   | 0.11             |
| UNH    | 0.10             |
| V      | 0.00             |
| XOM    | 0.19             |

### Notes:
These weights represent the proportion of the total investment that should be allocated to each stock in order to maximize the Sharpe Ratio based on the given optimization strategy.



This project demonstrates how to optimize a portfolio of stocks by maximizing the Sharpe Ratio using mean-variance optimization. The code uses historical stock data retrieved from Yahoo Finance, calculates daily returns, and applies optimization techniques to determine the best portfolio weights.

To create a table with the shares and their corresponding optimized weights for the minimized risk (rounded to two decimal points), we can structure it similarly to how we did for the Sharpe ratio optimization. This table will present the tickers and the weights that minimize portfolio risk:

### Table: Shares and Optimized Weights for Minimized Risk

| Ticker | Optimized Weight |
|--------|------------------|
| AAPL   | 0.00             |
| AMZN   | 0.07             |
| BABA   | 0.04             |
| BAC    | 0.00             |
| DIS    | 0.00             |
| GOOGL  | 0.00             |
| HD     | 0.00             |
| JNJ    | 0.39             |
| JPM    | 0.00             |
| KO     | 0.20             |
| MA     | 0.00             |
| META   | 0.00             |
| MSFT   | 0.00             |
| NFLX   | 0.03             |
| NVDA   | 0.00             |
| PEP    | 0.00             |
| PG     | 0.20             |
| TSLA   | 0.00             |
| UNH    | 0.00             |
| V      | 0.00             |
| XOM    | 0.06             |

This distribution is optimized to minimize portfolio volatility, likely favoring stocks with historically lower volatility and stable returns, thus aligning with risk-averse investment strategies.


# Portfolio Optimization Using Python

## 1. Data Retrieval and Preprocessing

We use `yfinance` to download historical adjusted closing prices for a list of tickers from `2020-01-01` to `2024-08-01`.

```python
import yfinance as yf

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
data = yf.download(tickers, start='2020-01-01', end='2024-08-01')['Adj Close']
print(data.head())
```


### Calculate Daily Returns

The daily returns are calculated as:
```
\[
\text{Daily Returns} = \frac{P_t - P_{t-1}}{P_{t-1}}
\]

Where:
- \( P_t \) is the price at time \( t \).
- \( P_{t-1} \) is the price at time \( t-1 \).
```
```python
returns = data.pct_change().dropna()
print(returns.head())
```

## 2. Initial Portfolio Setup

### Define Equal Weights

We initialize with equal weighting for each stock in the portfolio:

\[
\text{Weight}_i = \frac{1}{n}
\]

Where \( n \) is the number of stocks (tickers).

```python
import numpy as np

weights = np.array([1/len(tickers)] * len(tickers))
```

### Calculate Expected Annualized Portfolio Return

\[
\text{Expected Portfolio Return} = \sum_{i=1}^{n} w_i \cdot \mu_i \times 252
\]

Where:
- \( w_i \) is the weight of stock \( i \).
- \( \mu_i \) is the mean daily return of stock \( i \).
- \( 252 \) is the number of trading days in a year.

```python
portfolio_return = np.sum(returns.mean() * weights) * 252
```

### Calculate Portfolio Variance and Volatility

\[
\text{Portfolio Variance} = \mathbf{w}^T \cdot \Sigma \cdot \mathbf{w} \times 252
\]

\[
\text{Portfolio Volatility} = \sqrt{\text{Portfolio Variance}}
\]

Where:
- \( \mathbf{w} \) is the weight vector of the portfolio.
- \( \Sigma \) is the covariance matrix of the asset returns.

```python
portfolio_variance = np.dot(weights.T, np.dot(returns.cov() * 252, weights))
portfolio_volatility = np.sqrt(portfolio_variance)
print(f"Expected Portfolio Return: {portfolio_return:.2%}")
print(f"Portfolio Volatility: {portfolio_volatility:.2%}")
```

## 3. Optimization Objective Function

### Define the Sharpe Ratio

\[
\text{Sharpe Ratio} = \frac{\text{Expected Portfolio Return}}{\text{Portfolio Volatility}}
\]

To maximize the Sharpe Ratio, we minimize its negative value:

\[
-\text{Sharpe Ratio} = -\frac{\text{Expected Portfolio Return}}{\text{Portfolio Volatility}}
\]

```python
def sharpe_ratio(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return -portfolio_return / portfolio_volatility
```

## 4. Constraints and Bounds for Optimization

### Constraints

Ensure that the sum of the portfolio weights equals 1 (full investment):

\[
\sum_{i=1}^{n} w_i = 1
\]

```python
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
```

### Bounds

Each weight should be between 0 and 1 (no short selling):

\[
0 \leq w_i \leq 1 \quad \text{for each } i
\]

```python
bounds = tuple((0, 1) for _ in range(len(tickers)))
initial_weights = [1/len(tickers)] * len(tickers)
```

## 5. Optimization

Use the `SLSQP` method to optimize weights to maximize the Sharpe Ratio (minimize negative Sharpe Ratio):

```python
from scipy.optimize import minimize

optimized = minimize(sharpe_ratio, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
optimized_weights = optimized.x
print(f"Optimized Weights: {optimized_weights}")
```

## 6. Post-Optimization Metrics and Visualization

### Calculate Portfolio Returns with Optimized Weights

\[
\text{Optimized Portfolio Return} = \sum_{i=1}^{n} w_i \cdot r_i
\]

Where \( r_i \) is the daily return of stock \( i \).

```python
portfolio_returns = returns.dot(optimized_weights)
```

### Cumulative Returns for Visualization

\[
\text{Cumulative Returns} = \prod_{t=1}^{T} (1 + r_t)
\]

```python
cumulative_returns = (1 + portfolio_returns).cumprod()
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label='Optimized Portfolio')
plt.title('Optimized Portfolio Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
```

### Calculate Optimized Portfolio Metrics

- **Optimized Return**:

    \[
    \text{Optimized Portfolio Return} = \sum_{i=1}^{n} w_i \cdot \mu_i \times 252
    \]

- **Optimized Volatility**:

    \[
    \text{Optimized Portfolio Volatility} = \sqrt{\text{Optimized Portfolio Variance}}
    \]

- **Optimized Sharpe Ratio**:

    \[
    \text{Optimized Sharpe Ratio} = \frac{\text{Optimized Portfolio Return}}{\text{Optimized Portfolio Volatility}}
    \]

```python
optimized_return = np.sum(returns.mean() * optimized_weights) * 252
optimized_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(returns.cov() * 252, optimized_weights)))
optimized_sharpe_ratio = optimized_return / optimized_volatility

print(f"Optimized Portfolio Return: {optimized_return:.2%}")
print(f"Optimized Portfolio Volatility: {optimized_volatility:.2%}")
print(f"Optimized Sharpe Ratio: {optimized_sharpe_ratio:.2f}")
```

## Summary of Formulas Used

1. **Daily Returns**: \( \text{Daily Returns} = \frac{P_t - P_{t-1}}{P_{t-1}} \)
2. **Expected Portfolio Return**: \( \text{Expected Portfolio Return} = \sum_{i=1}^{n} w_i \cdot \mu_i \times 252 \)
3. **Portfolio Variance**: \( \text{Portfolio Variance} = \mathbf{w}^T \cdot \Sigma \cdot \mathbf{w} \times 252 \)
4. **Portfolio Volatility**: \( \text{Portfolio Volatility} = \sqrt{\text{Portfolio Variance}} \)
5. **Sharpe Ratio**: \( \text{Sharpe Ratio} = \frac{\text{Expected Portfolio Return}}{\text{Portfolio Volatility}} \)
6. **Negative Sharpe Ratio** (for minimization): \( -\frac{\text{Expected Portfolio Return}}{\text{Portfolio Volatility}} \)
7. **Optimization Constraints**: \( \sum_{i=1}^{n} w_i = 1 \) and \( 0 \leq w_i \leq 1 \)

These formulas and code components help optimize the portfolio by adjusting the weights of each asset to maximize the risk-adjusted return (Sharpe Ratio).

