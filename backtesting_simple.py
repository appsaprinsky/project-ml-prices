import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Input data: List of tickers and number of shares for each ticker
tickers = {
    'AAPL': 1,  # Example: 1 share of Apple
    'MSFT': 2,  # Example: 2 shares of Microsoft
    'GOOGL': 1  # Example: 1 share of Google
}

tickers = {
    'AD.AS': 1,  # Example: 1 share of Apple
    'ASML': 1,  # Example: 2 shares of Microsoft
    'BNP.PA': 1,  # Example: 1 share of Google
    'SAN': 1,  # Example: 1 share of Apple
    'SAP': 1,  # Example: 2 shares of Microsoft
    'UNA.AS': 1  # Example: 1 share of Google
}

START_DATE = '2021-01-01'
END_DATE = '2024-08-23'

# Download historical data for all tickers
data = yf.download(list(tickers.keys()), start=START_DATE, end=END_DATE)['Adj Close'].dropna()

# Create a DataFrame to store the number of shares
shares_df = pd.DataFrame(index=data.index, columns=data.columns)
for ticker, num_shares in tickers.items():
    shares_df[ticker] = num_shares

# Calculate the daily value of each stock position
portfolio_value = data * shares_df

# Calculate the total portfolio value for each day by summing across columns
total_portfolio_value = portfolio_value.sum(axis=1)

# Calculate the cumulative returns of the portfolio
cumulative_returns = total_portfolio_value / total_portfolio_value.iloc[0]

# Plot the cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label='Portfolio Cumulative Returns')
plt.title('Portfolio Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Print final cumulative return value
print(f"Final Cumulative Return: {cumulative_returns.iloc[-1]:.2f}")
