from ib_insync import *
import pandas as pd
import yfinance as yf

# Connect to IB TWS or Gateway
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=4)  # Use port 7497 for paper trading, 4001 for live

# Fetch portfolio positions
positions = ib.positions()


tickers_dict = {
    'AD': 'AD.AS',  
    'BNP': 'BNP.PA', 
    'SAN': 'SAN.PA', 
    'SAP': 'SAP.DE',  
    'UNA': 'UNA.AS' 
}
# Fetch market data for each position using yfinance
def get_market_price_yahoo(symbol):
    ticker_data = yf.Ticker(tickers_dict[symbol])
    market_price = ticker_data.history(period='1mo')['Close'].iloc[-1]  # Get the latest closing price
    return market_price

# Prepare a list to hold portfolio information
portfolio_data = []

for pos in positions:
    contract = pos.contract
    symbol = contract.symbol
    position = pos.position
    avg_cost = pos.avgCost
    
    # Get current market price from Yahoo Finance
    market_price = get_market_price_yahoo(symbol)
    
    # Calculate initial value and current value
    initial_value = position * avg_cost
    current_value = position * market_price
    gain_loss = current_value - initial_value
    
    portfolio_data.append({
        'Symbol': symbol,
        'Position': position,
        'Avg Cost': avg_cost,
        'Market Price': market_price,
        'Initial Value': initial_value,
        'Current Value': current_value,
        'Gain/Loss': gain_loss
    })

# Create a DataFrame for easy viewing
portfolio_df = pd.DataFrame(portfolio_data)

# Calculate total initial value, total current value, and total gain/loss
total_initial_value = portfolio_df['Initial Value'].sum()
total_current_value = portfolio_df['Current Value'].sum()
total_gain_loss = portfolio_df['Gain/Loss'].sum()

print(portfolio_df)
print(f"\nTotal Initial Value: ${total_initial_value:,.2f}")
print(f"Total Current Value: ${total_current_value:,.2f}")
print(f"Total Gain/Loss: ${total_gain_loss:,.2f}")

# Disconnect from IB
ib.disconnect()
