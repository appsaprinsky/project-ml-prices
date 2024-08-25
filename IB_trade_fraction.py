from ib_insync import *
import pandas as pd

opt_portfolio = pd.read_csv("portfolio/output/optimised_portfolio.csv")
optimal_weights = list(opt_portfolio['Weights'])
tickers = list(opt_portfolio['Tickets'])
total_wealth_stocks = 100
order_optimal = [int(num * total_wealth_stocks) for num in optimal_weights]
print(order_optimal)

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)

for we in range(len(order_optimal)):
    ticker = tickers[we]
    order_num = order_optimal[we] # Amount in EUR you want to invest per ticker
    if order_num <= 0:
        continue

    # Define the stock contract
    stock = Stock(ticker, 'SMART', 'EUR')
    # Request market data to ensure the stock details are correct (Optional)
    ib.qualifyContracts(stock)
    print(f"Contract Details: {stock}")

    # Get the current market price
    market_data = ib.reqMktData(stock, '', False, False)
    ib.sleep(1)  # Allow some time for the market data to come through
    current_price = market_data.last

    if current_price is None or current_price <= 0:
        print(f"Could not retrieve market price for {ticker}. Skipping.")
        continue

    # Calculate the fractional share based on the investment amount
    shares_to_buy = order_num / current_price

    # Create a market order using fractional shares
    order = MarketOrder('BUY', shares_to_buy, cashQty=order_num)
    
    # Place the order
    trade = ib.placeOrder(stock, order)
    
    # Wait for the order to be filled
    trade.filledEvent += lambda trade, fill: print(f'Order Filled: {fill}')
    ib.sleep(1)  # Adjust sleep time based on how long you expect to wait

# Disconnect from IB
ib.disconnect()
print("Trade completed.")
