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
# for we in range(1):
    ticker = tickers[we]
    order_num = order_optimal[we]
    if order_num <= 0:
        continue
    stock = Stock(ticker, 'SMART', 'EUR')#'USD'
    # Request market data to ensure the stock details are correct (Optional)
    ib.qualifyContracts(stock)
    print(f"Contract Details: {stock}")
    # Create a market order to buy shares
    order = MarketOrder('BUY', order_num)
    # Place the order
    trade = ib.placeOrder(stock, order)
    # Wait for the order to be filled
    trade.filledEvent += lambda trade, fill: print(f'Order Filled: {fill}')
    ib.sleep(1)  # Adjust sleep time based on how long you expect to wait
    # Disconnect
ib.disconnect()
print("Trade completed.")
