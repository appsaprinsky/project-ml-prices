from ib_insync import *
import pandas as pd

# Connect to IB API
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=2)  # Ensure the port matches the TWS/IB Gateway API settings

# Get all positions in the account
positions = ib.positions()
print(positions)
tickets = []
values = []
avgcosts = []

# Check and print positions
if positions:
    print("Your current positions:")
    for position in positions:
        tickets.append(position.contract.symbol)
        values.append(position.position)
        avgcosts.append(position.avgCost)
        print(f'Symbol: {position.contract.symbol}, Shares: {position.position}, Avg Cost: {position.avgCost}')
else:
    print("No positions found.")

positions_pd = pd.DataFrame({'Ticket':tickets, 'Value':values, 'AvgCost':avgcosts})
positions_pd.to_csv('portfolio/current_positions/' + position.account + '.csv')
# Disconnect
ib.disconnect()
