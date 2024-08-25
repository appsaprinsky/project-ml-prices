from ib_insync import *

# Connect to IB API
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=2)  # Ensure the port matches the TWS/IB Gateway API settings

# Get all positions in the account
positions = ib.positions()
print(positions)

# Check and print positions
if positions:
    print("Your current positions:")
    for position in positions:
        print(f'Symbol: {position.contract.symbol}, Shares: {position.position}, Avg Cost: {position.avgCost}')
else:
    print("No positions found.")

# Disconnect
ib.disconnect()
