from ib_insync import *

# Connect to IB API
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)  # Ensure the port matches the TWS/IB Gateway API settings

# Check account balance (Optional)
account = ib.accountSummary()
print("Account Summary:")
for summary in account:
    print(f'{summary.tag}: {summary.value} {summary.currency}')

ib.sleep(1)  # Adjust sleep time based on how long you expect to wait
# Disconnect
ib.disconnect()
print("Summary completed.")