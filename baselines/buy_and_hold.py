import pandas as pd
import matplotlib.pyplot as plt

def buy_and_hold(data, initial_investment=1000000):
    # Calculate the number of units purchased at the initial price
    initial_price = data['open'].iloc[0]
    units_purchased = initial_investment / initial_price
    
    # Calculate the net worth history based on the units purchased and the price over time
    net_worth_history = data['open'] * units_purchased
    return net_worth_history

# Assuming you have 'eth_price.csv' in the correct format and filepath
data = pd.read_csv('/home/qian/qian/CryptoTrade/data/eth_202401.csv', parse_dates=['snapped_at'], index_col='snapped_at')
net_worth_history = buy_and_hold(data, 1000000)  # Pass in the initial investment amount here

# Calculate the return of the strategy
strategy_return = net_worth_history.iloc[-1] / net_worth_history.iloc[0] - 1

# Create a DataFrame for storing the return and the coin name
results_df = pd.DataFrame({
    'Coin': ['ETH'],  # Coin name
    'Strategy Return': [strategy_return]  # Strategy return
})

# Save the results to a CSV file
results_df.to_csv('strategy_return.csv', index=False)

print(f"Strategy Return: {strategy_return*100:.2f}%")