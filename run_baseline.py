# preprocess table and run baseline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from eth_env import ETHTradingEnv
from argparse import Namespace

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

strategies = ['SMA', 'MACD']
# strategies = ['SMA', 'MACD', 'SLMA', 'BollingerBands', 'buy_and_hold', 'optimal', 'LSTM', 'Multimodal']
sma_periods = [5, 10, 15, 20, 30]
dates = ['2022-02-01','2023-02-01', '2024-02-01']
VAL_START, VAL_END = dates[-3], dates[-2]
TEST_START, TEST_END = dates[-2], dates[-1]
RATIO = 0.5 # txn amount ratio
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


df = pd.read_csv('data/eth_daily.csv')
df['date'] = pd.to_datetime(df['snapped_at'])

# SMA
for period in sma_periods:
    df[f'SMA_{period}'] = df['open'].rolling(window=period).mean()
    df[f'STD_{period}'] = df['open'].rolling(window=period).std()

# MACD and Signal Line
df['EMA_12'] = df['open'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['open'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# dataset stats
for mi in range(len(dates)-1):
    starting_date = dates[mi]
    ending_date = dates[mi+1]
    y, m, _ = starting_date.split('-')
    df_m = df[(df['date'] >= starting_date) & (df['date'] < ending_date)]
    print(f'{starting_date} to {ending_date} length:', len(df_m))
    stat = [df_m.iloc[0]['open'], df_m['open'].max(), df_m['open'].min(), df_m.iloc[-1]['open']]
    print('open, max, min, close:', [f'{s:.2f}' for s in stat])
    # df_m.to_csv(f'data/eth_f'{y}{m}'.csv', index=False)
print()

# create dataset code for lstm
def create_dataset(dataset, look_back=5):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# LSTM strategy function
def lstm_strategy(df, start_date, end_date, look_back=5):
    # Filter the data
    data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    data = data['open'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Create the dataset
    X, Y = create_dataset(data_scaled, look_back)
    
    # Reshape X for sklearn compatibility
    X = X.reshape(X.shape[0], look_back)
    
    # Split the data into training and test sets
    train_size = int(len(X) * 0.67)
    trainX, trainY = X[:train_size], Y[:train_size]
    
    # Define and train the linear regression model
    model = LinearRegression()
    model.fit(trainX, trainY)
    
    # Make predictions
    last_train_batch = trainX[-1:].reshape(1, look_back)
    next_day_prediction = model.predict(last_train_batch)
    next_day_prediction = scaler.inverse_transform(next_day_prediction.reshape(-1, 1))
    current_price = scaler.inverse_transform(trainY[-1].reshape(-1, 1))
    
    # Decide action based on prediction, buy, sell or hold
    if next_day_prediction > current_price:
        action = 'Buy'
    elif next_day_prediction < current_price:
        action = 'Sell'
    else:
        action = 0
    
    return action

    
    

# 1st strategy: Simple MA 
# when the asset's open price is below the its SMA, and the volume is above the its SMA it's a buying signal, and vice versa for selling.

# 2nd strategy: MACD
# MACD = 12-day EMA - 26-day EMA
# Signal Line = 9-day EMA of MACD
# When MACD crosses above the signal line, it's a buying signal, and vice versa for selling.

# 3rd strategy: short and long strategy (SLMA) - If the short period SMA is below the long period SMA, it means that the trend is going down, so it's a sell signal, it's also known as the death cross.
# Otherwise, the trend is shiftting up, and it's a buy signal, it's also called the golden cross.
    
# 4th strategy: Bollinger Bands




def run_strategy(strategy, sargs):
    env = ETHTradingEnv(Namespace(starting_date=sargs['starting_date'], ending_date=sargs['ending_date']))
    df_tmp = df[(df['date'] >= sargs['starting_date']) & (df['date'] <= sargs['ending_date'])]
    df_tmp.reset_index(drop=True, inplace=True)
    state, reward, done, info = env.reset()  # only use env to act and track profit

    starting_net_worth = state['net_worth']
    month_irrs = []
    previous_month = int(sargs['starting_date'].split('-')[1])
    month_starting_net_worth = state['net_worth']
    previous_signal = None  # Track the previous day signal
    # Iterate through each row in the DataFrame to simulate trading
    for index, row in df_tmp.iterrows():
        open_price = state['open']
        cash = state['cash']
        eth_held = state['eth_held']
        net_worth = state['net_worth']
        date = row['date']
        y, m, d = date.year, date.month, date.day
        if previous_month != m:
            month_irrs.append((net_worth / month_starting_net_worth) - 1)
            month_starting_net_worth = net_worth
            previous_month = m
        if done:
            break

        if strategy == 'SMA':
            period = sargs['period']
            sma_column = f'SMA_{period}'
            ratio = sargs['ratio']
            current_signal = 'hold'
            if open_price > row[sma_column]:  # golden cross?
                current_signal = 'buy'
            elif open_price < row[sma_column]:  # death cross?
                current_signal = 'sell'
                
            action = 0
            if current_signal != previous_signal:
                if current_signal == 'buy' and cash > 0:
                    action = -ratio
                elif current_signal == 'sell' and eth_held > 0:
                    action = ratio
            previous_signal = current_signal
                
        elif strategy == 'SLMA':
            ratio = sargs['ratio']
            short = sargs['short']
            long = sargs['long']
            current_signal = 'hold'
            if row[short] > row[long]:  # golden cross?
                current_signal = 'buy'
            elif row[short] < row[long]:  # death cross?
                current_signal = 'sell'

            action = 0
            if current_signal != previous_signal:
                if current_signal == 'buy':
                    action = -ratio
                elif current_signal == 'sell' and eth_held > 0:
                    action = ratio
            previous_signal = current_signal

        elif strategy == 'MACD':
            ratio = sargs['ratio']
            current_signal = 'hold'
            if row['MACD'] < row['Signal_Line']:
                current_signal = 'buy'
            elif row['MACD'] > row['Signal_Line']:
                current_signal = 'sell'

            action = 0
            if current_signal != previous_signal:
                if current_signal == 'buy' and cash > 0:
                    action = -ratio
                elif current_signal == 'sell' and eth_held > 0:
                    action = ratio
            previous_signal = current_signal

        elif strategy == 'BollingerBands':
            period = sargs['period']  # e.g., 20 for a 20-day SMA
            multiplier = sargs['multiplier']  # Commonly set to 2
            ratio = sargs['ratio']
            sma = row[f'SMA_{period}']
            sd = row[f'STD_{period}']
            
            upper_band = sma + (sd * multiplier)
            lower_band = sma - (sd * multiplier)

            current_signal = 'hold'
            if open_price < lower_band:
                current_signal = 'buy'
            elif open_price > upper_band:
                current_signal = 'sell'

            action = 0
            if current_signal != previous_signal:
                if current_signal == 'buy' and cash > 0:
                    action = -ratio
                elif current_signal == 'sell' and eth_held > 0:
                    action = ratio
            previous_signal = current_signal

        elif strategy == 'buy_and_hold':
            action = 0
            if cash > 0:
                action = -1
        
        # here to add LSTM strategy
        elif strategy == 'LSTM':
            action = lstm_strategy(df, sargs['starting_date'], sargs['ending_date'], look_back=1)
            ratio = sargs['ratio']
            if action == 'Buy' and cash > 0:
                action = -ratio
            elif action == 'Sell' and eth_held > 0:
                action = ratio
            else:
                action = 0

        elif strategy == 'optimal':
            next_open = df_tmp.iloc[index+1]['open']
            if open_price < next_open:
                action = -1
            elif open_price > next_open:
                action = 1
            else:
                action = 0

        else:
            raise ValueError('Invalid strategy')

        state, reward, done, info = env.step(action)


    net_worth = state['net_worth']
    total_irr = (net_worth / starting_net_worth) - 1
    month_irrs = np.array(month_irrs) * 100
    month_irr_mean = np.mean(month_irrs)
    month_irr_std = np.std(month_irrs)
    month_risk_free_rate = 0.01 * 100  # e.g., US treasury bond
    result = {
        'total_irr': total_irr,
        'month_irr_mean': month_irr_mean,
        'month_irr_std': month_irr_std,
        'sharp_ratio': (month_irr_mean - month_risk_free_rate) / month_irr_std,
    }
    result_str = f'Total IRR: {total_irr*100:.2f}%, Month Mean: {month_irr_mean:.2f}%, Month Std: {month_irr_std:.2f}%, Sharp Ratio: {result["sharp_ratio"]:.2f}'
    print(result_str)
    

strategy = 'LSTM'
print(strategy)
run_strategy(strategy, {'ratio': RATIO, 'starting_date': VAL_START, 'ending_date': VAL_END})

# strategy = 'optimal'
# print(strategy)
# run_strategy(strategy, {'starting_date': VAL_START, 'ending_date': VAL_END})

# print(strategy)
# run_strategy(strategy, {'starting_date': TEST_START, 'ending_date': TEST_END})


# strategy = 'buy_and_hold'
# print(strategy)
# run_strategy(strategy, {'starting_date': VAL_START, 'ending_date': VAL_END})

# print(strategy)
# run_strategy(strategy, {'starting_date': TEST_START, 'ending_date': TEST_END})


# strategy = 'SMA'
# for period in sma_periods:
#     sargs = {'period': period, 'ratio': RATIO, 'starting_date': VAL_START, 'ending_date': VAL_END}
#     print(f'{strategy}, Period: {period}')
#     run_strategy(strategy, sargs)

# period = 30
# print(f'{strategy}, Period: {period}')
# sargs = {'period': period, 'ratio': RATIO, 'starting_date': TEST_START, 'ending_date': TEST_END}
# run_strategy(strategy, sargs)


# strategy = 'SLMA'
# # for i in range(len(sma_periods)-1):
# #     for j in range(i+1, len(sma_periods)):
# #         short = f'SMA_{sma_periods[i]}'
# #         long = f'SMA_{sma_periods[j]}'
# #         sargs = {'ratio': RATIO, 'short': short, 'long': long, 'starting_date': VAL_START, 'ending_date': VAL_END}
# #         print(f'{strategy}, Short: {short}, Long: {long}')
# #         run_strategy(strategy, sargs)

# short, long = 'SMA_20', 'SMA_30'
# sargs = {'ratio': RATIO, 'short': short, 'long': long, 'starting_date': TEST_START, 'ending_date': TEST_END}
# print(f'{strategy}, Short: {short}, Long: {long}')
# run_strategy(strategy, sargs)


# strategy = 'MACD'
# sargs = {'ratio': RATIO, 'starting_date': VAL_START, 'ending_date': VAL_END}
# print(f'{strategy}')
# run_strategy(strategy, sargs)

# sargs = {'ratio': RATIO, 'starting_date': TEST_START, 'ending_date': TEST_END}
# print(f'{strategy}')
# run_strategy(strategy, sargs)


# strategy = 'BollingerBands'
# period = 20
# multiplier = 2
# sargs = {'period': period, 'multiplier': multiplier, 'ratio': RATIO, 'starting_date': VAL_START, 'ending_date': VAL_END}
# print(f'{strategy}, Period: {period}, Multiplier: {multiplier}')
# run_strategy(strategy, sargs)

# period = 20
# multiplier = 2
# sargs = {'period': period, 'multiplier': multiplier, 'ratio': RATIO, 'starting_date': TEST_START, 'ending_date': TEST_END}
# print(f'{strategy}, Period: {period}, Multiplier: {multiplier}')
# run_strategy(strategy, sargs)
