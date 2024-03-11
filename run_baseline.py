import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from prep_table import sma_periods
from eth_env import ETHTradingEnv
from argparse import Namespace

strategies = ['SMA', 'MACD']
yms_val = ['202310', '202312']
yms_test = ['202309', '202311', '202401']
yms_all = sorted(yms_val + yms_test)
RATIO = 1.0  # txn amount ratio 
GAS_LIMITS = 21000  # unit
GAS_PRICE = 70  # gwei
GAS_FEE = GAS_LIMITS * GAS_PRICE * 1e-9  # eth per txn
EX_RATE = 4e-3  # exchange fee = txn_amount * ex_rate
# GAS_FEE = 0
# EX_RATE = 0
starting_cash = 1_000_000
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# first strategy: Simple MA 
# when the asset's open price is below the its SMA, and the volume is above the its SMA it's a buying signal, and vice versa for selling.

# second strategy: MACD
# MACD = 12-day EMA - 26-day EMA
# Signal Line = 9-day EMA of MACD
# When MACD crosses above the signal line, it's a buying signal, and vice versa for selling.

# third strategy: short and long strategy, moving average crossover (MAC) - If the short period SMA is below the long period SMA, it means that the trend is going down, so it's a sell signal, it's also known as the death cross.
# Otherwise, the trend is shiftting up, and it's a buy signal, it's also called the golden cross.
def run_strategy(strategy, file_path, sargs):
    env = ETHTradingEnv(Namespace(ym=sargs['ym']))
    state, reward, done, info = env.reset()

    df = pd.read_csv(file_path)
    previous_signal = None  # Track the previous day signal
    
    # Iterate through each row in the DataFrame to simulate trading
    for index, row in df.iterrows():
        if index == len(df) - 1:  # no action on last day to avoid fee
            break
        open_price = state['open']
        cash = state['cash']
        eth_held = state['eth_held']

        if strategy == 'SMA':
            sma_column = sargs['column']
            ratio = sargs['ratio']
            current_signal = 'hold'
            if open_price > row[sma_column]:
                current_signal = 'buy'
            elif open_price < row[sma_column]:
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
            current_signal = 'hold'
            if row[short] < row[long]:
                current_signal = 'buy'
            elif row[short] > row[long]:
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

        elif strategy == 'buy':
            action = 0
            if cash > 0:
                action = -1

        elif strategy == 'sell':
            action = 0
            if eth_held > 0:
                action = 1

        else:
            raise ValueError('Invalid strategy')

        # uniform action entry
        state, reward, done, info = env.step(action)
        if done:
            break

    net_worth = cash + (eth_held * df.iloc[-1]['open'])
    roi = (net_worth / starting_cash) - 1
    return roi


for strategy in ['buy', 'sell']:
    roi_test = []
    for ym in yms_test:
        env = ETHTradingEnv(Namespace(ym=ym))
        file_path = f'../data/eth_{ym}.csv'
        roi = run_strategy(strategy, file_path, {'ym': ym})
        print(f'{strategy}, Time: {ym}, ROI: {roi*100:.2f}%')
        roi_test.append(roi)
    roi_test = np.array(roi_test) * 100
    print(f'ROI: {roi_test.mean():.2f}+-{roi_test.std():.2f}')


strategy = 'SMA'
roi_val = defaultdict(float)
for ym in yms_val:
    file_path = f'../data/eth_{ym}.csv'
    for period in sma_periods:
        # for ratio in np.arange(0.2, 1.1, 0.2):
        column = f'SMA_{period}'
        sargs = {'column': column, 'ratio': RATIO, 'ym': ym}
        roi = run_strategy(strategy, file_path, sargs)
        print(f'{strategy}, Column: {column}, Time: {ym}, ROI: {roi*100:.2f}%')
        roi_val[column] += roi
print(roi_val)

period = 30
roi_test = []
for ym in yms_test:
    file_path = f'../data/eth_{ym}.csv'
    column = f'SMA_{period}'
    sargs = {'column': column, 'ratio': RATIO, 'ym': ym}
    roi = run_strategy(strategy, file_path, sargs)
    print(f'{strategy}, Column: {column}, Time: {ym}, ROI: {roi*100:.2f}%')
    roi_test.append(roi)
roi_test = np.array(roi_test) * 100
print(f'ROI: {roi_test.mean():.2f}+-{roi_test.std():.2f}')


print()
strategy = 'MACD'
roi_test = []
for ym in yms_test:
# for ym in yms_all:
# for ym in ['202309']:
    file_path = f'../data/eth_{ym}.csv'
    # for ratio in np.arange(0.2, 1.1, 0.2):
    sargs = {'ratio': RATIO, 'ym': ym}
    roi = run_strategy(strategy, file_path, sargs)
    print(f'{strategy}, Time: {ym}, ROI: {roi*100:.2f}%')
    roi_test.append(roi)
roi_test = np.array(roi_test) * 100
print(f'ROI: {roi_test.mean():.2f}+-{roi_test.std():.2f}')


print()
strategy = 'SLMA'
roi_val = defaultdict(float)
for ym in yms_val:
# for ym in yms_all:
    file_path = f'../data/eth_{ym}.csv'
    ratio = 1
    for i in range(len(sma_periods)-1):
        for j in range(i+1, len(sma_periods)):
            short = f'SMA_{sma_periods[i]}'
            long = f'SMA_{sma_periods[j]}'
            sargs = {'ratio': 1, 'short': short, 'long': long, 'ym': ym}
            roi = run_strategy(strategy, file_path, sargs)
            print(f'{strategy}, Short: {short}, Long: {long}, Time: {ym}, ROI: {roi*100:.2f}%')
            roi_val[(short, long)] += roi
print(roi_val)

short, long = 'SMA_10', 'SMA_20'
# short, long = 'SMA_5', 'SMA_20'
roi_test = []
for ym in yms_test:
    file_path = f'../data/eth_{ym}.csv'
    ratio = 1
    sargs = {'ratio': ratio, 'short': short, 'long': long, 'ym': ym}
    roi = run_strategy(strategy, file_path, sargs)
    print(f'{strategy}, Short: {short}, Long: {long}, Time: {ym}, ROI: {roi*100:.2f}%')
    roi_test.append(roi)
roi_test = np.array(roi_test) * 100
print(f'ROI: {roi_test.mean():.2f}+-{roi_test.std():.2f}')


print()
strategy = 'BollingerBands'
period = 20
multiplier = 2
roi_test = []
for ym in yms_test:
    file_path = f'../data/eth_{ym}.csv'
    ratio = 1
    sargs = {'period': period, 'multiplier': multiplier, 'ratio': ratio, 'ym': ym}
    roi = run_strategy(strategy, file_path, sargs)
    print(f'{strategy}, Period: {period}, Multiplier: {multiplier}, Time: {ym}, ROI: {roi*100:.2f}%')
    roi_test.append(roi)
roi_test = np.array(roi_test) * 100
print(f'ROI: {roi_test.mean():.2f}+-{roi_test.std():.2f}')
