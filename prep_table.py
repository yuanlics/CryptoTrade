import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

sma_periods = [5, 10, 15, 20, 30]
starting_dates = ['2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01', '2024-01-01']
ending_dates = ['2023-09-30', '2023-10-31', '2023-11-30', '2023-12-31', '2024-01-31']

if __name__ == '__main__':

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


    for starting_date, ending_date in zip(starting_dates, ending_dates):
        y, m, _ = starting_date.split('-')
        ym = f'{y}{m}'
        df_m = df[(df['date'] >= starting_date) & (df['date'] <= ending_date)]
        print(len(df_m))
        stat = [df_m.iloc[0]['open'], df_m['open'].max(), df_m['open'].min(), df_m.iloc[-1]['open']]
        print([f'{s:.2f}' for s in stat])
        df_m.to_csv(f'../data/eth_{ym}.csv', index=False)

        # macd_signal = df_m['MACD'] - df_m['Signal_Line']
        # macd_signal = macd_signal / macd_signal.abs().max()
        # slma_5_20_signal = df_m['SMA_5'] - df_m['SMA_20']
        # slma_5_20_signal = slma_5_20_signal / slma_5_20_signal.abs().max()
        # dates = df_m['snapped_at'].apply(lambda x: x.split('-')[2].split(' ')[0])
        # prices = df_m['open']
        # prices = (prices - prices.min()) / (prices.max() - prices.min()) * 2 - 1
        
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.plot(dates, prices, label='open')
        # ax.plot(dates, macd_signal, label='macd_signal (1 to sell, -1 to buy)')
        # ax.plot(dates, slma_5_20_signal, label='slma_5_20_signal (1 to sell, -1 to buy)')
        # # ax.plot(df_m['date'], df_m['SMA_5'], label='SMA_5')
        # # ax.plot(df_m['date'], df_m['SMA_10'], label='SMA_10')
        # # ax.plot(df_m['date'], df_m['SMA_15'], label='SMA_15')
        # # ax.plot(df_m['date'], df_m['SMA_20'], label='SMA_20')
        # # ax.plot(df_m['date'], df_m['SMA_30'], label='SMA_30')
        # # ax.plot(df_m['date'], df_m['EMA_12'], label='EMA_12')
        # # ax.plot(df_m['date'], df_m['EMA_26'], label='EMA_26')
        # # ax.plot(df_m['date'], df_m['MACD'], label='MACD')
        # # ax.plot(df_m['date'], df_m['Signal_Line'], label='Signal_Line')
        # ax.axhline(0, color='black', lw=1, ls='--')
        # ax.set_xlabel('Date')
        # ax.set_ylabel('Price')
        # ax.set_title(f'ETH Price {ym}')
        # ax.legend()
        # plt.savefig(f'eth_run/eth_{ym}.png')
