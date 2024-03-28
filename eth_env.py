import numpy as np
import pandas as pd
import re
import random
from datetime import datetime, timedelta
import json
from argparse import Namespace
import os

PATH_PRICE = 'data/eth_daily.csv'
DIR_NEWS  = 'data/gnews'
PATH_TXN_STAT = 'data/eth_more_transaction_statistics.csv'
PRICE_TIME_FMT = "%Y-%m-%d %H:%M:%S UTC"
STARTING_NET_WORTH = 1_000_000
STARTING_CASH_RATIO = 0.5
# STARTING_CASH_RATIO = 1
GAS_LIMITS = 21000  # unit
GAS_PRICE = 70  # gwei
GAS_FEE = GAS_LIMITS * GAS_PRICE * 1e-9  # eth per txn
EX_RATE = 4e-3  # exchange fee = txn_amount * ex_rate
# GAS_FEE = 0
# EX_RATE = 0
SMA_PERIODS = [5, 10, 15, 20, 30]


class ETHTradingEnv:
    def __init__(self, args):
        starting_date, ending_date = args.starting_date, args.ending_date
        df = pd.read_csv(PATH_PRICE)
        df['date'] = pd.to_datetime(df['snapped_at'], format=PRICE_TIME_FMT)
        # SMA
        for period in SMA_PERIODS:
            df[f'SMA_{period}'] = df['open'].rolling(window=period).mean()
            df[f'STD_{period}'] = df['open'].rolling(window=period).std()
        # MACD and Signal Line
        df['EMA_12'] = df['open'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['open'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        self.data = df[(df['date'] >= starting_date) & (df['date'] <= ending_date)]  # only use ending_date for open price
        
        self.txn_stat = pd.read_csv(PATH_TXN_STAT)
        self.txn_stat['date'] = pd.to_datetime(self.txn_stat['day'], format="%d/%m/%y %H:%M")  # 27/1/24 0:00
        self.total_steps = len(self.data)
        self.starting_net_worth = STARTING_NET_WORTH
        self.starting_cash_ratio = STARTING_CASH_RATIO
        # self.reset()

    def get_close_state(self, today, next_day, first_day=False):
        next_open_price = next_day['open']
        close_net_worth = self.cash + self.eth_held * next_open_price
        close_roi = close_net_worth / self.starting_net_worth - 1  # return on investment
        today_roi = close_net_worth / self.last_net_worth - 1
        self.last_net_worth = close_net_worth

        date = today['snapped_at']
        parsed_time = datetime.strptime(date, PRICE_TIME_FMT)
        if first_day:
            parsed_time = parsed_time - timedelta(days=1)
        year, month, day = parsed_time.year, parsed_time.month, parsed_time.day

        # next day's opening technical indicators
        ma5 = next_day['SMA_5']
        ma10 = next_day['SMA_10']
        ma15 = next_day['SMA_15']
        ma20 = next_day['SMA_20']
        slma_signal = 'hold'
        short_ma = ma15
        long_ma = ma20
        if short_ma > long_ma:
            slma_signal = 'sell'
        elif short_ma < long_ma:
            slma_signal = 'buy'
        
        sma = next_day[f'SMA_20']
        sd = next_day[f'STD_20']
        multiplier = 2
        upper_band = sma + (sd * multiplier)
        lower_band = sma - (sd * multiplier)
        boll_signal = 'hold'
        if next_open_price < lower_band:
            boll_signal = 'buy'
        elif next_open_price > upper_band:
            boll_signal = 'sell'

        macd = next_day['MACD']
        macd_signal_line = next_day['Signal_Line']
        macd_signal = 'hold'
        if macd < macd_signal_line:
            macd_signal = 'buy'
        elif macd > macd_signal_line:
            macd_signal = 'sell'

        # today's txn stats
        txn_stat = self.txn_stat[self.txn_stat['date'] == parsed_time]
        if txn_stat.empty:
            num_txns = 'N/A'
            unique_addrs = 'N/A'
            value_transferred = 'N/A'
            avg_gas_price = 'N/A'
            total_gas_used = 'N/A'
            successful_txns = 'N/A'
        else:
            num_txns = txn_stat['total_transactions'].values[0]
            unique_addrs = txn_stat['unique_addresses'].values[0]
            value_transferred = txn_stat['total_value_transferred'].values[0]
            avg_gas_price = txn_stat['average_gas_price'].values[0]
            total_gas_used = txn_stat['total_gas_used'].values[0]
            successful_txns = txn_stat['successful_transactions'].values[0]

        # today's news
        news_path = f"{DIR_NEWS}/{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.json"
        if not os.path.exists(news_path):
            news = 'N/A'
        else:
            news = json.load(open(news_path))

        close_state = {  # selectively used in prompt
            'cash': self.cash,
            'eth_held': self.eth_held,
            'open': next_open_price,
            'net_worth': close_net_worth,
            'roi': close_roi,
            'today_roi': today_roi,
            'technical': {
                'short_long_ma_signal': slma_signal,
                'macd_signal': macd_signal,
                'bollinger_bands_signal': boll_signal,
            },
            'txnstat': {
                'num_transactions': num_txns,
                'unique_addresses': unique_addrs,
                'value_transferred': value_transferred,
                'average_gas_price': avg_gas_price,
                'total_gas_used': total_gas_used,
                'successful_txns': successful_txns,
            },
            'news': news,
            'date': date,
        }
        return close_state

    def reset(self):
        self.current_step = 0
        next_day = today = self.data.iloc[self.current_step]
        self.starting_price = today['open']
        self.cash = self.starting_net_worth * STARTING_CASH_RATIO
        self.eth_held = (self.starting_net_worth - self.cash) / self.starting_price
        self.last_net_worth = self.starting_net_worth
        close_state = self.get_close_state(today, next_day, first_day=True)
        info = {
            'starting_cash': self.cash,
        }
        reward = 0
        self.done = False
        self.last_state = close_state
        return close_state, reward, self.done, info

    # the agent receives last state and reward, takes an action, and receives new state and reward.
    # last state: yesterday's news, today's open price, cash, held ETH
    # last reward: yesterday's profit
    # action: buy, sell, or hold
    # new state: today's news, tomorrow's open price, cash, held ETH
    # new reward: today's profit
    def step(self, action):
        raw_action = action
        if type(action) == str:
            # actions = re.findall(r"[-+]?\d*\.\d+|\d+", action)
            actions = re.findall(r"-?(?:0(?:\.\d{1})|1\.0)", action)
            
            if len(actions) == 0:
                print(f'ERROR: Invalid llm response: {action}. Set to no action.')
                action = 0.00
            elif len(actions) == 1:
                action = float(actions[0])
            else:
                # print(f'Multiple actions in llm response: {action}. Pick one action.')
                # action = float(actions[0])
                action = float(actions[-1])
        
        if not -1 <= action <= 1:
            print(f"ERROR: Invalid action: {action}. Set to no action.")
            action = 0.00

        today = self.data.iloc[self.current_step]
        next_day = self.data.iloc[self.current_step + 1]
        open_price = today['open']
        next_open_price = next_day['open']  # assume today's close = next day's open
        
        if 0 < action <= 1 and self.eth_held > 0:  # Sell ETH
            eth_diff = abs(action) * self.eth_held
            cash_diff = eth_diff * open_price
            self.eth_held -= eth_diff
            self.cash += cash_diff
            self.cash -= GAS_FEE * open_price + cash_diff * EX_RATE
        elif -1 <= action < 0 and self.cash > 0:  # Buy ETH
            cash_diff = abs(action) * self.cash
            eth_diff = cash_diff / open_price
            self.cash -= cash_diff
            self.eth_held += eth_diff
            self.cash -= GAS_FEE * open_price + cash_diff * EX_RATE
        
        self.current_step += 1
        if self.current_step >= self.total_steps - 1:
            self.done = True

        close_state = self.get_close_state(today, next_day)
        reward = close_state['roi'] - self.last_state['roi']  # reward = today's roi gain.
        self.last_state = close_state
        info = {
            'raw_action': raw_action,
            'actual_action': action,
            'starting_cash': self.starting_net_worth,
            'ref_all_in': self.starting_net_worth / self.starting_price * next_open_price,
            'today': today['snapped_at'],
        }
        return close_state, reward, self.done, info
    
    def close(self):
        pass


if __name__ == "__main__":
    # args = Namespace(starting_date='2023-02-01', ending_date='2024-02-01')
    # args = Namespace(starting_date='2023-02-01', ending_date='2023-03-01')
    args = Namespace(starting_date='2024-01-01', ending_date='2024-02-01')
    env = ETHTradingEnv(args)
    ACTIONS = np.arange(-1, 1.1, 0.2).tolist()
    # ACTIONS = [1]
    state, reward, done, info = env.reset()
    print(f"init state: {state}, info: {info}, step: {env.current_step} \n")
    while not done:
        action = random.choice(ACTIONS)
        state, reward, done, info = env.step(action)
        print(f"Action: {action}")
        print_state = {k: v for k, v in state.items() if k != 'news'}
        # print_state['news'] = [{'id': item['id'], 'title': '.', 'summary': '.'} for item in state['news']]
        print(f"State: {print_state}, reward: {reward}, done: {done}, info: {info} \n")
    roi = state['roi']
    print(f"ROI: {roi*100:.2f}%")
