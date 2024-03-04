import numpy as np
import pandas as pd
import re
import random
from datetime import datetime
import json

# PATH_PRICE = 'data/eth_202401.csv'
# PATH_PRICE = 'data/eth_202311.csv'
PATH_PRICE = 'data/eth_202309.csv'
DIR_NEWS  = 'data/gnews'
PRICE_TIME_FMT = "%Y-%m-%d %H:%M:%S UTC"
GAS_LIMITS = 21000  # unit
GAS_PRICE = 70  # gwei
GAS_FEE = GAS_LIMITS * GAS_PRICE * 1e-9  # eth per txn
EX_RATE = 4e-3  # exchange fee = txn_amount * ex_rate
# GAS_FEE = 0
# EX_RATE = 0


class ETHTradingEnv:
    def __init__(self):
        self.data = pd.read_csv(PATH_PRICE)
        self.total_steps = len(self.data)
        self.starting_cash = 1_000_000
        self.reset()

    def get_state(self, today, next_day, has_news=True):
        next_open_price = next_day['open']
        close_net_worth = self.cash + self.eth_held * next_open_price
        close_roi = close_net_worth / self.starting_cash - 1  # return on investment

        # technical state
        ma5 = next_day['SMA_5']
        ma20 = next_day['SMA_20']
        mac_signal = 'hold'  # MA Crossover
        if ma5 > ma20:
            mac_signal = 'sell'
        elif ma5 < ma20:
            mac_signal = 'buy'

        # news state
        date = today['snapped_at']
        parsed_time = datetime.strptime(date, PRICE_TIME_FMT)
        year, month, day = parsed_time.year, parsed_time.month, parsed_time.day
        news_path = f"{DIR_NEWS}/{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.json"
        news = json.load(open(news_path))
        if not has_news:
            news = ''  # no news for first action day

        close_state = {
            'cash': self.cash,
            'eth_held': self.eth_held,
            'open': next_open_price,
            'net_worth': close_net_worth,
            'roi': close_roi,
            'ma5': ma5,
            'ma20': ma20,
            'mac_signal': mac_signal,
            'news': news,
        }
        return close_state

    def reset(self):
        self.cash = self.starting_cash
        self.close_net_worth_history = [self.starting_cash]
        self.eth_held = 0
        self.current_step = 0  # start from the beginning
        # self.current_step = random.randint(0, self.total_steps)  # random starting time

        next_day = today = self.data.iloc[self.current_step]
        self.starting_price = today['open']
        close_state = self.get_state(today, next_day, has_news=False)
        info = {
            'starting_cash': self.starting_cash,
        }
        reward = 0
        self.done = False
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
            actions = re.findall(r"[-+]?\d*\.\d+|\d+", action)
            if len(actions) == 0:
                print(f'Invalid llm response: {action}. Set to no action.')
                action = 0.00
            elif len(actions) == 1:
                action = float(actions[0])
            else:
                # print(f'Multiple actions in llm response: {action}. Set to first action.')
                action = float(actions[0])
        
        if not -1 <= action <= 1:
            print(f"Invalid action: {action}. Set to no action.")
            action = 0.00

        today = self.data.iloc[self.current_step]
        next_day = self.data.iloc[self.current_step + 1]
        open_price = today['open']
        next_open_price = next_day['open']  # assume today's close = next day's open
        
        if -1 <= action < 0 and self.eth_held > 0:  # Sell ETH
            eth_diff = abs(action) * self.eth_held
            cash_diff = eth_diff * open_price
            self.eth_held -= eth_diff
            self.cash += cash_diff
            self.cash -= GAS_FEE * open_price + cash_diff * EX_RATE
        elif 0 < action <= 1 and self.cash > 0:  # Buy ETH
            cash_diff = action * self.cash
            eth_diff = cash_diff / open_price
            self.cash -= cash_diff
            self.eth_held += eth_diff
            self.cash -= GAS_FEE * open_price + cash_diff * EX_RATE
        
        close_net_worth = self.cash + self.eth_held * next_open_price
        close_roi = close_net_worth / self.starting_cash - 1  # return on investment
        self.close_net_worth_history.append(close_net_worth)
        
        self.current_step += 1
        if self.current_step >= self.total_steps - 1:
            self.done = True

        # technical state
        ma5 = next_day['SMA_5']
        ma20 = next_day['SMA_20']
        ma_signal = 'hold'
        if ma5 > ma20:
            ma_signal = 'sell'
        elif ma5 < ma20:
            ma_signal = 'buy'

        # news state
        date = today['snapped_at']
        parsed_time = datetime.strptime(date, PRICE_TIME_FMT)
        year, month, day = parsed_time.year, parsed_time.month, parsed_time.day
        news_path = f"{DIR_NEWS}/{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.json"
        news = json.load(open(news_path))
        
        close_state = {
            'cash': self.cash,
            'eth_held': self.eth_held,
            'open': next_open_price,
            'net_worth': close_net_worth,
            'roi': close_roi,
            'ma5': ma5,
            'ma20': ma20,
            'mac_signal': ma_signal,
            'news': news,
        }
        reward = self.close_net_worth_history[-1] - self.close_net_worth_history[-2]  # reward = today's profit.
        info = {
            'raw_action': raw_action,
            'actual_action': action,
            'starting_cash': self.starting_cash,
            'ref_all_in': self.starting_cash / self.starting_price * next_open_price,
            'today': today['snapped_at'],
        }
        return close_state, reward, self.done, info
    
    def close(self):
        pass


if __name__ == "__main__":
    env = ETHTradingEnv()
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
