import numpy as np
import pandas as pd
import re
import random

# PATH = 'data/eth_daily.csv'
PATH = 'data/eth_202401.csv'


# 2024-01
class ETHTradingEnv:
    def __init__(self, file_path=PATH):
        self.data = pd.read_csv(file_path)
        self.total_steps = len(self.data)
        self.starting_cash = 10000
        self.reset()

    def reset(self):
        self.cash = self.starting_cash
        self.close_net_worth_history = [self.starting_cash]
        self.eth_held = 0
        self.current_step = 0  # start from the beginning
        # self.current_step = random.randint(0, self.total_steps)  # random starting time
        self.starting_price = self.data.iloc[self.current_step]['open']
        self.done = False
        day = self.data.iloc[self.current_step]
        open_price = day['open']

        state = {
            'cash': self.cash,
            'eth_held': self.eth_held,
            'open_price': open_price,
        }
        net_worth = self.cash + self.eth_held * open_price
        info = {
            'net_worth': net_worth,
            'date': day['snapped_at'],
        }
        done = False
        reward = 0
        return state, reward, done, info

    def step(self, action):  # act for today
        raw_action = action
        if type(action) == str:
            actions = re.findall(r"[-+]?\d*\.\d+|\d+", action)
            if len(actions) == 0:
                print(f'Invalid llm response: {action}. Default to no action.')
                action = 0.00
            elif len(actions) == 1:
                action = float(actions[0])
            else:
                print(f'Multiple actions in llm response: {action}. Default to first action.')
                action = float(actions[0])
        
        if not -1 <= action <= 1:
            print(f"Invalid action: {action}. Default to no action.")
            action = 0.00

        today = self.data.iloc[self.current_step]
        next_day = self.data.iloc[self.current_step + 1]
        open_price = today['open']
        close_price = today['close'] if 'close' in today else next_day['open']
        next_open_price = next_day['open']
        
        if -1 <= action < 0 and self.eth_held > 0:  # Sell ETH
            eth_to_sell = abs(action) * self.eth_held
            self.cash += eth_to_sell * open_price
            self.eth_held -= eth_to_sell
        elif 0 < action <= 1 and self.cash > 0:  # Buy ETH
            cash_to_spend = action * self.cash
            self.eth_held += cash_to_spend / open_price
            self.cash -= cash_to_spend
        
        close_net_worth = self.cash + self.eth_held * close_price
        self.close_net_worth_history.append(close_net_worth)
        
        self.current_step += 1
        if self.current_step >= self.total_steps - 1:
            self.done = True
        
        close_state = {
            'cash': self.cash,
            'eth_held': self.eth_held,
            'next_open_price': next_open_price,
            'close_net_worth': close_net_worth,
        }
        reward = self.close_net_worth_history[-1] - self.close_net_worth_history[-2]  # reward = today's profit.
        info = {
            'raw_action': raw_action,
            'actual_action': action,
            'starting_cash': self.starting_cash,
            'ref_all_in': self.starting_cash / self.starting_price * close_price,
            'date': today['snapped_at'],
        }
        return close_state, reward, self.done, info
    
    def close(self):
        pass


if __name__ == "__main__":
    # the agent receives last state and reward, takes an action, and receives new state and reward.
    # state = observation
    env = ETHTradingEnv()
    ACTIONS = np.arange(-1, 1.1, 0.2).tolist()
    # ACTIONS = [1]
    state, reward, done, info = env.reset()
    print(f"init state: {state}, info: {info}, step: {env.current_step}")
    while not done:
        action = random.choice(ACTIONS)
        state, reward, done, info = env.step(action)
        print(f"action: {action} -> state: {state}, reward: {reward}, done: {done}, info: {info}")
    profit = state['close_net_worth'] - info['starting_cash']
    print(f"final profit: {profit}")
