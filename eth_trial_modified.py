"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import yaml
import openai
import importlib
from utils import Model, get_chat
# from eth_env import ETHTradingEnv
from eth_env_modified import ETHTradingEnv
from env_history import EnvironmentHistory

from typing import List, Dict, Any, Tuple
 
FOLDER = './prompts'

def llm(prompt: str, model: Model, stop: List[str] = ["\n"]):
    try:
        text = get_chat(prompt=prompt, model=model, temperature=0.0, stop_strs=stop)
        return text
    except Exception as e:
        print(prompt)
        print(e)
        import sys
        sys.exit(1)

def eth_run(env, base_prompt, memory: List[str], max_steps=50, to_print=True, state_query='', model: Model = "gpt-3.5-turbo") -> Tuple[EnvironmentHistory, bool]:
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, state_query, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, state_query, memory, [])
    env_history.reset()
    if to_print:
        print(state_query)
        sys.stdout.flush()
    cur_step = 0
    while cur_step < max_steps:
        prompt = str(env_history) + ">"
        action = llm(prompt, stop=['\n'], model=model).strip()
        state, reward, done, info = env.step(action)
        action = info['actual_action']
        action = f'{action:.2f}'
        env_history.add("action", action)
        env_history.add("state", state)
        print_state = {k: v for k, v in state.items() if k != 'today_news'}
        if to_print:
            print(f'> {action}\n{print_state}')
            sys.stdout.flush()
        if done:
            break
        cur_step += 1
    is_success = (state['close_net_worth'] - info['starting_cash'])/info['starting_cash'] > 0.2 # modify sucess condition
    return env_history, is_success

def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model: Model,
        max_steps: int = 50
    ) -> List[Dict[str, Any]]:

    env = ETHTradingEnv()

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs):
        state, reward, done, info = env.reset()
        # print(state)
        state_query = f"Your current cash is {state['cash']:.2f} dollars and holding {state['eth_held']:.2f} ETH. The ETH open price is {state['open_price']:.2f} dollars. Related news is summarized in {state['news']}. What's your action? (precision-two float-value in range [-1,1])."

        if env_config["is_success"]:
            num_successes += 1

            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        base_prompt = f"You are now stepping into the shoes of a seasoned Ether (ETH) trader, embarking on a virtual trading challenge. Your goal is straightforward but challenging: maximize your profits from trading ETH over a one-month simulation period, starting from January 1, 2024, to January 31, 2024. This simulation is designed to test your analytical skills, market intuition, and strategic decision-making. Here’s what you need to know:\
1. **Starting Capital:** You begin with a $1 million cash reserve. Your mission is to grow this initial investment by wisely trading Ether. Success is measured by the total value of your cash and ETH holdings at the end of the simulation.\
2. **Daily Market Data & News:** Every day, you will receive vital market data including ETH's price movements, market cap, and total volume. Additionally, a curated list of news summaries will provide insights into the crypto market's sentiment and potential ETH price drivers.\
3. **Transaction Fees:** Each buy or sell transaction incurs a fee, calculated as a percentage of the transaction value. This simulates real-world trading conditions where fees can impact your trading strategy and profitability.\
**Decision-Making Criteria:**\
- **Analyzing Market Data with a Nuanced Approach:**\
  - Explore various strategies, including the moving averages MA5 and MA20. A buy signal is indicated when MA5 crosses below MA20 and subsequently begins to rise, suggesting an upward trend in ETH’s price. Conversely, a sell signal is suggested when MA5 crosses above MA20 and starts to decline, indicating a potential price drop. It's crucial to utilize these signals as part of a broader analysis, combining them with other market indicators for a well-rounded decision.\
  - Instead of defaulting to common responses, you're encouraged to analyze the market data deeply and consider the full spectrum of investment actions. For instance, subtle market movements might warrant precise adjustments in your position, such as investments or sales representing 15%, 35%, or even 85% of your assets, reflecting a granular approach to trading based on nuanced market interpretations.\
  - An upward trend in ETH’s price not only suggests a buying opportunity but calls for a tailored decision on how much to invest. Rather than broad strokes like 50% or 100%, consider the exactitude of your confidence in the trend—might 22%, 47%, or 76% of your cash reserve be more appropriate? Similarly, in a downward trend, decide on the precise portion of your ETH holdings to sell—could 18%, 33%, or 88% better reflect your strategic outlook and risk assessment?\
- **Interpreting News:**\
  - Positive news could bolster confidence in ETH, suggesting a buy signal. The extent of your investment should reflect the perceived impact of the news.\
  - Negative news might raise red flags, hinting at a sell action. The proportion of ETH you decide to sell should align with the news' potential negative impact on ETH’s price.\
- **Neutral Signals:** In cases of mixed signals or uncertainty, maintaining your position (hold) or making minor adjustments might be prudent.\
**Your Daily Decision:**\
Every day, you will decide whether to 'buy', 'sell', or 'hold' based on your analysis. Your decision must be quantified as follows:\
- **To Buy:** Specify a positive decimal fraction of your remaining cash to invest in ETH, reflecting your confidence and strategy (e.g., 0.25 for 25%).\
- **To Sell:** Indicate a negative decimal fraction of your ETH holdings you wish to sell, capturing your strategic decision (e.g., -0.40 for 40%).\
- **To Hold:** A decision to hold requires no action but is an active strategy based on your market analysis.\
**Precision in Decision:** Ensure your decision is presented as a two-decimal value within the [-1, 1] range. This precision reflects the nuanced analysis and strategic thought behind your decision."

        final_env_history, is_success = eth_run(env, base_prompt, env_config["memory"] if use_memory else [], to_print=True, state_query=state_query, model=model, max_steps=max_steps)

        # update env config
        if is_success:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
            env_configs[z]['is_success'] = True
            num_successes += 1
            num_additional_successes += 1
        else:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

        # log to world log
        with open(world_log_path, 'a') as f:
            f.write(status_str + '\n')

        # log env results to trial log
        with open(trial_log_path, 'a') as wf:
            wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

    # close environment object
    env.close()

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs
