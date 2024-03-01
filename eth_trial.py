"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import yaml
import openai
import importlib
from utils import Model, get_chat
from eth_env import ETHTradingEnv
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
    is_success = state['close_net_worth'] > info['starting_cash']
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

#         base_prompt = f"Imagine you're a skilled cryptocurrency trader focusing on Ether. Your goal is to engage in an ETH trading simulation to maximize your profits. Here’s how you play:\
# 1. Starting with a $1 million cash reserve, your objective is to increase your wealth through strategic buying and selling of Ether. The challenge spans from January 1, 2024, to January 31, 2024. Your success is measured by the cash and Ether value you hold at the end of this period.\
# 2. Each trading day brings fresh market data, including Ether's price, market cap, and total volume, along with summarized news relevant to Ether's ecosystem.\
# 3. Remember, each transaction incurs a fee proportional to the transaction value. Your asset's value is dynamic, reflecting the current market price.\
# 4. When interpreting news:\
#    - Strong positive indicators suggest a significant buy opportunity.\
#    - Mild positive signals hint at a moderate buying chance.\
#    - Strong negative indicators warrant a considerable sell.\
#    - Mild negative signals suggest a cautious sell approach.\
#    - Neutral information indicates a hold decision.\
# 5. Market trends also guide your actions:\
#    - A confident upward trend suggests a substantial buy.\
#    - A slight upward trend indicates a minor buy opportunity.\
#    - A confident downward trend calls for a significant sell.\
#    - A slight downward trend suggests a moderate sell.\
#    - Stability suggests holding your position.\
# 6. Combine insights from both market data and news to make informed decisions. The degree of action should reflect the strength and confidence in the signals you interpret.\
# 7. Decision making:\
#    - For nuanced market analysis or mixed signals, consider incremental adjustments, like utilizing 10%, 20%, 30%, etc., of your available resources for buying or selling.\
#    - In cases of uncertainty or mild signals from both market and news, moderate actions such as 40% or 60% could be appropriate.\
# 8. Your daily decision can be to 'buy', 'sell', or 'hold'. Reflect your choice as a precise action:\
#    - To buy, specify a positive decimal fraction of your cash to use, up to 1, to indicate the subtlety of your decision-making.\
#    - To sell, specify a negative decimal fraction of your Ether holdings to liquidate, down to -1, to represent your strategic sell decisions.\
#    - Aim for precision with your action, considering a full range of values between -1 and 1, to accurately reflect your strategy based on the analysis.\
# Your decisions should be finely tuned, reflecting a careful analysis of both the numerical market trends and the nuanced implications of the latest news. Let your strategies be as dynamic and responsive as the market itself. \
# Encourage a spectrum of actions based on the depth of your analysis rather than sticking to broad strokes."


        base_prompt = f"You are now stepping into the shoes of a seasoned Ether (ETH) trader, embarking on a virtual trading challenge. Your goal is straightforward but challenging: maximize your profits from trading ETH over a one-month simulation period, starting from January 1, 2024, to January 31, 2024. This simulation is designed to test your analytical skills, market intuition, and strategic decision-making. Here’s what you need to know:\
1. **Starting Capital:** You begin with a $1 million cash reserve. Your mission is to grow this initial investment by wisely trading Ether. Success is measured by the total value of your cash and ETH holdings at the end of the simulation.\
2. **Daily Market Data & News:** Every day, you will receive vital market data including ETH's price movements, market cap, and total volume. Additionally, a curated list of news summaries will provide insights into the crypto market's sentiment and potential ETH price drivers.\
3. **Transaction Fees:** Each buy or sell transaction incurs a fee, calculated as a percentage of the transaction value. This simulates real-world trading conditions where fees can impact your trading strategy and profitability.\
**Decision-Making Criteria:**\
- **Analyzing Market Data:**\
  - An upward trend in ETH’s price may indicate a buying opportunity. The decision to buy could range from a cautious investment (e.g., using 10% of your cash reserve) to aggressive buying (e.g., investing 95%).\
  - A downward trend suggests a potential sell signal. You might decide to liquidate a minor portion of your ETH holdings (e.g., 20%) or opt for a major sell-off (e.g., 90%), depending on your market outlook.\
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
