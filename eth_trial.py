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
        if to_print:
            print(f'> {action}\n{state}')
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
        state, info = env.reset()
        state_query = f"Your current cash is {state['cash']:.2f} dollars and holding {state['eth_held']:.2f} ETH. The ETH open price is {state['open_price']:.2f} dollars. What's your action? (precision-two float-value in range [-1,1])"

        if env_config["is_success"]:
            num_successes += 1

            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        base_prompt = 'Play an ETH trading game to maximize your overall profit. At each step, directly return an action in precision-two float-value in range [-1,1] and nothing more. If action in range [-1,0], you will sell the portion of holding ETH at hand. If action in range [0,1], you will spend the portion of cash at hand to buy ETH. If action is 0, no action is taken. You start with 10000 dollars. \n'
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
