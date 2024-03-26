from typing import List, Dict


class EnvironmentHistory:
    def __init__(self, base_query: str, start_state, memory: List[str], history: List[Dict[str, str]], args) -> None:
        self.args = args
        self._cur_query: str = f'{_get_base_query(base_query, memory)}'
        self._history: List[Dict[str, str]] = history
        self.prompt_log = []
        self.add('state', start_state)

    def add(self, label, value) -> None:
        self._history += [{
            'label': label,
            'value': value,
        }]

    def reset(self) -> None:
        self._history = []
    
    def get_prompt(self) -> str:
        window = self.args.window  # look back window for prompt
        use_tech = self.args.use_tech
        use_txnstat = self.args.use_txnstat

        state = self._history[-1]['value']
        news_s = f"You are a cryptocurrency trading analyst. Your news summaries: {state['news']}. Analyze the news and estimate the market trend accordingly."

        price_s = 'You are a cryptocurrency trading analyst. The recent price and auxiliary information is given in chronological order below:\n'
        for i, item in enumerate(self._history[-window:]):
            if item['label'] == 'state':
                state = item['value']
                state_log = f'Open price: {state["open"]:.2f}'
                if use_txnstat:
                    state_log += f', number of transactions: {state["num_txns"]}'
                if use_tech:
                    # state_log += f', MACD: {state["macd"]:.2f}, MACD Signal Line: {state["macd_signal_line"]:.2f}, MACD signal: {state["macd_signal"]}'
                    state_log += f', MACD signal: {state["macd_signal"]}'
                    # state_log += f', SMA5: {state["ma5"]:.2f}, SMA20: {state["ma20"]:.2f}, MA Crossover signal: {state["mac_5_20_signal"]}'
                    state_log += f', Short-Long MA signal: {state["mac_5_20_signal"]}'
                price_s += state_log + '\n'
        price_s += f'Analyze the recent information and estimate the market trend accordingly.'

        reflection_s = 'You are a cryptocurrency trading analyst. Your analysis and action history is given in chronological order:\n'
        for i, item in enumerate(self._history[-window:]):
            if item['label'] == 'prompt':
                reflection_s += f'Prompt: {item["value"]}\n'
            elif item['label'] == 'action':
                reflection_s += f'Action: {item["value"]}\n'
            elif item['label'] == 'state':
                reflection_s += f'State: {item["value"]}\n'
        if len(self._history[-window:]) == 0:
            reflection_s += 'N/A.\n'
        reflection_s += f'Reflect on your recent performance and give advice on your future decision making, e.g., identify what information is currently more important.'

        base_prompt = 'You are an experienced crypto currency trader and you are trying to maximize your overall profit by trading ETH. In each day, you will make an action represented by a precision-one float value in range [-1,1]. If action is in negative range [-1,0), you will spend this portion of cash at hand to buy ETH. If action is in positive range (0,1], you will sell this portion of holding ETH at hand. If action is 0, no action is taken and no transaction fee is charged. Examples are -0.3 (buy with 30% cash), 0.8 (sell 80% ETH), -1.0 (buy with 100% cash), 1.0 (sell 100% ETH), 0.0 (no action), and so on. You will start with 1 million dollars, half in cash and half in ETH. You are assisted by a few analysts and need to decide the final action\n'
        # 'A small transaction fee is charged based on the transaction amount, so frequent large transactions are penalized.'
        template_s = base_prompt + 'News analyst report: {}\n' + 'On-chain analyst report: {}\n' + 'Reflection analyst report: {}\n'
        template_s += 'Start your response with your reasoning over the given context. Wisely select the information, conclude a clear market trend, and avoid being too conservative. When the trend is upwards, it is profitable to buy ETH, and sell ETH when the trend is downwards. Finally, suggest a 1-decimal float action in the range of [-1,1]. Remember to give a negative action to buy ETH when the trend is bullish and upwards, positive action to sell ETH when the trend is bearish and downwards, and 0 to hold. Your action is based on your prediction and confidence on the market trend, where a larger absolute value indicates a higher confidence on your trend prediction. Make sure to directly give the value at the very end without any explanation.'

        return news_s, price_s, reflection_s, template_s

def _get_base_query(base_query: str, memory: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
    query += f"\nHere is the task:\n"
    return query
