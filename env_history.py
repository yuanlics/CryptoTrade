from typing import List, Dict


class EnvironmentHistory:
    def __init__(self, base_query: str, start_state, memory: List[str], history: List[Dict[str, str]], args) -> None:
        self.args = args
        self._cur_query: str = f'{_get_base_query(base_query, memory)}'
        self._history: List[Dict[str, str]] = history  # prompt, action, state, ...
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
        price_window = self.args.price_window
        reflection_window = self.args.reflection_window
        use_tech = self.args.use_tech
        use_txnstat = self.args.use_txnstat

        price_s = 'You are an ETH cryptocurrency trading analyst. The recent price and auxiliary information is given in chronological order below:\n'
        for i, item in enumerate(self._history[-price_window * 3:]):
            if item['label'] == 'state':
                state = item['value']
                state_log = f'Open price: {state["open"]:.2f}'
                if use_txnstat:
                    txnstat_dict = state['txnstat']
                    for k, v in txnstat_dict.items():
                        state_log += f', {k}: {v}'
                if use_tech:
                    tech_dict = state['technical']
                    for k, v in tech_dict.items():
                        state_log += f', {k}: {v}'
                price_s += state_log + '\n'
        price_s += f'Analyze the recent information and estimate the market trend accordingly.'

        state = self._history[-1]['value']
        news_s = f"You are an ETH cryptocurrency trading analyst. You are required to analyze the following news: {state['news']}. Analyze the news and estimate the market trend accordingly."

        reflection_s = 'You are an ETH cryptocurrency trading analyst. Your analysis and action history is given in chronological order:\n'
        for i, item in enumerate(self._history[-reflection_window * 3:]):
            if item['label'] == 'prompt':
                reflection_s += f'PROMPT:\n{item["value"]}\n'
            elif item['label'] == 'action':
                reflection_s += f'ACTION:\n{item["value"]}\n'
            elif item['label'] == 'state':
                reflection_s += f'DAILY RETURN:\n{item["value"]["today_roi"]}\n'
        if len(self._history[-reflection_window:]) == 0:
            reflection_s += 'N/A.\n'
        # reflection_s += 'Reflect on your recent performance and instruct your future trades from a high level, e.g., identify what information is currently more important, and what to be next, like aggresive or conversative.'
        reflection_s += 'Reflect on your recent trading performance with a focus on the effective strategies and information that led to the most successful outcomes, and the ineffective strategies and information that led to loss of profit. Identify key trends and indicators in the current cryptocurrency market that are likely to influence future trades. Also assess whether a more aggressive or conservative trading approach is warranted. '

        base_prompt = 'You are an experienced ETH cryptocurrency trader and you are trying to maximize your overall profit by trading ETH. In each day, you will make an action to buy or sell ETH. You will start with 1 million dollars, half in cash and half in ETH. You are assisted by a few analysts below and need to decide the final action.\n'
        # 'A small transaction fee is charged based on the transaction amount, so frequent large transactions are penalized.'
        template_s = base_prompt + '\nON-CHAIN ANALYST REPORT: {}\n\n' + 'NEWS ANALYST REPORT: {}\n\n' + 'REFLECTION ANALYST REPORT: {}\n\n'
        template_s += 'Start your response with your reasoning over the given context. Wisely select the information, conclude a clear market trend, and avoid always being conservative. When the trend is upwards, it is profitable to buy ETH, and vice versa. Finally, suggest a 1-decimal float action in the range of [-1,1]. When the trend is bullish and upwards, you need to buy ETH with a negative value. When the trend is bearish and downwards, you need to sell ETH with a positive value. Your action is based on your prediction and confidence on the market trend, where a larger absolute value indicates a higher confidence on your trend prediction. Examples are: -0.3 (buy with 30% cash), 0.5 (sell 50% ETH), and so on. Make sure to directly give the value (could be negative) at the very end without any explanation.'

        return price_s, news_s, reflection_s, template_s

def _get_base_query(base_query: str, memory: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
    query += f"\nHere is the task:\n"
    return query
