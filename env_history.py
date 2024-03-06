from typing import List, Dict


class EnvironmentHistory:
    def __init__(self, base_query: str, start_state, memory: List[str], history: List[Dict[str, str]], args) -> None:
        self.args = args
        self._cur_query: str = f'{_get_base_query(base_query, memory)}'
        self._history: List[Dict[str, str]] = history
        self.add('state', start_state)

    def add(self, label: str, value: str) -> None:
        assert label in ['action', 'state']
        self._history += [{
            'label': label,
            'value': value,
        }]

    def reset(self) -> None:
        self._history = []

    def __str__(self) -> str:
        use_news = self.args.use_news
        use_history = self.args.use_history
        use_mac = self.args.use_mac
        use_macd = self.args.use_macd
        use_txnstat = self.args.use_txnstat

        s = self._cur_query + '\n'
        if use_history:
            s += 'The action and state history is given in chronological order:\n'
        else:
            s += 'Your current state is:\n'
        for i, item in enumerate(self._history):
            if not use_history and i < len(self._history) - 1:  # if no history, only keep last state
                continue
            if item['label'] == 'action':
                s += f'Action: {item["value"]}' + '\n'
            elif item['label'] == 'state':
                state = item['value']
                state_hisotry = f'State: Cash: {state["cash"]:.2f}$, ETH held: {state["eth_held"]:.2f}, Open price: {state["open"]:.2f}'
                if use_txnstat:
                    state_hisotry += f', Last number of transactions: {state["num_txns"]}'
                if use_macd:
                    # state_hisotry += f', Technical Indicator MACD: {state["macd"]:.2f}, MACD Signal Line: {state["macd_signal_line"]:.2f}, MACD signal: {state["macd_signal"]}'
                    state_hisotry += f', Technical Indicator MACD signal: {state["macd_signal"]}'
                if use_mac:
                    # state_hisotry += f', Technical Indicator SMA5: {state["ma5"]:.2f}, SMA20: {state["ma20"]:.2f}, MA Crossover signal: {state["mac_5_20_signal"]}'
                    state_hisotry += f', Technical Indicator MA Crossover signal: {state["mac_5_20_signal"]}'
                s += state_hisotry + '\n'

        state = self._history[-1]['value']
        if use_news:
            s += f"Your news summaries: {state['news']}. "
        # s += f"Start your response with an action (a precision-two float value in range [-1,1]). Then, explain your reason behind it."
        s += f"Start your response with your reasoning over the given context to make an action. Then, give your action in the end in format <precision-two float value, [-1,0) to buy, (0,1] to sell, 0 to hold>"
        return s

def _get_base_query(base_query: str, memory: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
    query += f"\nHere is the task:\n"
    return query
