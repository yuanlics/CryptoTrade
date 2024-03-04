from typing import List, Dict


class EnvironmentHistory:
    def __init__(self, base_query: str, start_state, memory: List[str], history: List[Dict[str, str]] = []) -> None:
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
        s: str = self._cur_query + '\n'
        s += 'You are given your action and state history below for reference:\n'
        for i, item in enumerate(self._history):
            if item['label'] == 'action':
                s += f'Action: {item["value"]}' + '\n'
            elif item['label'] == 'state':
                state = item['value']
                state_hisotry = f'\
Cash: {state["cash"]:.2f}$, \
ETH held: {state["eth_held"]:.2f}, \
ROI: {state["roi"]*100:.2f}%, \
Open price: {state["open"]:.2f}, \
MA5: {state["ma5"]:.2f}, \
MA20: {state["ma20"]:.2f}, \
MA Crossover signal: {state["mac_signal"]}'
                s += state_hisotry + '\n'
        state = self._history[-1]['value']

        s += f"\Your current state is {state_hisotry}, \
MA Crossover signal is {state['mac_signal']}, \
News summaries: {state['news']}. \
Start your response with an action (a precision-two float value in range [-1,1]). Then, explain your reason behind it."
# Gas fee: {state['gas_fee']//1e-9} gwei, \
# Exchange commision rate: {state['ex_rate']*100:.2f}%, \
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
