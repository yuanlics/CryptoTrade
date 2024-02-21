from typing import List, Dict


class EnvironmentHistory:
    def __init__(self, base_query: str, start_info, memory: List[str], history: List[Dict[str, str]] = []) -> None:
        self._cur_query: str = f'{_get_base_query(base_query, start_info, memory)}'
        self._history: List[Dict[str, str]] = history

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
        s += 'Here is your action and state history (could be empty):\n'
        for i, item in enumerate(self._history):
            if item['label'] == 'action':
                s += f'Action > {item["value"]}' + '\n'
            elif item['label'] == 'state':
                state = item['value']
                state_str = f'Cash: {state["cash"]:.2f}, ETH held: {state["eth_held"]:.2f}, Close net worth: {state["close_net_worth"]:.2f}, Open price: {state["next_open_price"]:.2f}'
                s += state_str + '\n'
        s += 'Please respond in the specified format.\n'
        return s

def _get_base_query(base_query: str, start_info: str, memory: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
    query += f"\nHere is the task:\n{start_info}"
    return query
