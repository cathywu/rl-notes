from collections import defaultdict
from value_function import ValueFunction

class TabularValueFunction(ValueFunction):
    def __init__(self, default=0.0):
        self.value_table = defaultdict(lambda: default)

    def update(self, state, value):
        self.value_table[state] = value

    def merge(self, value_table):
        for state in value_table.value_table.keys():
            self.update(state, value_table.get_value(state))

    def get_value(self, state):
        return self.value_table[state]

