class ExtensiveFormGame:

    ''' Get the list of players for this game as a list [1, ..., N] '''
    def get_players(self): abstract

    ''' Get the valid actions at a state '''
    def get_actions(self, state): abstract

    ''' Return the state resulting from playing an action in a state '''
    def get_transition(self, state, action): abstract

    ''' Return the reward for a state, return as a dictionary mapping players to rewards '''
    def get_reward(self, state, action, next_state): abstract

    ''' Return true if and only if state is a terminal state of this game '''
    def is_terminal(self, state): abstract

    ''' Return the player who selects the action at this state (whose turn it is) '''
    def get_player_turn(self, state): abstract

    ''' Return the initial state of this game '''
    def get_initial_state(self): abstract

    ''' Return a game tree for this game '''
    def game_tree(self):
        return self.state_to_node(self.get_initial_state())

    def state_to_node(self, state):
        if self.is_terminal(state):
            node = GameNode(state, None, self.get_reward(state))
            return node

        player = self.get_player_turn(state)
        children = dict()
        for action in self.get_actions(state):
            next_state = self.get_transition(state, action)
            child = self.state_to_node(next_state)
            children[action] = child
        node = GameNode(state, player, None, children = children)
        return node

class GameNode:

    # record a unique node id to distinguish duplicated states
    next_node_id = 0

    def __init__(self, state, player_turn, value, is_best_action = False, children = dict()):
        self.state = state
        self.player_turn = player_turn
        self.value = value
        self.is_best_action = is_best_action
        self.children = children

        self.id = GameNode.next_node_id
        GameNode.next_node_id += 1
