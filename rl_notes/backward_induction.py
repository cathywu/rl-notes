from extensive_form_game import GameNode

class BackwardInduction:
    def __init__(self, game, do_cache = False):
        self.game = game
        self.do_cache = do_cache
        self.cache = dict()

    def backward_induction(self, state):

        if self.game.is_terminal(state):
            node = GameNode(state, None, self.game.get_reward(state))
            return node

        best_child = None
        best_action = None
        player = self.game.get_player_turn(state)
        children = dict()
        for action in self.game.get_actions(state):
            next_state = self.game.get_transition(state, action)
            child = self.backward_induction(next_state)
            if best_child is None or child.value[player] > best_child.value[player]:
                if best_child is not None:
                    best_child.is_best_action = False
                child.is_best_action = True
                best_child = child
            children[action] = child
        node = GameNode(state, player, best_child.value, children = children)
        return node

    def backward_induction_with_cache(self, state):

        state_key = self.game.to_string(state)
        if self.do_cache and state_key in self.cache.keys():
            return self.cache[state_key]

        if self.game.is_terminal(state):
            node = GameNode(state, None, self.game.get_reward(state))
            if self.do_cache:
                self.cache[state_key] = node
            return node

        best_child = None
        best_action = None
        player = self.game.get_player_turn(state)
        children = dict()
        for action in self.game.get_actions(state):
            next_state = self.game.get_transition(state, action)
            child = self.backward_induction(next_state)
            if best_child is None or child.value[player] > best_child.value[player]:
                if best_child is not None:
                    best_child.is_best_action = False
                child.is_best_action = True
                best_child = child
            children[action] = child
        node = GameNode(state, player, best_child.value, children = children)
        if self.do_cache:
            self.cache[state_key] = node
        return node
