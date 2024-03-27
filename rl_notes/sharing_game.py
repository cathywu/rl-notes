BROTHER = "B"
SISTER = "S"

from extensive_form_game import ExtensiveFormGame

class SharingGame(ExtensiveFormGame):

    ''' Get the list of players for this game as a list [1, ..., N] '''
    def get_players(self):
        return [BROTHER, SISTER]

    ''' Get the valid actions at a state '''
    def get_actions(self, state):
        actions = dict()
        actions[1] = ["2-0", "1-1", "0-2"]
        actions[2] = ["yes", "no"]
        actions[3] = ["yes", "no"]
        actions[4] = ["yes", "no"]

        if state in actions.keys():
            return actions[state]
        else:
            return []

    ''' Return the state resulting from playing an action in a state '''
    def get_transition(self, state, action):
        transitions = dict()
        if state == 1:
            transitions["2-0"] = 2
            transitions["1-1"] = 3
            transitions["0-2"] = 4
            return transitions[action]
        elif state > 1 and state <= 4:
            transitions["no"] = state * 2 + 1
            transitions["yes"] = state * 2 + 2
            return transitions[action]
        else:
            transitions[action] = []
        return transitions[action]

    ''' Return the reward for a state, return as a dictionary mapping players to rewards '''
    def get_reward(self, state):
        rewards = dict()
        if state > 4:
            rewards[5] = {BROTHER:0, SISTER:0}
            rewards[6] = {BROTHER:2, SISTER:0}
            rewards[7] = {BROTHER:0, SISTER:0}
            rewards[8] = {BROTHER:1, SISTER:1}
            rewards[9] = {BROTHER:0, SISTER:0}
            rewards[10] = {BROTHER:0, SISTER:2}
            return rewards[state]
        else:
            return {BROTHER:0, SISTER:0}

    ''' Return true if and only if state is a terminal state of this game '''
    def is_terminal(self, state):
        return state > 4

    ''' Return the player who selects the action at this state (whose turn it is) '''
    def get_player_turn(self, state):
        if state == 1:
            return BROTHER
        else:
            return SISTER
    
    ''' Return the initial state of this game '''
    def get_initial_state(self):
        return 1

    def to_string(self, state):
        return str(state)
