from extensive_form_game import ExtensiveFormGame

EMPTY = ' '
NOUGHT = 'o'
CROSS = 'x'

class TicTacToe(ExtensiveFormGame):



    ''' Initialise a TicTacToe game '''
    def __init__(self):
        self.players = [CROSS, NOUGHT]
    
    ''' Get the list of players for this game as a list [1, ..., N] '''
    def get_players(self):
        return self.players

    ''' Get the valie actions at a state '''
    def get_actions(self, state):

        #use a nicer variable name for this implementation
        board = state
        
        actions = []
        for x in range(len(board)):
            for y in range(len(board[x])):
                if board[x][y] == EMPTY:
                    actions += [(x,y)]
        return actions

    '''
        Deep copy this state
    '''
    def copy(self, state):
        next_state = []
        for x in range(len(state)):
            new_row = []
            for y in range(len(state[x])):
                 new_row += [state[x][y]]
            next_state += [new_row]
        return next_state

    ''' Return the state resulting from playing an action in a state '''
    def get_transition(self, state, action):
        next_state = self.copy(state)
        next_state[action[0]][action[1]] = self.get_player_turn(state)
        return next_state

    ''' Return the reward for a state '''
    def get_reward(self, state):
        winner = self.get_winner(state)
        if winner == None:
            return {CROSS:0, NOUGHT:0}
        elif winner == CROSS:
            return {CROSS:1, NOUGHT:-1}
        elif winner == NOUGHT:
            return {CROSS:-1, NOUGHT:1}
        
    def count_empty(self, board):
        empty = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                if board[x][y] == EMPTY:
                    empty += 1
        return empty
    
    ''' Return true if and only if state is a terminal state of this game '''
    def is_terminal(self, state):
        return self.count_empty(state) == 0 or self.get_winner(state) is not None


    ''' Return the player who selects the action at the current state (whose turn it is) '''
    def get_player_turn(self, state):

        #use a nicer variable name for this implementation
        board = state
        
        empty = self.count_empty(board)
        
        #crosses starts the game, so if there is an odd number of empty cells, it is crosses turn
        if empty % 2 == 0:
            return NOUGHT
        else:
            return CROSS
    
    ''' Return the initial state of this game '''
    def get_initial_state(self):
        board = [[EMPTY, EMPTY, EMPTY],
                 [EMPTY, EMPTY, EMPTY],
                 [EMPTY, EMPTY, EMPTY]]
        return board

    def get_winner(self, state):

        #use a nicer variable name for this implementation
        board = state

        #check columns
        for x in range(0, len(board)):
            noughts = 0
            crosses = 0
            for y in range(0, len(board[x])):
                if board[x][y] == NOUGHT:
                    noughts += 1
                elif board[x][y] == CROSS:
                    crosses += 1
            if noughts == len(board[0]):
                return NOUGHT
            elif crosses == len(board[0]):
                return CROSS

        #check rows
        for y in range(0, len(board[0])):
            noughts = 0
            crosses = 0
            for x in range(0, len(board)):
                if board[x][y] == NOUGHT:
                    noughts += 1
                elif board[x][y] == CROSS:
                    crosses += 1
            if noughts == len(board):
                return NOUGHT
            elif crosses == len(board):
                return CROSS

        #check top-left to bottom-right diagonal
        if board[0][0] == NOUGHT and board[1][1] == NOUGHT and board[2][2] == NOUGHT:
            return NOUGHT
        elif board[0][0] == CROSS and board[1][1] == CROSS and board[2][2] == CROSS:
            return CROSS

        #check bottom-left to top-right diagonal
        if board[0][2] == NOUGHT and board[1][1] == NOUGHT and board[2][0] == NOUGHT:
            return NOUGHT
        elif board[0][2] == CROSS and board[1][1] == CROSS and board[2][0] == CROSS:
            return CROSS

        #no winner
        return None
            

    def to_string(self, state):
        """
        Formats a board as a string replacing cell values with enum names.
        Args:
            board (numpy.ndarray): two dimensional array representing the board
                after the move
        Returns:
            str: the board represented as a string
        """
        # Join columns using '|' and rows using line-feeds
        result = str('\\n'.join(['|'.join([item for item in row]) for row in state]))
        return result
