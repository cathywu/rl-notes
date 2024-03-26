from collections import defaultdict

from mdp import *
from rendering_utils import *
import numpy as np
import warnings

SUNK_REWARD = -10
BERTH_REWARD = 10

#six directions for movement - W, NW, NE, E, SE, SW - and can shoot
W = 0
NW = 1
NE = 2
E = 3
SE = 4
SW = 5
SHOOT = 6

#region types
SHORE1 = 10
SHORE2 = 11
LAND = 12
SEA = 13
#danger zones
HI_DGR = 3
LO_DGR = 4
NO_DGR = 5

DEFAULT_REGIONS = {(2,0):SHORE2,(3,0):LAND,(4,0):LAND,(5,0):LAND,(2,1):SHORE2,(3,1):LAND,(4,1):LAND,(3,2):SHORE2,
                   (4,2):LAND,(5,2):LAND,(3,3):SHORE2,(4,3):SHORE2,(3,5):SHORE1,(4,5):SHORE1,(3,6):SHORE1,(4,6):LAND,
                   (5,6):LAND,(2,7):SHORE1,(3,7):LAND,(4,7):LAND,(2,8):SHORE1,(3,8):LAND,(4,8):LAND,(5,8):LAND,
                   (1,9):SHORE1,(2,9):LAND,(3,9):LAND,(4,9):LAND,(5,9):LAND}

DEFAULT_DANGER =  {(1,1):LO_DGR,(2,1):LO_DGR,(1,2):LO_DGR,(2,2):LO_DGR,(3,2):HI_DGR,
                   (1,3):LO_DGR,(2,3):HI_DGR,(3,3):HI_DGR,(4,3):HI_DGR,(1,4):LO_DGR,(2,4):HI_DGR,
                   (3,4):HI_DGR,(4,4):HI_DGR,(1,5):LO_DGR,(2,5):HI_DGR,(3,5):HI_DGR,(4,5):HI_DGR,(1,6):LO_DGR,
                   (2,6):LO_DGR,(3,6):HI_DGR,(1,7):LO_DGR,(2,7):LO_DGR,(2,8):LO_DGR}
DEFAULT_WIDTH = 5
DEFAULT_HEIGHT = 10
DEFAULT_BATTERY = (3,3)
DEFAULT_SHIP = (2,8)
REGION_COLOURS = {SHORE1:'#876445',SHORE2:'#876445',LAND:'#EDDFB3',SEA:'#EFEFEF',HI_DGR:'#ff0000',LO_DGR:'#ff8800',NO_DGR:'#FFFFFF',-1:'#FFFFFF',}
SHIP_COLOR = "#4466aa"
BATTERY_COLOR = "#441111"
SHIP_SYMBOL = "\u03e1" 
BATTERY_SYMBOL = "\u273A"


left_cell = lambda o: (o[0]-1,o[1])
right_cell = lambda o: (o[0]+1,o[1])
down_left_cell = lambda o: (o[0]+(o[1]%2)-1,o[1]+1)
down_right_cell = lambda o: (o[0]+(o[1]%2),o[1]+1)        
up_left_cell = lambda o: (o[0]+(o[1]%2)-1,o[1]-1)
up_right_cell = lambda o: (o[0]+(o[1]%2),o[1]-1)        

warnings.filterwarnings('error')

class ContestedCrossing(MDP):
    #labels for actions and states
    TERMINATE = 'terminate'
    TERMINAL = ('terminal','terminal','terminal','terminal','terminal')
    
    #state of the single-agent game is given by 5 integers:
    #xpos, ypos, ship_damage, battery1_damage, battery2_damage
    
    def __init__(
        self,
        high_danger=0.9,
        low_danger=0.1,
        battery_health = 2,
        ship_health = 3,
        discount_factor=0.9,
        action_cost=0.0,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        regions=DEFAULT_REGIONS,
        danger=DEFAULT_DANGER,
        battery=DEFAULT_BATTERY,
        ship=DEFAULT_SHIP
    ):
        self.width = width
        self.height = height
        self.regions = defaultdict(lambda:SEA,regions)
        self.terrain = self._make_terrain(self.regions)
        self.danger_zones = defaultdict(lambda:NO_DGR,danger)
        self.battery = battery
        self.ship = ship
        self.high_danger = high_danger
        self.low_danger = low_danger
        self.battery_full_health = battery_health
        self.ship_full_health = ship_health
        self.battery_health = battery_health
        self.ship_health = ship_health
        self.discount_factor = discount_factor
        self.action_cost = action_cost
        self.initial_state=(self.ship[0],self.ship[1],ship_health,battery_health,W)
        self.blocks = self._make_block_dict(self.regions)
        self.goal_states = self._make_goals(self.regions)

        # A list of numpy arrays that records all rewards given at each step
        # for each episode of a simulated world
        self.rewards = [] 
        # The rewards for the current episode
        self.episode_rewards = []

    ### HELPER FUNCTIONS ###
    def _make_block_dict(self, regions):
        """
        route is blocked if both ends are land or shore - this works as long as we ensure
        the two shores are more than 1 step apart. Disallows creeping along the beach
        """
        blocks = defaultdict(lambda: False)
        for x in range(self.width):
            for y in range(self.height):
                if regions[(x,y)] in [LAND, SHORE1,SHORE2]:
                    if regions[(x+1,y)] in [LAND, SHORE1,SHORE2]:
                        blocks[(x,y,E)]=True
                        blocks[(x+1,y,W)]=True
                    if regions[(x+(y%2),y-1)] in [LAND, SHORE1,SHORE2]:
                        blocks[(x,y,NE)]=True
                        blocks[(x+(y%2),y-1,SW)]=True
                    if regions[(x+(y%2),y+1)] in [LAND, SHORE1, SHORE2]:
                        blocks[(x,y,SE)]=True
                        blocks[(x+(y%2),y+1,NW)]=True
        for x in range(self.width):
            blocks[(x,0,NW)]=blocks[(x,0,NE)]=True
            blocks[(x,self.height-1,SE)]=blocks[(x,self.height-1,SW)]=True
        for y in range(self.height):
            blocks[(0,y,W)]=blocks[(self.width-1,y,E)]=True
            if (y%2==0):
                blocks[(0,y,SW)]=blocks[(0,y,NW)]=True
            else:
                blocks[(self.width-1,y,SE)]=blocks[(self.width-1,y,NE)]=True
        return blocks

    def _make_terrain(self, regions):
        """
        identify and save coordinates of faces and edges. Coordinates are:
         - x,y,0 for an upward triangle with top point at x,y
         - x,y,1 for a downward triangle with top left at x,y
         - x,y,SW for a line down-left of x,y
         - x,y,SE for a line down-right of x,y
         - x,y,E for a line right of x,y
         Yes this means we have to be careful never to redefine 'E','SW' or 'SE' as 0 or 1
         """
        terrain = defaultdict(lambda:SEA)
        for x in range (self.width):
            for y in range (self.height-1):
                if regions[(x,y)] in [LAND, SHORE1, SHORE2]:
                    if regions[down_left_cell((x,y))] in [LAND, SHORE1, SHORE2] and regions[down_right_cell((x,y))] in [LAND, SHORE1, SHORE2]:
                        terrain[(x,y,0)]=LAND
                    if regions[right_cell((x,y))] in [LAND, SHORE1, SHORE2] and regions[down_right_cell((x,y))] in [LAND, SHORE1, SHORE2]:
                        terrain[(x,y,1)]=LAND
                    if regions[(x,y)] in [SHORE1, SHORE2] and regions[down_left_cell((x,y))] == regions[(x,y)]:
                        terrain[(x,y,SW)] = regions[(x,y)]
                    if regions[(x,y)] in [SHORE1, SHORE2] and regions[down_right_cell((x,y))] == regions[(x,y)]:
                        terrain[(x,y,SE)] = regions[(x,y)]
                    if regions[(x,y)] in [SHORE1, SHORE2] and regions[right_cell((x,y))] == regions[(x,y)]:
                        terrain[(x,y,E)] = regions[(x,y)]
        return terrain
                        
    def _direction(self, point1,point2):
        """
        backform action direction from knowing start and end points
        """
        x1,y1=point1
        x2,y2=point2
        if x2==x1+1 and y2==y1:
            return E
        if x2==x1-1 and y2==y1:
            return W
        if x2==x1+(y1%2) and y2==y1+1:
            return SE
        if x2==x1+(y1%2) and y2==y1-1:
            return NE
        if x2==x1-1+(y1%2) and y2==y1+1:
            return SW
        if x2==x1-1+(y1%2) and y2==y1-1:
            return NW
        return -1
        
    def _move(self, point,direction):
        """
        figure out end point from direction
        """
        x,y=point
        if self.blocks[(x,y,direction)]:
            x,y = point
        elif direction == E:
            x,y = right_cell(point)
        elif direction == W:
            x,y = left_cell(point)
        elif direction == SE:
            x,y = down_right_cell(point)
        elif direction == NE:
            x,y = up_right_cell(point)
        elif direction == SW:
            x,y = down_left_cell(point)
        elif direction == NW:
            x,y = up_left_cell(point)

        return max(0,min(x,self.width+1)),max(0,min(y,self.height+1))
        

    def _make_goals(self, regions):
        states={}
        for x in range(self.width):
            for y in range(self.height):
                #every state with ship at shore2 is a 'shore' goal, adjusted for damage
                if self.regions[(x,y)]==SHORE2:
                    for bh in range(self.battery_full_health+1):
                        for sh in range(self.ship_full_health+1):
                            for d in range(6):
                                states[(x,y,sh,bh,d)]=BERTH_REWARD + self.ship_full_health - self.ship_health
                #every state with sunk ship offshore is a 'sunk' goal
                elif self.regions[(x,y)]!=LAND:
                    for bh in range(self.battery_full_health+1):
                        for d in range(6):
                            states[(x,y,0,bh,d)]=SUNK_REWARD
        return states

    def _blocked(self, point, direct):
        return self.blocks[(point[0],point[1],direct)]

    ### MAIN FUNCTIONS ###      
    def get_states(self):
        """
        get all game states - 2 positional (x and y), 3 environmental
        (ship health, battery health, ship direction)
        """
        states = [self.TERMINAL]
        for x in range(self.width):
            for y in range(self.height):
                if self.regions[(x,y)]!=LAND:
                    for sh in range (self.ship_full_health+1):
                        for bh in range (self.battery_full_health+1):
                            for d in range(6):
                                states.append((x,y,sh,bh,d))
        return states

    def get_actions(self, state=None):
        """
        get all actions allowable from this state
        """
        actions = [W, NW, NE, E, SE, SW, SHOOT, self.TERMINATE]
        if state is None:
            return actions
        
        x,y,sh,bh,d = state
        if sh==0 or self.regions[(x,y)] == SHORE2:
            #sunk or reached the shore
            return [self.TERMINATE]

        valid_actions = [] if bh==0 or self.danger_zones[(x,y)] == NO_DGR else [SHOOT]
        for act in [W, NW, NE, E, SE, SW]:
          if not self._blocked((x,y),act):
              valid_actions.append(act)
        if valid_actions==[]:
            print("no valid actions at {0}".format(state))
        return valid_actions
            
    def get_initial_state(self):
        self.episode_rewards = []
        return self.initial_state

    def get_goal_states(self):
        return self.goal_states

    def get_current_state(self):
        return self.ship[0], self.ship[1], self.ship_health, self.battery_health, self.direction

    def get_state_danger(self,state):
        """
        return danger level for this state
        """
        dzone = self.danger_zones[(state[0],state[1])]
        danger = self.high_danger if dzone==HI_DGR else self.low_danger if dzone==LO_DGR else 0.0
        death_chance= 1 - pow((1 - pow(danger,state[2])),state[3])
        return death_chance
        

    def get_transitions(self, state, action):
        """
        Transition probabilities governed by probability of damage from active battery, probability of movement
        failure due to existing damage, probability of damaging the battery with fire
        """
        transitions = []

        if state == self.TERMINAL:
            if action == self.TERMINATE:
                return [(self.TERMINAL, 1.0)]
            else:
                return []

        x,y,sh,bh,d = state
        damage_type = self.danger_zones[(x,y)]
        damage_prob = 0.0 if bh<1 else self.high_danger if damage_type == HI_DGR else self.low_danger if damage_type == LO_DGR else 0.0
        move_prob = sh/self.ship_full_health
        #if ship is shooting, keep going in the direction we're going
        direction = d if action == SHOOT else action
        xnew,ynew = self._move((x,y),direction)
        
        if state in self.get_goal_states().keys():
            if action == self.TERMINATE:
                transitions += [(self.TERMINAL, 1.0)]
        elif action in [W, NW, NE, E, SE, SW]:
            #move to next space unless we have a move failure
            transitions += [((xnew, ynew, max(sh-1,0), bh, action),damage_prob*move_prob)]
            transitions += [((xnew, ynew, sh, bh, action),(1-damage_prob)*move_prob)]
            transitions += [((x, y, max(sh-1,0), bh, action),damage_prob*(1-move_prob))]
            transitions += [((x, y, sh, bh, action),(1-damage_prob)*(1-move_prob))]
        elif action == SHOOT:
            #move to next space unless we have a move failure - both ship and battery can get damaged
            transitions += [((xnew, ynew, max(sh-1,0), max(bh-1,0), d),damage_prob*damage_prob*move_prob)]
            transitions += [((xnew, ynew, sh, max(bh-1,0), d),damage_prob*(1-damage_prob)*move_prob)]
            transitions += [((x, y, max(sh-1,0), max(bh-1,0), d),damage_prob*damage_prob*(1-move_prob))]
            transitions += [((x, y, sh, max(bh-1,0), d),damage_prob*(1-damage_prob)*(1-move_prob))]
            transitions += [((xnew, ynew, max(sh-1,0), bh, d),(1-damage_prob)*damage_prob*move_prob)]
            transitions += [((xnew, ynew, sh, bh, d),(1-damage_prob)*(1-damage_prob)*move_prob)]
            transitions += [((x, y, max(sh-1,0), bh, d),(1-damage_prob)*damage_prob*(1-move_prob))]
            transitions += [((x, y, sh, bh, d),(1-damage_prob)*(1-damage_prob)*(1-move_prob))]

        transitions = [t for t in transitions if t[1]>0.0]
        # Merge any duplicate outcomes
        merged = defaultdict(lambda: 0.0)
        for (state, probability) in transitions:
            merged[state] = merged[state] + probability

        transitions = []
        for outcome in merged.keys():
            transitions += [(outcome, merged[outcome])]

        #goal_transitions = [t for t in transitions if t[0] in self.get_goal_states().keys() and t[0][2]>0]
        #if goal_transitions != []:
        #    print(goal_transitions)

        return transitions

    def get_reward(self, state, action, new_state):
        reward = 0.0
        if state in self.get_goal_states().keys() and new_state == self.TERMINAL:
            reward = self.get_goal_states().get(state)
        else:
            reward = self.action_cost
        step = len(self.episode_rewards)
        self.episode_rewards += [reward * (self.discount_factor ** step)]
        return reward

    def get_discount_factor(self):
        return self.discount_factor

    def is_terminal(self, state):
        if state == self.TERMINAL:
            return True
        return False

    def get_rewards(self):
        return self.rewards

    """
        Create a world from a string representation of the ContestedCrossing numeric parameters
        - high_danger (the probability of being damaged in a high-danger location)
        - low_danger (the probability of being damaged in a low-danger location)
        - battery_health (damage a battery can take until it's inoperable)
        - ship_health (damage the ship can take until it sinks)
        - discount_factor
        - action_cost
    """

    @staticmethod
    def create(string):
        # Parse the numeric parameters as a comma-separated list
        vars = string.split(",")
        return ContestedCrossing(
            high_danger=float(vars[0]),
            low_danger=float(vars[1]),
            battery_health = int(vars[2]),
            ship_health = int(vars[3]),
            discount_factor=float(vars[4]),
            action_cost=float(vars[5])
        )

    @staticmethod
    def open(file):
        file = open(file, "r")
        string = file.read().splitlines()
        file.close()
        return ContestedCrossing.create(string)

    @staticmethod
    def matplotlib_installed():
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            return True
        except ModuleNotFoundError:
            return False

    """ Visualise a Contested Crossing problem """

    def visualise(self, agent_position=None, title="", cell_size=1, gif=False):
        if self.matplotlib_installed():
            return self.visualise_as_image(agent_position=agent_position, title=title, cell_size=cell_size, gif=True)[0]
        else:
            print(self.to_string(title=title))

    """ Visualise a Contested Crossing value function """
    def visualise_value_function(self, value_function, title="", cell_size=1, gif=False, mode=3):
        """
        Because the information is 5-dimensional, other metrics must be extracted in order to display it on a 2-D map
        Default view is just mean (sd) per location
        """
        flat_values = self._make_values_flat(value_function.value_table)
        if self.matplotlib_installed():
            return self.visualise_as_image(title=title, cell_size=cell_size, gif=gif, values=flat_values, mode=mode,plot=True)
        else:
            print(self.to_string(values=flat_values, title=title))

    def visualise_q_function(self, qfunction, title="", cell_size=1, gif=False):
        flat_q = self._make_q_flat(qfunction.qtable)
        if self.matplotlib_installed():
            return self.visualise_as_image(title=title, cell_size=cell_size, gif=gif, qfunction=flat_q, plot=True)
        else:
            print(self.q_function_to_string(qfunction, title=title))

    def visualise_policy(self, policy, title="", cell_size=1, gif=False, mode=0):
        if self.matplotlib_installed():
            return self.visualise_as_image(title=title, cell_size=cell_size, gif=gif, policy=policy, mode=mode, plot=True)
        else:
            print(self.policy_to_string(policy, title=title))

    def visualise_stochastic_policy(self, policy, title="", cell_size=1, gif=False):
        if self.matplotlib_installed():
            return self.visualise_stochastic_policy_as_image(policy, title=title, cell_size=cell_size, gif=gif, plot=True)
        else:
            # TODO make a stochastic policy to string
            pass

    """ Visualise a contested crossing as a formatted string """
    def to_string(self, title="",values=None):
        w_arrow = "<" #"\u1f860"
        nw_arrow = "b" #"\u1f864"
        ne_arrow = "d" #"\u1f865"
        e_arrow = ">" #"\u1f862"
        se_arrow = "q" #"\u1f866"
        sw_arrow = "p" #"\u1f867"
        hi_danger = "\u263c"
        lo_danger = "." #"\u00a4"
        nothing = " "
        land_fill = "#"
        shore_fill="|"
        ship = "\u224b"
        battery = "*"

        danger_settings = defaultdict(lambda:nothing,{LO_DGR:lo_danger,HI_DGR:hi_danger})                           
        fill_settings = defaultdict(lambda:nothing,{LAND:land_fill,SHORE1:shore_fill,SHORE2:shore_fill})                           
        shipx,shipy,_,_,_ = self.initial_state
        leftpad = ['----','    ','   |','  | ',' |  ','|   ']

        def constr_cell(lineno,origin):
            x,y=origin
            if lineno == 0:
                cell_str = [" ","-","-","-","-","-","-","-"]
                if self.terrain[(x,y,1)] != LAND:
                    cell_str[1]=e_arrow
                    cell_str[7]=w_arrow
                cell_str[0]=danger_settings[self.danger_zones[origin]]
                if origin==(shipx,shipy):
                    cell_str[0]=ship
                elif origin==self.battery:
                    cell_str[0]=battery

                if values is not None:
                    return cell_str[0]+cell_str[1]+'{0:+}     '.format(round(values[(origin)],1))[:5]+cell_str[7]
                else:
                    return "".join(cell_str)
            if lineno == 1:
                terrain = self.terrain[(x,y,1)]
                cell_fill = fill_settings[terrain]
                cell_str = ["|",cell_fill,cell_fill,cell_fill,cell_fill,cell_fill,cell_fill,cell_fill]
                if terrain!=LAND:
                    cell_str[1]=se_arrow
                    cell_str[7]=sw_arrow
                return "".join(cell_str)
            if lineno == 2:
                cell_fill_down = fill_settings[self.terrain[(x,y,0)]]
                cell_fill_right = fill_settings[self.terrain[(x,y,1)]]
                return cell_fill_down+"|"+cell_fill_right+cell_fill_right+cell_fill_right+cell_fill_right+cell_fill_right+"|"
            if lineno == 3:
                cell_fill_down = fill_settings[self.terrain[(x,y,0)]]
                cell_fill_center = fill_settings[self.terrain[(x,y,1)]]
                cell_fill_right = fill_settings[self.terrain[(x+1,y,0)]]
                return cell_fill_down+cell_fill_down+"|"+cell_fill_center+cell_fill_center+cell_fill_center+"|"+cell_fill_right
            if lineno == 4:
                cell_fill_down = fill_settings[self.terrain[(x,y,0)]]
                cell_fill_center = fill_settings[self.terrain[(x,y,1)]]
                cell_fill_right = fill_settings[self.terrain[(x+1,y,0)]]
                return cell_fill_down+cell_fill_down+cell_fill_down+"|"+cell_fill_center+"|"+cell_fill_right+cell_fill_right
            if lineno == 5:
                cell_fill = fill_settings[self.terrain[(x,y,0)]]
                cell_fill_right = fill_settings[self.terrain[(x+1,y,0)]]
                cell_str = [cell_fill,cell_fill,cell_fill,cell_fill,"|",cell_fill_right,cell_fill_right,cell_fill_right]
                if cell_fill != land_fill:
                    cell_str[1] = nw_arrow
                if cell_fill_right != land_fill:
                    cell_str[7] = ne_arrow
                return "".join(cell_str)

        cell_strings=[]
        for y in range(self.height-1):
            for lineno in range(6):
                if y%2==1:
                    cell_strings.append(leftpad[lineno])
                for x in range(self.width):
                    cell_strings.append(constr_cell(lineno,(x,y)))
                cell_strings.append("\n")
        #for last height line, just do lineno 0
        y=self.height-1
        if y%2==1:
            cell_strings.append(leftpad[0])
        for x in range(self.width):
            cell_strings.append(constr_cell(0,(x,y)))
        cell_strings.append("\n")
        return "".join(cell_strings)

        
    def q_function_to_string(self, qfunction, title=""):
        #TODO
        return ""

    def policy_to_string(self, policy, title=""):
        #TODO
        return ""

    def initialise_world(self, cell_size=1, values=None, policy=None, qfunction=None, mode=0):
        x_cell = cell_size
        y_cell = cell_size * (1/np.sqrt(2))
        pt_size = cell_size*100
        line_size = cell_size*2
        grid_pos = lambda o:((o[0]+(o[1]%2)/2)*x_cell,(self.height-o[1])*y_cell)

        fig = plt.figure(figsize=(self.width * x_cell, self.height * y_cell))                     
        plt.subplots_adjust(top=0.92, bottom=0.01, right=1, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_ticklabels([])  # clear x tick labels
        ax.axes.yaxis.set_ticklabels([])  # clear y tick labels
        ax.tick_params(which='both', top=False, left=False, right=False, bottom=False)
                         
        # Initialise the map points and map regions using separate arrays for up-pointing and down-pointing triangles
        img_pts = [[REGION_COLOURS[NO_DGR] for _ in range(self.width)] for _ in range(self.height)]
        img_down_tri = [[REGION_COLOURS[NO_DGR] for _ in range(self.width)] for _ in range(self.height)]
        img_up_tri = [[REGION_COLOURS[NO_DGR] for _ in range(self.width)] for _ in range(self.height)]

        # Set appropriate colours
        
        for y in range(self.height-1):
            for x in range(self.width):
                img_pt=REGION_COLOURS[self.danger_zones[(x,y)]]
                img_down_tri=REGION_COLOURS[self.terrain[(x,y,1)]]+"33"
                img_up_tri=REGION_COLOURS[self.terrain[(x,y,0)]]+"33"
                img_edge_sw = REGION_COLOURS[self.terrain[(x,y,SW)]]+"33"
                img_edge_se = REGION_COLOURS[self.terrain[(x,y,SE)]]+"33"
                img_edge_e = REGION_COLOURS[self.terrain[(x,y,E)]]+"33"
                thispoint=grid_pos((x,y))
                if not (x==self.width-1 and y%2==1):
                    t1 = plt.Polygon([grid_pos((x,y)),grid_pos(right_cell((x,y))),grid_pos(down_right_cell((x,y)))],color=img_down_tri)
                    plt.gca().add_patch(t1)
                if not (x==0 and y%2==0):
                    t2 = plt.Polygon([grid_pos((x,y)),grid_pos(down_left_cell((x,y))),grid_pos(down_right_cell((x,y)))],color=img_up_tri)
                    plt.gca().add_patch(t2)

                if img_edge_se!=REGION_COLOURS[-1]:
                    endpoint = grid_pos(down_right_cell((x,y)))
                    plt.plot([thispoint[0],endpoint[0]],[thispoint[1],endpoint[1]],color=img_edge_se, linewidth=line_size)
                if img_edge_sw!=REGION_COLOURS[-1]:
                    endpoint = grid_pos(down_left_cell((x,y)))
                    plt.plot([thispoint[0],endpoint[0]],[thispoint[1],endpoint[1]],color=img_edge_sw, linewidth=line_size)
                if img_edge_e!=REGION_COLOURS[-1]:
                    endpoint = grid_pos(right_cell((x,y)))
                    plt.plot([thispoint[0],endpoint[0]],[thispoint[1],endpoint[1]],color=img_edge_e, linewidth=line_size)
                if values is not None:
                    self._values_plot(ax,thispoint,cell_size,values[(x,y)], mode)
                if policy is not None:
                    if mode==0:
                        self._policy_plot(ax,(x,y),thispoint, cell_size, policy.policy_table)
                    ax.scatter(thispoint[0],thispoint[1], s=pt_size*10, color=img_pt+"33", marker="o",edgecolor='none')
                if qfunction is not None:
                    self._q_plot(ax, (x,y), thispoint, cell_size, qfunction)
                if policy is None and qfunction is None:
                    ax.scatter(thispoint[0],thispoint[1], s=pt_size, color=img_pt, marker="*")
        
        if policy is not None and mode==1:
            self._path_plot(ax, policy, cell_size)
        return fig, ax

    def _policy_plot(self,ax, point, gridorigin, cell_size, policy):
        sizefactor=40
        pointsize=sizefactor*10
        text_args = dict(ha='center', va='center', fontsize=cell_size*sizefactor)
        arrow_color = "#00000008"
        arrow = "$\u2007\u2007\u2192$" #a right arrow
        explode = "$\u0489$" # possibilities \u263c\u25cc\u0489\u0488
        rotations={E:0,NE:60,NW:120,W:180,SW:240,SE:300}
        x,y=point
        importance={a:0 for a in self.get_actions()}
        #"policy" is a tabular_policy
        for sh in range(self.ship_full_health+1):
            for bh in range(self.battery_full_health+1):
                for d in range(6):
                    dir=policy[(x,y,sh,bh,d)]
                    if dir in self.get_actions():
                        importance[dir]+=1
        i_weight=sum(importance.values())
        if i_weight==0:
            return
        #show shooting action
        fade='{:02x}'.format(int(importance[SHOOT]*255/i_weight))
        ax.scatter(gridorigin[0],gridorigin[1], s=cell_size*pointsize, color="#ff1111"+fade, marker=explode)
        #all others
        for dir in [E,NE,NW,W,SW,SE]:
            if importance[dir]*sizefactor/i_weight >= 1:
                fade='{:02x}'.format(int(importance[dir]*255/i_weight))
                rot = rotations[dir]
                text_args = dict(ha='center', va='center', fontsize=cell_size*sizefactor,
                                 color="#000000"+fade, rotation=rot)
                ax.text(gridorigin[0]+0.09,gridorigin[1],arrow,**text_args)

    def _path_plot(self, ax, policy, cell_size):
        x_cell = cell_size
        y_cell = cell_size * (1/np.sqrt(2))
        grid_pos = lambda o:((o[0]+(o[1]%2)/2)*x_cell,(self.height-o[1])*y_cell)
        pathcount=100
        breaktime=200
        ship_health_cols = {(a,b):(1-(a/self.ship_full_health),0,1-(b/self.battery_full_health),0.5) for a in range (self.ship_full_health+1)
                            for b in range (self.battery_full_health+1)}
        for i in range(pathcount):
            state = self.get_initial_state()
            endcheck=0
            while endcheck<breaktime and not self.is_terminal(state):
                endcheck+=1
                action = policy.select_action(state)
                (next_state, reward) = self.execute(state, action)
                thispoint=grid_pos(state[:2])
                if not self.is_terminal(next_state):
                    endpoint=grid_pos(next_state[:2])
                    xoffs = (pathcount/2-i)*cell_size/500
                    yoffs = (pathcount/2-i)*cell_size/5000
                    #if action==SHOOT:
                    #    ax.scatter(endpoint[0],endpoint[1], s=cell_size*250, color=ship_health_cols[(state[2],state[3])], marker="$\u0489$")
                    ax.plot([thispoint[0]+xoffs,endpoint[0]+xoffs],[thispoint[1]+yoffs,endpoint[1]+yoffs],color=ship_health_cols[(state[2],state[3])])
                    if(next_state[2]==0):
                        ax.scatter(endpoint[0],endpoint[1], s=cell_size*250, color=(0,0,0,0.5), marker="*")
                state = next_state


    def _q_plot(self, ax, point, gridorigin, cell_size, qtab):
        x,y = point
        move_offsets = {E:(0.2,0),NE:(0.1,0.17),NW:(-0.1,0.17),W:(-0.2,0),SW:(-0.1,-0.17),SE:(0.1,-0.17)}
        text_args = dict(ha='center', va='center', fontsize=cell_size*3, color='#000000')
        greyed_args = dict(ha='center', va='center', fontsize=cell_size*3, color='#00000022')
        shoot_text='{0}\n({1})'.format(round(qtab[(x,y,SHOOT)]['mean'],2),round(qtab[(x,y,SHOOT)]['sd'],2))
        args = greyed_args if qtab[(x,y,SHOOT)]['mean'] == 0 and qtab[(x,y,SHOOT)]['sd'] == 0 else text_args
        ax.text(gridorigin[0],gridorigin[1],shoot_text,**args)
        for m in move_offsets:
            move_text='{0}\n({1})'.format(round(qtab[(x,y,m)]['mean'],2),round(qtab[(x,y,m)]['sd'],2))
            args = greyed_args if qtab[(x,y,m)]['mean'] == 0 and qtab[(x,y,m)]['sd'] == 0 else text_args
            xpos=gridorigin[0]+cell_size*move_offsets[m][0]
            ypos=gridorigin[1]+cell_size*move_offsets[m][1]
            ax.text(xpos, ypos, move_text, **args)


    def _values_plot(self,ax, gridorigin, cell_size, pt_values, mode = 0):
        text_args = dict(ha='left', va='top', fontsize=cell_size*8, color='#343434')
        smalltext_head = dict(ha='center', va='top', fontsize=cell_size*6, color='#343488')
        smalltext_body = dict(ha='left', va='top', fontsize=cell_size*4, color='#343434')
        subcell=cell_size/12
        bar_colors = {True:"#111111",False:"#ff1111"}
        #"values" is a point-indexed dict of dict with keys 'mean' and headings for other vars
        mytext='{0} ({1})'.format(round(pt_values['mean'],2),round(pt_values['sd'],2))
        ax.text(gridorigin[0],gridorigin[1],mytext,**text_args)
        if pt_values['mean']==0.0 or mode==3:
            return
        if mode==2:
            #mode == 2 - max/min/mode
            ax.text(gridorigin[0],gridorigin[1]-3*subcell,"max:  {0}".format(round(max(pt_values['key_vals'].keys()),2)),**smalltext_body)
            ax.text(gridorigin[0],gridorigin[1]-4*subcell,"min:  {0}".format(round(min(pt_values['key_vals'].keys()),2)),**smalltext_body)
            ax.text(gridorigin[0],gridorigin[1]-5*subcell,"mode: {0}".format(round(sorted(pt_values['key_vals'].items(),
                                                                                key = lambda v : len(v[1]),reverse=True)[0][0],2)),**smalltext_body)
            return
        hspace=1.0
        for k in pt_values['sub_means']:
            vspace=1.5
            smalltext_head['rotation']=k[1]
            ax.text(gridorigin[0]+(hspace+0.5)*subcell,gridorigin[1]-vspace*subcell,k[0],**smalltext_head)
            vspace+=0.5
            for d in pt_values['sub_means'][k]:
                vspace+=1.0 if mode==0 else 0.5
                if mode==0:
                    mytext="{0}: {1}".format(d,round(pt_values['sub_means'][k][d],2))
                    ax.text(gridorigin[0]+hspace*subcell,gridorigin[1]-vspace*subcell,mytext,**smalltext_body)
                elif mode==1:
                    linestartx = gridorigin[0]+hspace*subcell
                    liney = gridorigin[1]-vspace*subcell
                    ax.plot([linestartx,linestartx+abs(subcell*pt_values['sub_means'][k][d]/BERTH_REWARD)],[liney,liney],
                                 color=bar_colors[pt_values['sub_means'][k][d]>0], linewidth=cell_size)
            hspace+=4.0

    def _make_values_flat(self, value_table):
        flat_values = {}
        for x in range(self.width):
            for y in range(self.height):
                vals={k:value_table[k] for k in value_table if k[0]==x and k[1]==y}
                flat_values[(x,y)]={}
                xyvals = [vals[v] for v in vals]
                flat_values[(x,y)]['mean'] = 0.0 if xyvals==[] else np.mean(xyvals)
                flat_values[(x,y)]['sd'] = 0.0 if xyvals==[] else np.std(xyvals)
                flat_values[(x,y)]['sub_means'] = {}
                sms = {d:[vals[v] for v in vals if v[2]==d] for d in range(self.ship_full_health+1)}
                smb = {d:[vals[v] for v in vals if v[3]==d] for d in range(self.battery_full_health+1)}
                flat_values[(x,y)]['sub_means'][(SHIP_SYMBOL,240)] = {d:np.mean(sms[d]) for d in sms if len(sms[d])>0}
                flat_values[(x,y)]['sub_means'][(BATTERY_SYMBOL,0)] = {d:np.mean(smb[d]) for d in smb if len(smb[d])>0}
                flat_values[(x,y)]['key_vals']={k:[] for k in set(xyvals)}
                for v in vals:
                    flat_values[(x,y)]['key_vals'][vals[v]].append(v[2:])
        return flat_values

    def _make_q_flat(self, qtable):
        flat_values = {}
        for x in range(self.width):
            for y in range(self.height):
                for act in self.get_actions():
                    vals={k:qtable[k] for k in qtable if k[0][0]==x and k[0][1]==y and k[1]==act}
                    flat_values[(x,y,act)]={}
                    xyavals = [vals[v] for v in vals]
                    flat_values[(x,y,act)]['mean'] = 0.0 if xyavals==[] else np.mean(xyavals)
                    flat_values[(x,y,act)]['sd'] = 0.0 if xyavals==[] else np.std(xyavals)
        return flat_values

                
    def visualise_as_image(self, agent_position=None, title="", cell_size=1, gif=False, values=None, policy=None, qfunction=None, mode=0, plot=False):
        """
        visualise with optional overlay for values or policy.
        values modes: 0 - numeric means, 1 - graphical means, 2 - key values
        """

        ship_args = dict(ha='center', va='center', fontsize=cell_size*25, color=SHIP_COLOR, rotation=240)
        bat_args = dict(ha='center', va='center', fontsize=cell_size*15, color=BATTERY_COLOR)
        x_cell = cell_size
        y_cell = cell_size * (1/np.sqrt(2))
        pt_size = cell_size*100
        grid_pos = lambda o:((o[0]+(o[1]%2)/2)*x_cell,(self.height-o[1])*y_cell)
        current_position=agent_position
        if current_position is None:
            x,y,_,_,_=self.get_initial_state()
            current_position = (x,y)
        fig, ax = self.initialise_world(cell_size=cell_size, values=values, policy=policy, qfunction=qfunction, mode=mode)
        shipx, shipy = grid_pos(current_position)
        ship_texts = plt.text(shipx, shipy, SHIP_SYMBOL, **ship_args)
        batx, baty = grid_pos(self.battery)
        texts = plt.text(batx, baty, BATTERY_SYMBOL, **bat_args)
        plt.title(title)

        if gif:
            return fig, ax
        else:
            plt.show()

    def visualise_policy_as_image(self, policy, title="", agent_position=None,  cell_size=1, gif=False, values=None, mode=0):
        return self.visualise_as_image(agent_position=agent_position, title=title, cell_size=cell_size, gif=gif, values=values, policy=policy, mode=mode)

    def execute(self, state, action):
        if state in self.goal_states:
            self.rewards += [self.episode_rewards]
            return MDP.execute(self, state=state, action=self.TERMINATE)
        return super().execute(state, action)

LONG_REGIONS = {(2,0):SHORE2,(3,0):LAND,(4,0):LAND,(5,0):LAND,(6,0):LAND,(7,0):LAND,(2,1):SHORE2,(3,1):LAND,
                (4,1):LAND,(5,1):LAND,(6,1):LAND,(3,2):SHORE2,(4,2):LAND,(5,2):LAND,(6,2):LAND,(7,2):LAND,
                (3,3):SHORE2,(4,3):SHORE2,(5,3):SHORE2,(6,3):SHORE2,(5,27):SHORE1,(6,27):SHORE1,(5,28):SHORE1,
                (6,28):LAND,(7,28):LAND,(4,29):SHORE1,(5,29):LAND,(6,29):LAND,(4,30):SHORE1,(5,30):LAND,
                (6,30):LAND,(7,30):LAND,(3,31):SHORE1,(4,31):LAND,(5,31):LAND,(6,31):LAND,(7,31):LAND}

LONG_DANGER =  {(1,1):LO_DGR,(2,1):LO_DGR,(1,2):LO_DGR,(2,2):LO_DGR,(3,2):HI_DGR,(1,3):LO_DGR,(2,3):HI_DGR,
                (3,3):HI_DGR,(4,3):HI_DGR,(5,3):LO_DGR,(1,4):LO_DGR,(2,4):HI_DGR,(3,4):HI_DGR,(4,4):HI_DGR,
                (5,4):HI_DGR,(6,4):LO_DGR,(1,5):LO_DGR,(2,5):HI_DGR,(3,5):HI_DGR,(4,5):HI_DGR,(5,5):LO_DGR,
                (1,6):LO_DGR,(2,6):LO_DGR,(3,6):HI_DGR,(4,6):HI_DGR,(5,6):LO_DGR,(6,6):LO_DGR,(1,7):LO_DGR,
                (2,7):LO_DGR,(3,7):HI_DGR,(4,7):LO_DGR,(5,7):LO_DGR,(2,8):LO_DGR,(3,8):HI_DGR,(4,8):HI_DGR,
                (5,8):LO_DGR,(6,9):HI_DGR,(0,10):HI_DGR,(2,10):HI_DGR,
                (3,10):HI_DGR,(4,10):HI_DGR,(6,10):HI_DGR,(0,11):LO_DGR,(0,12):LO_DGR,
                (1,12):LO_DGR,(2,12):HI_DGR,(0,13):LO_DGR,(1,13):HI_DGR,(2,13):HI_DGR,(3,13):HI_DGR,(4,13):LO_DGR,
                (0,14):LO_DGR,(1,14):HI_DGR,(2,14):HI_DGR,(3,14):HI_DGR,(4,14):HI_DGR,(5,14):LO_DGR,(0,15):LO_DGR,
                (1,15):HI_DGR,(2,15):HI_DGR,(3,15):HI_DGR,(4,15):LO_DGR,(0,16):LO_DGR,(1,16):LO_DGR,(2,16):HI_DGR,
                (3,16):HI_DGR,(4,16):LO_DGR,(5,16):LO_DGR,(0,17):LO_DGR,(1,17):LO_DGR,(2,17):HI_DGR,(3,17):LO_DGR,
                (4,17):LO_DGR,(1,18):LO_DGR,(2,18):LO_DGR,(3,18):LO_DGR,(4,18):LO_DGR,(5,18):LO_DGR,(6,18):LO_DGR,
                (3,19):LO_DGR,(4,19):LO_DGR,(0,20):HI_DGR,(1,20):HI_DGR,
                (3,20):HI_DGR,(4,20):HI_DGR,(5,20):HI_DGR,
                (6,20):HI_DGR,(2,21):LO_DGR,(3,21):LO_DGR,(2,22):LO_DGR,(3,22):LO_DGR,(4,22):HI_DGR,(0,23):HI_DGR,
                (1,23):HI_DGR,(2,23):HI_DGR,
                (3,23):HI_DGR,(4,23):HI_DGR,(6,23):LO_DGR,(2,24):LO_DGR,(3,24):HI_DGR,(4,24):HI_DGR,
                (6,24):HI_DGR,(7,24):LO_DGR,(2,25):LO_DGR,(3,25):HI_DGR,(5,25):HI_DGR,
                (6,25):LO_DGR,(2,26):LO_DGR,(3,26):LO_DGR,(5,26):HI_DGR,(6,26):LO_DGR,(7,26):LO_DGR,
                (2,27):LO_DGR,(4,27):HI_DGR,(5,27):LO_DGR,(6,27):LO_DGR,(4,28):LO_DGR,
                (5,28):LO_DGR,(6,28):LO_DGR,(3,29):LO_DGR,(4,29):LO_DGR,(5,29):LO_DGR}
LONG_WIDTH = 7
LONG_HEIGHT = 32
LONG_BATTERY = (3,3)
LONG_SHIP = (4,30)

class LongCrossing(ContestedCrossing):
    def __init__(
        self,
        high_danger=0.9,
        low_danger=0.1,
        battery_health = 2,
        ship_health = 3,
        discount_factor=0.9,
        action_cost=0.0,
        width=LONG_WIDTH,
        height=LONG_HEIGHT,
        regions=LONG_REGIONS,
        danger=LONG_DANGER,
        battery=LONG_BATTERY,
        ship=LONG_SHIP
    ):
        super().__init__(
            high_danger=high_danger,
            low_danger=low_danger,
            battery_health = battery_health,
            ship_health = ship_health,
            discount_factor=discount_factor,
            action_cost=action_cost,
            width=width,
            height=height,
            regions=regions,
            danger=danger,
            battery=battery,
            ship=ship

        )
        


