
from python_code.contested_crossing import ContestedCrossing
import python_code.contested_crossing as cc

ccross = ContestedCrossing()
for cpoint in [(1,2),(1,3),(2,4)]:
    #check: right,down_right,down_left,left,up_left,up_right in various combinations gives initial point
    assert(cc.right_cell(cc.down_right_cell(cc.down_left_cell(cc.left_cell(cc.up_left_cell(cc.up_right_cell(cpoint))))))==cpoint)
    assert(cc.left_cell(cc.down_left_cell(cc.down_right_cell(cc.right_cell(cc.up_right_cell(cc.up_left_cell(cpoint))))))==cpoint)
    assert(cc.right_cell(cc.up_right_cell(cc.up_left_cell(cc.left_cell(cc.down_left_cell(cc.down_right_cell(cpoint))))))==cpoint)
    assert(cc.left_cell(cc.up_left_cell(cc.up_right_cell(cc.right_cell(cc.down_right_cell(cc.down_left_cell(cpoint))))))==cpoint)

    #check: direction matches
    assert(ccross._direction(cpoint,cc.left_cell(cpoint))==cc.W)
    assert(ccross._direction(cpoint,cc.up_left_cell(cpoint))==cc.NW)
    assert(ccross._direction(cpoint,cc.up_right_cell(cpoint))==cc.NE)
    assert(ccross._direction(cpoint,cc.right_cell(cpoint))==cc.E)
    assert(ccross._direction(cpoint,cc.down_right_cell(cpoint))==cc.SE)
    assert(ccross._direction(cpoint,cc.down_left_cell(cpoint))==cc.SW)
    
#check where exploration takes us
init = ccross.get_initial_state()
explored=set()
nextStates=[init]
while len(nextStates)>0:
    thisState=nextStates[-1]
    acts = ccross.get_actions(thisState)
    explored.add(thisState)
    nextStates = nextStates[:-1]
    assert(len(acts)>0)
    for a in acts:
        if a!=ccross.TERMINATE:
            transitions = set([t[0] for t in ccross.get_transitions(thisState,a) if t[1]>0])
            for t in transitions:
                if not t in explored and not t in nextStates:
                    nextStates += [t]
        
xvals = set([e[0] for e in explored])
assert(min(xvals)>=0 and max(xvals)<=ccross.width)
yvals = set([e[1] for e in explored])
assert(min(yvals)>=0 and max(yvals)<=ccross.height)
shvals = set([e[2] for e in explored])
assert(min(shvals)>=0 and max(shvals)<=ccross.ship_full_health)
bhvals = set([e[3] for e in explored])
assert(min(bhvals)>=0 and max(bhvals)<=ccross.battery_full_health)
