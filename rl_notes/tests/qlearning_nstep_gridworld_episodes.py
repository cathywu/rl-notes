from gridworld import GridWorld
from tests.compare_convergence_curves import qlearning_vs_nstep
from gridworld import CliffWorld
from tests.plot import Plot

from contested_crossing import ContestedCrossing

mdp_q = GridWorld()
mdp_s = GridWorld()
#mdp_q = ContestedCrossing()
#mdp_s = ContestedCrossing()
qlearning_vs_nstep(mdp_q, mdp_s, episodes=2000, n=3)