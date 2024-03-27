from gridworld import GridWorld
from policy_iteration import PolicyIteration
from tabular_policy import TabularPolicy

mdp = GridWorld(width=20, height=15)
policy = TabularPolicy(default_action=mdp.get_actions()[0])
iterations = PolicyIteration(mdp, policy).policy_iteration(max_iterations=100)
print("Number of iterations until convergence: %d" % (iterations))

mdp.visualise_policy(policy)