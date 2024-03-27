from gridworld import GridWorld
from value_iteration import ValueIteration
from value_policy import ValuePolicy
from stochastic_value_policy import StochasticValuePolicy
from tabular_value_function import TabularValueFunction
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot

gridworld = GridWorld()
for iterations in [0, 1, 2, 3, 4, 5, 10, 100]:
    values = TabularValueFunction()
    ValueIteration(gridworld, values).value_iteration(max_iterations=iterations)
    gridworld.visualise_value_function(values, "After %d iterations" % (iterations))

values = TabularValueFunction()
ValueIteration(gridworld, values).value_iteration(max_iterations=100)
policy = ValuePolicy(gridworld, values)
gridworld.visualise_policy(policy, "Policy after 100 iterations")

maze = GridWorld.open("../python_code/layouts/maze.txt")
values = TabularValueFunction()
ValueIteration(maze, values).value_iteration(max_iterations=100)
maze.visualise_value_function(values, grid_size=0.8, title="100 iterations")
policy = ValuePolicy(maze, values)
maze.visualise_policy(policy, "", grid_size=0.8)

values = TabularValueFunction()
gridworld = GridWorld()
policy = ValuePolicy(gridworld, values)
rewards = gridworld.execute_policy(policy, episodes=1)
for _ in range(50):
    ValueIteration(gridworld, values).value_iteration(max_iterations=1)
    policy = ValuePolicy(gridworld, values)
    rewards += gridworld.execute_policy(policy, episodes=1)

Plot.plot_cumulative_rewards(["Value iteration"], [rewards], smoothing_factor=0.0)
Plot.plot_cumulative_rewards(["Value iteration"], [rewards], smoothing_factor=0.9)

# Plot the curve on a deterministic version of GridWorld (noise=0.0)
values = TabularValueFunction()
gridworld = GridWorld(noise=0.0)
policy = StochasticValuePolicy(gridworld, values, EpsilonGreedy(epsilon=0.1))
rewards = gridworld.execute_policy(policy, episodes=1)
for _ in range(50):
    ValueIteration(gridworld, values).value_iteration(max_iterations=1)
    policy = StochasticValuePolicy(gridworld, values, EpsilonGreedy(epsilon=0.1))
    rewards += gridworld.execute_policy(policy, episodes=1)

Plot.plot_cumulative_rewards(["Value iteration"], [rewards], smoothing_factor=0.0)
Plot.plot_cumulative_rewards(["Value iteration"], [rewards], smoothing_factor=0.9)
