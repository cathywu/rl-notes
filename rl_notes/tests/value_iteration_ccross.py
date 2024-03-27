from contested_crossing import ContestedCrossing
from value_iteration import ValueIteration
from tabular_value_function import TabularValueFunction
from value_policy import ValuePolicy
from stochastic_value_policy import StochasticValuePolicy
from tests.plot import Plot


ccross = ContestedCrossing()
values = TabularValueFunction()
ValueIteration(ccross, values).value_iteration(max_iterations=10)
enemy_health = direction = 1
for x in [1, 2]:
    for y in [1, 2]:
        for ship_health in [1, 2]:
            print(
                "state: {0} - value: {1}".format(
                    (x, y, ship_health, enemy_health, direction),
                    round(
                        values.get_value(
                            (x, y, ship_health, enemy_health, direction)
                        ),
                        3,
                    ),
                )
            )

for iterations in [10]:
    values = TabularValueFunction()
    ValueIteration(ccross, values).value_iteration(max_iterations=iterations)
    ccross.visualise_value_function(
        values, "After %d iterations" % (iterations), mode=3, cell_size=1.6
    )

values = TabularValueFunction()
ValueIteration(ccross, values).value_iteration(max_iterations=2)
policy = StochasticValuePolicy(ccross, values)
ccross.visualise_policy(policy, "Policy plot after 100 iterations", mode=0)
ccross.visualise_policy(policy, "Path Plot after 100 iterations", mode=1)


values = TabularValueFunction()
ccross = ContestedCrossing()
policy = StochasticValuePolicy(ccross, values)
rewards = ccross.execute_policy(policy, episodes=1)
for _ in range(50):
    ValueIteration(ccross, values).value_iteration(max_iterations=1)
    policy = StochasticValuePolicy(ccross, values)
    rewards += ccross.execute_policy(policy, episodes=1)

Plot.plot_cumulative_rewards(["Value iteration"], [rewards], smoothing_factor=0.0)
Plot.plot_cumulative_rewards(["Value iteration"], [rewards], smoothing_factor=0.9)
