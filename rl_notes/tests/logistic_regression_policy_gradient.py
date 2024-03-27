from gridworld import GridWorld
from gridworld import OneDimensionalGridWorld
from policy_gradient import PolicyGradient
from logistic_regression_policy import LogisticRegressionPolicy

gridworld = GridWorld(
    height=1, width=11, initial_state=(5, 0), goals=[((0, 0), -1), ((10, 0), 1)]
)
policy = LogisticRegressionPolicy(
    actions=[GridWorld.LEFT, GridWorld.RIGHT],
    num_params=len(gridworld.get_initial_state()),
)
policy_gradient = PolicyGradient(gridworld, policy, alpha=0.1)
policy_gradient.execute(episodes=1000)
policy_image = gridworld.visualise_stochastic_policy(policy)
