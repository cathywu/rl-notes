from multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing
from tests.multi_armed_bandit_tests.run_bandit import run_bandit
from tests.plot import Plot


def plot_epsilon_decreasing(drift=False):
    alpha09 = run_bandit(EpsilonDecreasing(alpha=0.9), drift=drift)
    alpha099 = run_bandit(EpsilonDecreasing(alpha=0.99), drift=drift)
    alpha0999 = run_bandit(EpsilonDecreasing(alpha=0.999), drift=drift)
    alpha1 = run_bandit(EpsilonDecreasing(alpha=1.0), drift=drift)

    Plot.plot_rewards(
        ["alpha = 0.9", "alpha = 0.99", "alpha= 0.999", "alpha = 1.0"],
        [alpha09, alpha099, alpha0999, alpha1],
    )


plot_epsilon_decreasing()
