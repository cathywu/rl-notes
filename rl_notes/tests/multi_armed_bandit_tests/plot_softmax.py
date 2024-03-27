from multi_armed_bandit.softmax import Softmax
from tests.multi_armed_bandit_tests.run_bandit import run_bandit
from tests.plot import Plot


def plot_softmax(drift=False):
    tau10 = run_bandit(Softmax(tau=1.0), drift=drift)
    tau11 = run_bandit(Softmax(tau=1.1), drift=drift)
    tau15 = run_bandit(Softmax(tau=1.5), drift=drift)
    tau20 = run_bandit(Softmax(tau=2.0), drift=drift)

    Plot.plot_rewards(
        ["tau = 1.0", "tau = 1.1", "tau = 1.5", "tau = 2.0"],
        [tau10, tau11, tau15, tau20],
    )


plot_softmax(drift=False)
