from gridworld import GridWorld
from gif_maker import GifMaker
from policy_gradient import PolicyGradient
from logistic_regression_policy import LogisticRegressionPolicy


gridworld = GridWorld(
    height=1, width=11, initial_state=(5, 0), goals=[((0, 0), -1), ((10, 0), 1)]
)
gif_maker = GifMaker(mdp=gridworld)
policy = LogisticRegressionPolicy(
    actions=[GridWorld.LEFT, GridWorld.RIGHT],
    num_params=len(gridworld.get_initial_state()),
)
for iterations in range(0, 20):
    title = "Iterations %d" % (iterations)
    image_texts = gridworld.visualise_stochastic_policy(policy, title=title, gif=True)
    gif_maker.add_frame(image_texts, title=title)
    PolicyGradient(gridworld, policy, alpha=0.1).execute(episodes=1)

gif_maker.save("../../assets/gifs/logistic_regression_policy_gradient.gif")
