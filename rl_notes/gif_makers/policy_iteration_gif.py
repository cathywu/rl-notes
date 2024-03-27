from gridworld import GridWorld
from gif_maker import GifMaker
from policy_iteration import PolicyIteration
from tabular_policy import TabularPolicy


gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld)
policy = TabularPolicy(default_action=gridworld.LEFT)
for iterations in range(0, 10):
    title = "Iteration %d" % (iterations)
    image_texts = gridworld.visualise_policy(policy, title=title, gif=True)
    gif_maker.add_frame(image_texts, title=title)
    PolicyIteration(gridworld, policy).policy_iteration(max_iterations=1)

gif_maker.save("../../assets/gifs/policy_iteration.gif")
