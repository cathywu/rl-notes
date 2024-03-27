from gridworld import GridWorld
from deep_qfunction import DeepQFunction
from qlearning import QLearning
from deep_nn_policy import DeepNeuralNetworkPolicy
from q_actor_critic import QActorCritic
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from gif_maker import GifMaker
from side_by_side_comparison import join_gif

episodes = 4
grid_size = 1.5
gridworld = GridWorld()
qfunction = DeepQFunction(
    gridworld,
    state_space=len(gridworld.get_initial_state()),
    action_space=5,
    hidden_dim=16,
)
critic = QLearning(gridworld, EpsilonGreedy(), qfunction, alpha=1.0)

actor = DeepNeuralNetworkPolicy(
    gridworld, state_space=len(gridworld.get_initial_state()), action_space=4
)

learner = QActorCritic(mdp=gridworld, actor=actor, critic=critic)

"""
This currently does not work
GifMaker stores the current matplotlib plt and ax objects, and the two GifMarkers both write to the one that is defined 2nd
"""

qfunction_gif_maker = GifMaker(mdp=gridworld)
policy_gif_maker = GifMaker(mdp=gridworld)
for episode in range(0, episodes):
    learner.execute(episodes=1)
    title = "Critic after episode %d" % (episode)
    qfunction_image_texts = gridworld.visualise_q_function(
        qfunction, title=title, grid_size=grid_size, gif=True
    )
    qfunction_gif_maker.add_frame(qfunction_image_texts, title=title)
    title = "Actor after episode %d" % (episode)
    policy_image_texts = gridworld.visualise_stochastic_policy(
        actor, title=title, grid_size=grid_size, gif=True
    )
    policy_gif_maker.add_frame(policy_image_texts, title=title)

qfunction_gif_maker.save("../../assets/gifs/critic.gif")
policy_gif_maker.save("../../assets/gifs/actor.gif")

join_gif(
    filename1="../../assets/gifs/critic.gif",
    filename2="../../assets/gifs/actor.gif",
    out_filename="../../assets/gifs/deep_actor_critic.gif",
)
