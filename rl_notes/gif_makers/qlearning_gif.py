from gridworld import GridWorld
from gif_maker import GifMaker
from qtable import QTable
from qlearning import QLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy


grid_size = 1.5
gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld, grid_size=grid_size)
qfunction = QTable()
for episode in range(1, 101):
    QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=1)
    title = "Episode %d" % (episode)
    image_texts = gridworld.visualise_q_function(qfunction, title=title, grid_size=grid_size, gif=True)
    gif_maker.add_frame(image_texts, title=title)

gif_maker.save("../../assets/gifs/qlearning.gif")
