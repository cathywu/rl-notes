from gridworld import GridWorld
from gif_maker import GifMaker
from qtable import QTable
from qlearning import QLearning
from n_step_qlearning import NStepQLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from side_by_side_comparison import run_learner
from side_by_side_comparison import join_gif

episodes = 20
gridworld = GridWorld()
qfunction = QTable()
learner = QLearning(gridworld, EpsilonGreedy(), qfunction)
run_learner(
    mdp=gridworld,
    learner=learner,
    qfunction=qfunction,
    learner_name="1-Step Q-Learning",
    out_filename="../../assets/gifs/1_step_qlearning.gif",
    episodes=episodes,
)

gridworld = GridWorld()
qfunction = QTable()
learner = NStepQLearning(gridworld, EpsilonGreedy(), qfunction, 5)
run_learner(
    mdp=gridworld,
    learner=learner,
    qfunction=qfunction,
    learner_name="5-Step Q-Learning",
    out_filename="../../assets/gifs/5_step_qlearning.gif",
    episodes=episodes,
)

join_gif(
    filename1="../../assets/gifs/1_step_qlearning.gif",
    filename2="../../assets/gifs/5_step_qlearning.gif",
    out_filename="../../assets/gifs/1_step_vs_5_step_qlearning.gif",
)
