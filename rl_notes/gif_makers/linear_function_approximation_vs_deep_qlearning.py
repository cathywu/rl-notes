from gridworld import GridWorld
from qlearning import QLearning
from linear_qfunction import LinearQFunction
from deep_qfunction import DeepQFunction
from gridworld_better_feature_extractor import GridWorldBetterFeatureExtractor
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from side_by_side_comparison import run_learner
from side_by_side_comparison import join_gif

episodes=50
gridworld = GridWorld()
features = GridWorldBetterFeatureExtractor(gridworld)
qfunction = LinearQFunction(features)
learner = QLearning(gridworld, EpsilonGreedy(), qfunction)
run_learner(mdp=gridworld, learner=learner, qfunction=qfunction, learner_name="Linear Q-learning", out_filename="../../assets/gifs/linear_qlearning.gif", episodes=episodes)

gridworld = GridWorld()
qfunction = DeepQFunction(gridworld, state_space=len(gridworld.get_initial_state()), action_space=5, hidden_dim=16)
learner = QLearning(gridworld, EpsilonGreedy(), qfunction, alpha=1.0)
run_learner(mdp=gridworld, learner=learner, qfunction=qfunction, learner_name="Deep Q-learning", out_filename="../../assets/gifs/deep_qlearning.gif", episodes=episodes)


join_gif(filename1="../../assets/gifs/linear_qlearning.gif", filename2="../../assets/gifs/deep_qlearning.gif", out_filename="../../assets/gifs/linear_qlearning_vs_deep_qlearning.gif")
