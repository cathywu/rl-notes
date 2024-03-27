import numpy as np

from freeway_abstraction import FreewayAbstraction
from freeway import Freeway
from experience_replay_learner import ExperienceReplayLearner
from deep_qfunction import DeepQFunction
from ale_wrapper import ALEWrapper
from stochastic_q_policy import StochasticQPolicy


from multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot


#version="Breakout-ramDeterministic-v4"
version="Freeway-ramDeterministic-v4"
#version="ALE/Frogger-ram-v5"
print("==========\nQ-learning: " + version + "\n==========")

#mdp = FreewayAbstraction(render_mode="rgb_array", discount_factor=0.99)
#mdp = Freeway(render_mode="rgb_array", discount_factor=0.99)

mdp = ALEWrapper(version=version, render_mode="rgb_array", discount_factor=0.99)
policy_qfunction = DeepQFunction(mdp, state_space=len(mdp.get_initial_state()), action_space=len(mdp.get_actions()))
target_qfunction = DeepQFunction(mdp, state_space=len(mdp.get_initial_state()), action_space=len(mdp.get_actions()))

#QLearning(mdp, Softmax(), qfunction).execute(episodes=1)
#ExperienceReplayLearner(mdp, EpsilonGreedy(), qfunction, replay_period=100, batch_size=6).execute(episodes=1)
#policy = StochasticQPolicy(qfunction, mdp.get_actions(), EpsilonGreedy(epsilon=0.0))
#rewards = mdp.execute_policy(policy, episodes=1)
#print("Episode 0: {:.2f}".format(rewards[0]))


episodes = 200
episodes_per_evaluation = 10
epsilon = 1.0
epsilon_decay = 0.995
rewards = []
for i in range(1, int(episodes / episodes_per_evaluation) + 1):
    rewards += ExperienceReplayLearner(mdp, EpsilonGreedy(epsilon=epsilon), policy_qfunction, target_qfunction).execute(episodes=episodes_per_evaluation)
    print(f"Episode: {i * episodes_per_evaluation}, Epsilon: {epsilon}, Episode Reward: {np.mean(rewards[-1-episodes_per_evaluation:-1])}")
    epsilon = max(epsilon * (epsilon_decay ** episodes_per_evaluation), 0.01)
    #print("Episode {}: {:.2f}".format((i + 1) * episodes_per_evaluation, rewards[-1-episodes_per_evaluation:-1]))
    policy_qfunction.save("freeway.policy")

policy_qfunction.load("freeway.policy")


Plot.plot_cumulative_rewards(
    ["Q-learning"], [rewards], smoothing_factor=0.0, episodes_per_evaluation=episodes_per_evaluation
)

policy = StochasticQPolicy(policy_qfunction, EpsilonGreedy(epsilon=0.05))

#policy = FreewayHandPolicy()
#mdp = Freeway(render_mode="human")

#mdp = ALEWrapper(version=version, render_mode="human")
mdp = Freeway(render_mode="human", discount_factor=1.0)
exec_rewards = mdp.execute_policy(policy, episodes=1)
print(exec_rewards)
