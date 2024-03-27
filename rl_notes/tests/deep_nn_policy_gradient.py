from gridworld import GridWorld
from policy_gradient import PolicyGradient
from deep_nn_policy import DeepNeuralNetworkPolicy

gridworld = GridWorld()
policy = DeepNeuralNetworkPolicy(
    gridworld, state_space=len(gridworld.get_initial_state()), action_space=4
)
PolicyGradient(gridworld, policy, alpha=0.1).execute(episodes=1000)
gridworld_image = gridworld.visualise_stochastic_policy(policy)

gridworld.visualise_policy(policy)
