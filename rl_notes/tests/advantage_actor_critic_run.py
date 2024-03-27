from advantage_actor_critic import AdvantageActorCritic
from deep_nn_policy import DeepNeuralNetworkPolicy
from deep_value_function import DeepValueFunction
from gridworld import GridWorld

gridworld = GridWorld()

# Instantiate the critic
critic = DeepValueFunction(mdp=gridworld, state_space=len(gridworld.get_initial_state()), hidden_dim=16)

# Instantiate the actor
actor = DeepNeuralNetworkPolicy(
    gridworld, state_space=len(gridworld.get_initial_state()), action_space=4
)

advantage_actor_critic = AdvantageActorCritic(mdp=gridworld, actor=actor, critic=critic)
gridworld.visualise_value_function(critic, grid_size=0.8, title=f"Value Function: {0} iterations")
gridworld.visualise_stochastic_policy(actor)
gridworld.visualise_policy_as_image(actor)

advantage_actor_critic.execute(100)
gridworld.visualise_value_function(critic, grid_size=0.8, title=f"Value Function: {100} iterations")
gridworld.visualise_stochastic_policy(actor)
gridworld.visualise_policy_as_image(actor)

advantage_actor_critic.execute(1000)
gridworld.visualise_value_function(critic, grid_size=0.8, title=f"Value Function: {1000} iterations")
gridworld.visualise_stochastic_policy(actor)
gridworld.visualise_policy_as_image(actor)

