import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.nn.functional as F

from policy import StochasticPolicy


class DeepNeuralNetworkPolicy(StochasticPolicy):
    """
    An implementation of a policy that uses a PyTorch (https://pytorch.org/) 
    deep neural network to represent the underlying policy.
    """

    def __init__(self, mdp, state_space, action_space, hidden_dim=64, alpha=0.001):
        self.mdp = mdp
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha

        # Define the policy structure as a sequential neural network.
        self.policy_network = nn.Sequential(
            nn.Linear(in_features=self.state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.action_space),
        )

        # The optimiser for the policy network, used to update policy weights
        self.optimiser = Adam(self.policy_network.parameters(), lr=self.alpha)

        # A two-way mapping from actions to integer IDs for ordinal encoding
        actions = self.mdp.get_actions()
        self.action_to_id = {actions[i]: i for i in range(len(actions))}
        self.id_to_action = {
            action_id: action for action, action_id in self.action_to_id.items()
        }

    """ Select an action using a forward pass through the network """

    def select_action(self, state):
        # Convert the state into a tensor so it can be passed into the network
        state = torch.as_tensor(state, dtype=torch.float32)
        action_logits = self.policy_network(state)
        action_distribution = Categorical(logits=action_logits)
        action = action_distribution.sample()
        return self.id_to_action[action.item()]

    """ Get the probability of an action being selected in a state """

    def get_probability(self, state, action):
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_logits = self.policy_network(state)
        # A softmax layer turns action logits into relative probabilities
        probabilities = F.softmax(input=action_logits, dim=-1).tolist()

        # Convert from a tensor encoding back to the action space
        return probabilities[self.action_to_id[action]]

    def evaluate_actions(self, states, actions):
        action_logits = self.policy_network(states)
        action_distribution = Categorical(logits=action_logits)
        log_prob = action_distribution.log_prob(actions.squeeze(-1))
        return log_prob.view(1, -1)

    def update(self, states, actions, deltas):
        # Convert to tensors to use in the network
        deltas = torch.as_tensor(deltas, dtype=torch.float32)
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor([self.action_to_id[action] for action in actions])

        action_log_probs = self.evaluate_actions(states, actions)

        # Construct a loss function, using negative because we want to descend,
        # not ascend the gradient
        loss = -(action_log_probs * deltas).mean()
        self.optimiser.zero_grad()
        loss.backward()

        # Take a gradient descent step
        self.optimiser.step()
