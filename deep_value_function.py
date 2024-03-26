import torch.nn as nn
from value_function import ValueFunction
from torch.optim import Adam
from deep_agent import DeepAgent


class DeepValueFunction(ValueFunction, DeepAgent):
    """
    A neural network to represent the Value-function.
    This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(
            self, mdp, state_space, hidden_dim=64, alpha=0.001
    ) -> None:
        self.mdp = mdp
        self.state_space = state_space
        self.alpha = alpha

        # Create a sequential neural network to represent the Q function
        self.value_network = nn.Sequential(
            nn.Linear(in_features=self.state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )
        self.optimiser = Adam(self.value_network.parameters(), lr=self.alpha)

    def update(self, state, delta):
        self.optimiser.zero_grad()
        (delta ** 2).backward()  # Back-propagate the loss through the network
        self.optimiser.step()  # Do a gradient descent step with the optimiser

    def get_value(self, state):
        # pass through the network to get the value
        state = self.encode_state(state)
        value = self.value_network(state)
        return value
