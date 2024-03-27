import torch
import torch.nn as nn
from qfunction import QFunction
from torch.optim import Adam
from deep_agent import DeepAgent

class DeepQFunction(QFunction, DeepAgent):
    """ A neural network to represent the Q-function.
        This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(
        self, mdp, state_space, action_space, hidden_dim=64, alpha=0.001
    ) -> None:

        # Create a sequential neural network to represent the Q function
        self.q_network = nn.Sequential(
            nn.Linear(in_features=state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=action_space),
        )
        '''
        self.q_network = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),  # First hidden layer
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 32),   # Second hidden layer
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, action_space)
        )  
         ''' 
        self.optimiser = Adam(self.q_network.parameters(), lr=alpha)

        # A two-way mapping from actions to integer IDs for ordinal encoding
        actions = mdp.get_actions()
        self.action_to_id = {actions[i]: i for i in range(len(actions))}
        self.id_to_action = {
            action_id: action for action, action_id in self.action_to_id.items()
        }

    def update(self, state, action, delta):

        # Train the network based on the squared error.
        self.optimiser.zero_grad()  # Reset gradients to zero
        (delta ** 2).backward()  # Back-propagate the loss through the network
        self.optimiser.step()  # Do a gradient descent step with the optimiser

        #self.multi_update([(state, action, delta)])

    def multi_update(self, experiences):
        (states, actions, deltas, dones) = zip(*experiences)
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        deltas_tensor = torch.tensor(deltas, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)

        q_values = self.q_network(states_tensor).gather(dim=1, index=actions_tensor.unsqueeze(1))

        loss = nn.functional.mse_loss(q_values, q_values + deltas_tensor.unsqueeze(1))
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def get_q_value(self, state, action):
        # Convert the state into a tensor
        state = self.encode_state(state)
        q_values = self.q_network(state)

        # Index q-values by action
        q_value = q_values[self.action_to_id[action]]

        return q_value

    def get_max_q(self, state, actions):
        # Convert the state into a tensor
        state = torch.as_tensor(self.encode_state(state), dtype=torch.float32)

        # Since we have a multi-headed q-function, we only need to pass through the network once
        # call torch.no_grad() to avoid tracking the gradients for this network forward pass
        with torch.no_grad():
            q_values = self.q_network(state)
        arg_max_q = None
        max_q = float("-inf")
        for action in actions:
            value = q_values[self.action_to_id[action]].item()
            if max_q < value:
                arg_max_q = action
                max_q = value
        return (arg_max_q, max_q)

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))