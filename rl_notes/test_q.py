import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DeepQFunction:
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        learning_rate=0.001,
        soft_update_tau=0.01,
    ):
        self.q_network = QNetwork(input_size, output_size, hidden_size)
        self.target_q_network = QNetwork(input_size, output_size, hidden_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.soft_update_tau = soft_update_tau

    def update(self, experiences, gamma=0.99):
        states, actions, rewards, next_states, dones = experiences

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_q_network(next_states).max(1)[0].detach()

        target_q_values = rewards + (1 - dones) * gamma * next_q_values

        loss = nn.SmoothL1Loss()(q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_q_value(self, state, action):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        return self.q_network(state).gather(0, action.unsqueeze(0))

    def get_max_q_value(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self.q_network(state).max().item()

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def soft_update(self):
        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                (1 - self.soft_update_tau) * target_param.data
                + self.soft_update_tau * param.data
            )


# Example usage:
input_size = 4  # Assuming state size
output_size = 2  # Assuming number of actions
hidden_size = 64
learning_rate = 0.001
soft_update_tau = 0.01

deep_q_function = DeepQFunction(
    input_size, output_size, hidden_size, learning_rate, soft_update_tau
)
