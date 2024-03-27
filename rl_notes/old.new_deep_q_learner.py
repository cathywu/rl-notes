import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from qfunction import QFunction
from ale_wrapper import ALEWrapper
from collections import namedtuple

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the experience namedtuple
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

# Q-learning agent
class DeepQFunction(QFunction):

    
    def __init__(self, input_size, output_size, hidden_size, lr, gamma):
        self.q_network = QNetwork(input_size, output_size, hidden_size)
        self.target_network = QNetwork(input_size, output_size, hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(range(self.q_network.fc2.out_features))
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state))
                return torch.argmax(q_values).item()

    def update(self, experiences):
        batch = Experience(*zip(*experiences))

        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action)
        next_state_batch = torch.stack(batch.next_state)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        done_batch = torch.tensor(batch.done, dtype=torch.float32)

        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        #target_q_values = reward_batch + self.gamma * next_q_values

        loss = nn.functional.mse_loss(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Main training loop
class DeepQLearner():

    def __init__(self, env, hidden_size=128, lr=0.001, gamma=0.99):
        input_size = len(env.get_initial_state())
        output_size = len(env.get_actions())
        
        self.qfunction = DeepQFunction(input_size, output_size, hidden_size, lr, gamma)
    
    def execute(self, env, num_episodes, epsilon_decay, hidden_size=128, lr=0.001, gamma=0.99):
        
        epsilon = 1.0

        for episode in range(num_episodes):
            state = env.get_initial_state()
            episode_reward = 0

            while True:
                action = self.qfunction.select_action(state, epsilon)
                next_state, reward, done = env.execute(state, action)

                experience = Experience(
                    state=torch.FloatTensor(state),
                    action=action,
                    next_state=torch.FloatTensor(next_state),
                    reward=reward,
                    done=done
                )

                self.qfunction.update([experience])
                episode_reward += reward

                if done:
                    break

                state = next_state

            epsilon = max(epsilon * epsilon_decay, 0.01)  # Decay epsilon over time
            self.qfunction.update_target_network()

            if episode % 10 == 0:
                print(f"Episode: {episode}, Epsilon: {epsilon}, Episode Reward: {episode_reward}")

            torch.save(self.qfunction.target_network.state_dict(), "frogger.policy")

        env.close()
        return self.qfunction

    def test_dqn(self, env_name, qfunction, epsilon=0.0):
        env = gym.make(env_name, render_mode="human")
        state = env.get_initial_state()
        episode_reward = 0

        qfunction.target_network.load_state_dict(torch.load("frogger.policy"))

        while True:
            action = qfunction.select_action(state, epsilon)
            next_state, reward, done = env.execute(state, action)
            episode_reward += reward

            if done:
                break
            state = next_state

        #env.close()
        return episode_reward

# Example usage
if __name__ == "__main__":
    #env_name ="ALE/Frogger-ram-v5"
    env_name = 'Freeway-ramDeterministic-v4'  # Change to the desired Atari RAM game
    num_episodes = 200
    epsilon_decay = 0.995
    hidden_size = 128
    learning_rate = 0.0001
    discount_factor = 0.99

    env = ALEWrapper(version=env_name, render_mode="rgb_array", discount_factor=1.0)
    #qfunction = DeepQFunction(env, hidden_size, learning_rate, discount_factor)
    qfunction = DeepQLearner(env, hidden_size, learning_rate, discount_factor).execute(env, num_episodes, epsilon_decay, hidden_size, learning_rate, discount_factor)
    #env = gym.make(env_name)
    #input_size = env.observation_space.shape[0]
    #output_size = env.action_space.n
    #hidden_size = 128
    #agent = DQNAgent(input_size, output_size, hidden_size, 0, 0)
    input("Press Enter to continue...")
    DeepQLearner().test_dqn(env, qfunction)
