import sys
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from IPython import display

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from qfunction import QFunction
from ale_wrapper import ALEWrapper
from qlearning import QLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing


#version = "CartPole-v1"
version = "Freeway-ramDeterministic-v4"
policy_name = "Freeway.policy"
#version = "ALE/Frogger-ram-v5"
#policy_name = "Frogger-small.policy"
#version = "ALE/KingKong-ram-v5"
#version = "ALE/Riverraid-ram-v5"

env = ALEWrapper(version)
#env = gym.make(version)


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()


if (int(sys.argv[1]) == 1):
    extend_existing_policy = True
elif (int(sys.argv[1]) == 0):
    extend_existing_policy = False
else:
    print("Need to specify [0,1] whether to extend existing policy")
    sys.exit()

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'delta'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    


    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4

# Get number of actions from gym action space
#n_actions = env.action_space.n
n_actions = len(env.get_actions())
# Get the number of state observations
#state, info = env.get_initial_state()
state = env.get_initial_state()
n_observations = len(state)



class DQN(nn.Module, QFunction):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def get_q_value(self, state, action):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        action_tensor = torch.tensor(action, dtype=torch.long, device=device)
        
        with torch.no_grad():
            q_values = self.forward(state_tensor)
        q_value = q_values[action]
        #action = torch.tensor([action], device=device)
        #
        #q_value = self(state_tensor).gather(0, action.unsqueeze(1))[0]
        return q_value
    
    def get_max_q(self, state, actions):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            return (self(state_tensor).max(1).indices.view(1, 1).item(), None)

     
    def update(self, transitions):

        batch = Transition(*zip(*transitions))

        '''
        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=device)
        reward_batch = torch.as_tensor(batch.reward, device=device)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32, device=device)
        delta_batch = torch.tensor(batch.delta, dtype=torch.float32, device=device, requires_grad=True)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        next_state_values = target_net(next_state_batch).max(1).values
        deltas = (next_state_values * GAMMA) + reward_batch - state_action_values.squeeze(1)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        #print(deltas.unsqueeze(1) == delta_batch.unsqueeze(1))
        #print("deltas = " + str(deltas.unsqueeze(1)))
        #print("\n\n")
        #print("delta batch = " + str(delta_batch.unsqueeze(1)))
        #sys.exit()
        #loss = nn.functional.mse_loss(state_action_values, (state_action_values.squeeze(1) + deltas).unsqueeze(1))
        #loss = nn.functional.mse_loss(state_action_values, (state_action_values.squeeze(1) + delta_batch).unsqueeze(1))
        loss = nn.functional.mse_loss(delta_batch, torch.zeros_like(delta_batch))
        '''

        states_tensor = torch.tensor(batch.state, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(batch.action, dtype=torch.long, device=device)
        deltas_tensor = torch.tensor(batch.delta, dtype=torch.float32, device=device, requires_grad=True)
        rewards_tensor = torch.as_tensor(batch.reward, device=device)
        next_states_tensor = torch.tensor(batch.next_state, dtype=torch.float32, device=device)

        next_state_values = target_net(next_states_tensor).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + rewards_tensor

        # Compute Q-values for current states
        current_q_values = self.forward(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        #print("current =" + str(current_q_values))
        #print("\n\n")
        #print("expected_state_action = " + str(expected_state_action_values.unsqueeze(1)))
        #print("\n\n")
        #print("deltas = " + str(torch.neg(deltas_tensor)))
        #print("\n\n")
        #print("calculated deltas = " + str(current_q_values - expected_state_action_values.unsqueeze(1)))
        #print("\n\n")
        #print("calculated with deltas = " + str(current_q_values - ((current_q_values.squeeze(1) + deltas_tensor).unsqueeze(1))))
        #sys.exit()

        # Calculate the loss
        #loss = nn.functional.mse_loss(deltas_tensor, torch.zeros_like(deltas_tensor))
        #loss = nn.functional.mse_loss(current_q_values, (current_q_values.squeeze(1) + deltas_tensor).unsqueeze(1))

        #loss = nn.functional.mse_loss(deltas_tensor, torch.zeros_like(deltas_tensor))
        #below seems to learn something but inconsistently and stops learning after a few iterations (maybe one?)
        #loss = deltas_tensor.mean()
        #print(loss)
        #below works
        loss = nn.functional.mse_loss(current_q_values, expected_state_action_values.unsqueeze(1))
        #print(loss)
        #print("\n\n")

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        optimizer.step()

    def get_max_q_values(self, states):
        state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        with torch.no_grad():
            return self(state_tensor).max(1).values

    def soft_update(self, policy_network, tau=0.005):
        target_dict = self.state_dict()
        policy_dict = policy_network.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
        self.load_state_dict(target_dict)


def plot_rewards(episode_rewards, show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(rewards_t.numpy())
    # Take episode averages and plot them over a window
    window = 25
    if len(rewards_t) >= window:
        means = rewards_t.unfold(0, window, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(window - 1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


class ExperienceReplay(QLearning):

    def __init__(self, env, bandit):
        self.env = env
        self.bandit = bandit

    def execute(self, episodes):

        episode_rewards = []

        for _ in range(episodes):
            # Initialize the environment and get it's state
            state = env.get_initial_state()
            episode_reward = 0

            for t in count():
                actions = env.get_actions()
                action = self.bandit.select(state, actions, policy_net)
                next_state, reward, done = env.execute(state, action)
                #q_value = policy_net.get_q_value(state, action)
                #delta = self.get_delta(reward, q_value, state, next_state)

                delta = 0.0
                # Store the transition in memory
                #memory.push(state, action, next_state, reward, not done)
                memory.push(state, action, next_state, reward, delta)
                episode_reward += reward

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if len(memory) >= BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    policy_net.update(transitions)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net.soft_update(policy_net)

                if done:
                    break

            episode_rewards.append(episode_reward)
            torch.save(policy_net.state_dict(), policy_name)
            #plot_rewards(episode_rewards)

        return episode_rewards
    
    def get_delta(self, reward, q_value, state, next_state):
        next_state_value = target_net.get_max_q_values([next_state])
        delta = reward + GAMMA * next_state_value - q_value
        return delta

if extend_existing_policy:
    policy_net.load_state_dict(torch.load(policy_name))
    target_net.load_state_dict(policy_net.state_dict())

    print("loading existing policy " + policy_name)


import numpy as np
import time

learner = ExperienceReplay(env, EpsilonDecreasing())
start = time.time()
episode_rewards = learner.execute(episodes=30)
end = time.time()
print(("{:.2f}, {:.2f}, " + str(episode_rewards)).format(np.mean(episode_rewards), end - start))

#learner = ExperienceReplay(env, EpsilonDecreasing())
#episode_rewards = learner.execute(episodes=50)

print('Complete')
sys.exit()
plot_rewards(episode_rewards, show_result=True)
plt.ioff()
plt.show()

policy_net.load_state_dict(torch.load(policy_name))

env = ALEWrapper(version, render_mode="human")
bandit = EpsilonGreedy(epsilon=0.01)
state = env.get_initial_state()
done = False
while not done:
    actions = env.get_actions()
    action = bandit.select(state, actions, policy_net)
    observation, reward, done = env.execute(state, action)
    
    # Move to the next state
    state = observation
