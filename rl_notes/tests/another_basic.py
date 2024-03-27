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
from experience_replay_learner import ExperienceReplayLearner
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing


# version = "CartPole-v1"
#version = "Freeway-ramDeterministic-v4"
#policy_name = "Freeway.policy"
version = "ALE/Frogger-ram-v5"
policy_name = "Frogger-small.policy"
# version = "ALE/KingKong-ram-v5"
# version = "ALE/Riverraid-ram-v5"

env = ALEWrapper(version)
# env = gym.make(version)


# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()

if len(sys.argv) < 1:
    print("Need to specify [0,1] whether to extend existing policy")
    sys.exit()
elif int(sys.argv[1]) == 1:
    extend_existing_policy = True
elif int(sys.argv[1]) == 0:
    extend_existing_policy = False
else:
    print("Need to specify [0,1] whether to extend existing policy")
    sys.exit()

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


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
# n_actions = env.action_space.n
n_actions = len(env.get_actions())
# Get the number of state observations
# state, info = env.get_initial_state()
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
        # action = torch.tensor([action], device=device)
        #
        # q_value = self(state_tensor).gather(0, action.unsqueeze(1))[0]
        return q_value

    def get_max_q(self, state, actions):
        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            return (self(state_tensor).max(1).indices.view(1, 1).item(), None)

    def update(self, transitions):

        batch = Transition(*zip(*transitions))

        states_tensor = torch.tensor(batch.state, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(batch.action, dtype=torch.long, device=device)
        # deltas_tensor = torch.tensor(batch.delta, dtype=torch.float32, device=device, requires_grad=True)
        rewards_tensor = torch.as_tensor(batch.reward, device=device)
        next_states_tensor = torch.tensor(
            batch.next_state, dtype=torch.float32, device=device
        )

        next_state_values = target_net(next_states_tensor).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + rewards_tensor

        # Compute Q-values for current states
        current_q_values = self.forward(states_tensor).gather(
            1, actions_tensor.unsqueeze(1)
        )

        # Calculate the loss
        loss = nn.functional.mse_loss(
            current_q_values, expected_state_action_values.unsqueeze(1)
        )

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

    def soft_update(self, policy_qfunction, tau=0.005):
        target_dict = self.state_dict()
        policy_dict = policy_qfunction.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
        self.load_state_dict(target_dict)


def plot_rewards(episode_rewards, show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
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
    def __init__(self, mdp, bandit, policy_qfunction, target_qfunction):
        self.mdp = mdp
        self.bandit = bandit
        self.policy_qfunction = policy_qfunction
        self.target_qfunction = target_qfunction

    def execute(self, episodes):

        episode_rewards = []

        for _ in range(episodes):
            # Initialize the environment and get it's state
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.policy_qfunction)

            episode_reward = 0.0

            step = 0
            while not self.mdp.is_terminal(state):
                (next_state, reward, done) = self.mdp.execute(state, action)

                memory.push(state, action, next_state, reward)

                actions = self.mdp.get_actions()
                next_action = self.bandit.select(state, actions, self.policy_qfunction)

                # Perform one step of the optimization (on the policy network)
                if len(memory) >= BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    self.policy_qfunction.update(transitions)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                self.target_qfunction.soft_update(self.policy_qfunction)

                # Move to the next state
                state = next_state
                action = next_action
                episode_reward += reward
                step += 1

            episode_rewards.append(episode_reward)
            torch.save(self.policy_qfunction.state_dict(), policy_name)
            plot_rewards(episode_rewards)

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

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def get_capacity(self):
        return self.capacity

    def __len__(self):
        return len(self.memory)

from temporal_difference_learner import TemporalDifferenceLearner
class ExperienceReplayLearner(TemporalDifferenceLearner):
    def __init__(self, mdp, bandit, policy_qfunction, target_qfunction, 
                 replay_buffer=ReplayBuffer(10000), alpha=0.001, replay_period=1000, batch_size=64):
        super().__init__(mdp, bandit, policy_qfunction, alpha=alpha)
        self.replay_buffer = replay_buffer
        self.replay_period = replay_period
        self.policy_qfunction = policy_qfunction
        self.target_qfunction = target_qfunction
        self.batch_size = batch_size

    def execute(self, episodes=100):

        rewards = []
        for _ in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.policy_qfunction)

            episode_reward = 0.0
            step = 0
            while not self.mdp.is_terminal(state):
                (next_state, reward, done) = self.mdp.execute(state, action)
                actions = self.mdp.get_actions(next_state)
                next_action = self.bandit.select(next_state, actions, self.policy_qfunction)
                q_value = self.policy_qfunction.get_q_value(state, action)
                next_state_value = self.state_value(next_state, next_action)

                #delta = self.get_delta(reward, q_value, state, next_state, next_action)

                #experience = (state, action, delta, done)

                self.replay_buffer.push(state, action, next_state, reward)
                if len(self.replay_buffer) >= self.replay_buffer.get_capacity():
                    transitions = self.replay_buffer.sample(self.batch_size)
                    self.policy_qfunction.update(transitions)

                self.target_qfunction.soft_update(self.policy_qfunction)

                state = next_state
                action = next_action
                episode_reward += reward * (self.mdp.discount_factor ** step)
                step += 1

            rewards.append(episode_reward)
            #plot_rewards(rewards)

        return rewards

    """ Update from a mini batch """
    def update(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        self.policy_qfunction.multi_update(mini_batch)

    def state_value(self, state, action):
        (_, max_q_value) = self.target_qfunction.get_max_q(state, self.mdp.get_actions(state))
        return max_q_value

def main():

    start = time.time()
    learner = ExperienceReplay(
        env,
        EpsilonDecreasing(),
        policy_qfunction=policy_net,
        target_qfunction=target_net,
    )
    learner = ExperienceReplayLearner(
        env,
        EpsilonDecreasing(),
        policy_qfunction=policy_net,
        target_qfunction=target_net,
    )
    episode_rewards = learner.execute(episodes=300)
    end = time.time()
    print(
        ("({:.2f}, {:.2f}, " + str(episode_rewards) + "), ").format(
            np.mean(episode_rewards), end - start
        )
    )
    sys.exit()
    print("Complete")
    plot_rewards(episode_rewards, show_result=True)
    plt.ioff()
    plt.show()
    
    policy_net.load_state_dict(torch.load(policy_name))

    mdp = ALEWrapper(version, render_mode="human")
    bandit = EpsilonGreedy(epsilon=0.01)
    state = mdp.get_initial_state()
    done = False
    while not done:
        actions = mdp.get_actions()
        action = bandit.select(state, actions, policy_net)
        observation, reward, done = mdp.execute(state, action)

        # Move to the next state
        state = observation


import cProfile

if __name__ == "__main__":
    main()
    #cProfile.run("main()", sort="cumulative")
