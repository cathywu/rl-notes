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
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

policy_name = "Frogger.policy"

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
                        ('state', 'action', 'next_state', 'reward'))


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
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9 if extend_existing_policy else 0.2
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
#n_actions = env.action_space.n
n_actions = len(env.get_actions())
# Get the number of state observations
#state, info = env.get_initial_state()
state = env.get_initial_state()
n_observations = len(state)




steps_done = 0


class DQN(nn.Module, QFunction):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.update_time = 0.0

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def select_action(self, state, actions):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self(state).max(1).indices.view(1, 1).item()
        else:
            return torch.tensor([[random.choice(env.get_actions())]], device=device, dtype=torch.long)


    def update(self, transitions):

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)

        #reward_batch = torch.cat(batch.reward)
        reward_batch = torch.as_tensor(batch.reward, device=device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        import time 
        start = time.time()
        print(batch.action)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=device)
        state_action_values = self(state_batch).gather(1, action_batch.unsqueeze(1))
        end = time.time()
        self.update_time +=  (end-start)


        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        optimizer.step()

    def get_q_value(self, state, action):
        q_values = self(state)

        action = torch.cat([action])
        q_value = self(state).gather(1, action)[0]
        return q_value
    
    def get_max_q(self, state, action):
        with torch.no_grad():
            return (self(state).max(1).indices.view(1, 1), None)

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



if torch.cuda.is_available():
    episodes = 50
else:
    episodes = 20

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


class ExperienceReplay:

    def __init__(self, env):
        self.env = env
        self.bandit = EpsilonGreedy(epsilon=EPS_START)

    def execute(self, episodes):

        episode_rewards = []

        for i_episode in range(episodes):
            # Initialize the environment and get it's state
            state = env.get_initial_state()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = 0
            for t in count():
                actions = env.get_actions()
                #eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                #    math.exp(-1. * steps_done / EPS_DECAY)
                #self.bandit = EpsilonGreedy(epsilon=eps_threshold)
                #print((state, actions))
                #action = self.bandit.select(state, actions, policy_net)
                #actions = torch.tensor([actions], device=device, dtype=torch.long)
                action = policy_net.select_action(state, actions)

                observation, reward, done = env.execute(state, action)
                #reward = torch.tensor([reward], device=device)
                #done = terminated or truncated

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)
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
                #target_net_state_dict = target_net.state_dict()
                #policy_net_state_dict = policy_net.state_dict()
                #for key in policy_net_state_dict:
                #    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                #target_net.load_state_dict(target_net_state_dict)

                if done:
                    break

            episode_rewards.append(episode_reward)
            torch.save(policy_net.state_dict(), policy_name)
            #plot_rewards(episode_rewards)

        return episode_rewards

if extend_existing_policy:
    EPS_START = 0.3
    policy_net.load_state_dict(torch.load(policy_name))
    target_net.load_state_dict(policy_net.state_dict())

    print("loading existing policy " + policy_name)


import statistics
for i in range(30):
    learner = ExperienceReplay(env)

    episode_rewards = learner.execute(episodes=10)
    execute(num_episodes=50)
    print(episode_rewards)
    print(statistics.mean(episode_rewards))

print("update time = " + str(policy_net.update_time))

print('Complete')
#plot_rewards(episode_rewards, show_result=True)
plt.ioff()
plt.show()

policy_net.load_state_dict(torch.load(policy_name))

env = ALEWrapper(version)
state = env.get_initial_state()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
for t in count():
    actions = env.get_actions()
    action = policy_net.select_action(state, actions)
    observation, reward, terminated, truncated, _ = env.execute(state, action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    if terminated:
        next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    # Move to the next state
    state = next_state
    if done:
        break
