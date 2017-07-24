# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# Creating the architecture of the Neural Network

class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


# Implementing Experience Replay

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        # (state1, state2), (action1, action2), (reward1, reward2)...
        samples = zip(*random.sample(self.memory, batch_size))
        # torch.cat aligns everything as (state, action, reward)
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implementing Deep Q Learning

class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # Vector of dimension 6. The 3 signals of the 3 sensors and orientation/ - orientation
        # (batch_size, left, straight, right, orientation, -orientation)
        # batch_size dimension is added with unsqueeze
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        # 0, 1 or 2 (indexed), see the var action2rotation in map.py
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        """
        Take the input state (the 3 Q-values) and return the best possible action
        :param state: The input state of the neural net (left, straight, right, orientation, -orientation)
        :return: The best possible action probabilities
        """
        # volatile = True : We don't want the gradient in the graph of all the computation of the nn module
        # 100 is the temperature parameter or the certainty about the next action to play
        # The closer it is to 0 the less sure the nn will be
        # to take the action. Far from 0, the more sure it will be about the action to play
        # ex with T = 3: softmax([0.04, 0.11, 0.85]) => softmax([1,2,3] * 3) = [0, 0.02, 0.98]
        probs = F.softmax(self.model(Variable(state, volatile=True)) * 100)  # T=100
        # random draw from the probabilities
        action = probs.multinomial()
        # Retrieve the action at index [0, 0]
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        """
        Train the nn
        :param batch_state:
        :param batch_next_state:
        :param batch_reward:
        :param batch_action:
        :return:
        """
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        # td = temporal difference
        td_loss = F.smooth_l1_loss(outputs, target)
        # Reinitialize the Adam optimizer from the constructor
        self.optimizer.zero_grad()
        # Backprop, retain_variables=True to free the memory
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        # The new states are the current signals
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(
            (self.last_state, new_state,
             torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        # +1 to avoid dividing by 0
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
