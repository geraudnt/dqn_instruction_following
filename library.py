import numpy as np
import random
import os
import re
import hashlib
from multiprocessing import Process, Pipe

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


class LinearSchedule(object):
    """
    Linear schedule for exploration decay
    """
    
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def __call__(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size, batch_size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.batch_size = batch_size

    def __len__(self):
        return len(self._storage)

    def add(self, mission, state, action, reward, next_state, done, info):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param mission: the agent's task instruction
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        data = (mission, state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        missions, states, actions, rewards, next_states, dones = [], [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            mission, state, action, reward, next_state, done = data
            missions.append(np.array(mission, copy=False))
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return np.array(missions), np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def sample(self):
        """
        Randomly sample a batch of transitions from the buffer.
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage) - 1, size=self.batch_size)
        return self._encode_sample(indices)


class DQN(nn.Module):
    """
    DQN for Pixel observations and language instructions
    """

    def __init__(self, env):
        super(DQN, self).__init__()
        self.action_space = env.action_space
        self.observation_space = env.observation_space.spaces['image']
        
        ### Image embedding, Architecture from DQN nature paper 
        l, _, _ = self.observation_space.shape
        k1,s1 = (8,4); k2,s2 = (4,2); k3,s3 = (3,1); c_out = 64
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=k1, stride=s1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=k2, stride=s2),
            nn.ReLU(),
            nn.Conv2d(64, c_out, kernel_size=k3, stride=s3),
            nn.ReLU()
        )
        f = lambda l,k,s: (l-k)//s + 1
        self.embedding_size = f(f(f(l,k1,s1),k2,s2),k3,s3)**2*c_out
    
        ### Mission embedding
        self.vocab_max_size = 100
        self.word_embedding_size = 32
        self.word_embedding = nn.Embedding(self.vocab_max_size, self.word_embedding_size)
        self.text_embedding_size = self.embedding_size # If image and text embeddings are multiplied
        # self.text_embedding_size = 512
        self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        ### Q_values
        # self.embedding_size += self.text_embedding_size # If image and text embeddings are concatenated
        self.linear1 = nn.Linear(self.embedding_size, 512)
        self.head = nn.Linear(512, self.action_space.n)

    def forward(self, obs, mission):
        obs = obs.permute(0, 3, 1, 2)
        obs = self.image_conv(obs)
        image_embedding = obs.reshape(obs.size(0), -1)

        mission = self.word_embedding(mission)
        _, hidden = self.text_rnn(mission)
        text_embedding = hidden[-1]

        ### Combinining image and text embeddings
        embedding = image_embedding*text_embedding
        # embedding = torch.cat((image_embedding, text_embedding), dim=1)

        y = F.relu(self.linear1(embedding))
        y = self.head(y)
        return y.squeeze()


def tokenize(text,vocab):
    max_length = 16
    tokens = re.findall("([a-z]+)", text.lower())
    var_indexed_text = np.array([vocab[token] for token in tokens])
    tokenized = np.zeros(max_length)
    tokenized[:len(var_indexed_text)] = var_indexed_text
    return tokenized

class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
    
def load(path, env):
    if torch.cuda.is_available():
        model = torch.load(path)
    else:
        model = torch.load(path, map_location=torch.device('cpu'))
    vocab = Vocabulary(100)
    vocab.load_vocab(model['vocab'])
    model['vocab'] = vocab
    dqn = DQN(env)
    dqn.load_state_dict(model['params'])
    model['params'] = dqn
    if torch.cuda.is_available():
        model['params'].cuda()
    return model


def select_action(model,obs,mission):
    obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
    mission = torch.from_numpy(mission).type(LongTensor).unsqueeze(0)
    values = model['params'](obs, mission)
    action = values.squeeze().max(0)[1].item()
    return action


class Agent(object):
    def __init__(self,
                 env,
                 path='',
                 gamma=0.95,
                 learning_rate=1e-4,
                 replay_buffer=None):

        self.path = path
        self.gamma = gamma
        self.replay_buffer = replay_buffer
        self.vocab = Vocabulary(100)
        
        self.q_func = DQN(env)
        self.target_q_func = DQN(env)
        self.target_q_func.load_state_dict(self.q_func.state_dict())

        if use_cuda:
            self.q_func.cuda()
            self.target_q_func.cuda()

        self.optimizer = optim.Adam(self.q_func.parameters(), lr=learning_rate)
        
    def update_td_loss(self):
        missions_batch, obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample()
        missions_batch = Variable(torch.from_numpy(missions_batch).type(LongTensor))
        obs_batch = Variable(torch.from_numpy(obs_batch).type(FloatTensor))
        act_batch = Variable(torch.from_numpy(act_batch).type(LongTensor))
        rew_batch = Variable(torch.from_numpy(rew_batch).type(FloatTensor))
        next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(FloatTensor))
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(FloatTensor)

        if use_cuda:
            act_batch = act_batch.cuda()
            rew_batch = rew_batch.cuda()

        current_q_values = self.q_func(obs_batch, missions_batch).gather(1, act_batch.unsqueeze(1)).squeeze()
        next_max_q = self.target_q_func(next_obs_batch, missions_batch).detach().max(1)[0]
        next_q_values = not_done_mask * next_max_q
        target_q_values = rew_batch + (self.gamma * next_q_values)

        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        for params in self.q_func.parameters():
            params.grad.data.clamp_(-1, 1)
        self.optimizer.step()   

    def update_target_network(self):
        self.target_q_func.load_state_dict(self.q_func.state_dict())

    def save(self):
        model = {'params': self.q_func.state_dict(), 'vocab': self.vocab.vocab}
        torch.save(model, self.path)