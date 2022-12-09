"""
Visualising Learned Tasks
"""

import torch
import numpy as np
import os
import itertools

from library import *
import envs.envs
from envs.wrappers import *
from matplotlib import pyplot as plt
from gym_minigrid.window import Window

if __name__ == '__main__':
    env_key="Minigrid-PickUpObj-Custom-v0"
    env = gym.make(env_key) # gym.make(env_key, num_dists=9, size=11)
    env = FullyObsWrapper(env, egocentric=True) # Wrapper for egocentric full observations
    env = RGBImgObsWrapper(env) # Wrapper for pixel observations
    # env = RGBImgObsWrapper(env, obs_size=84) # Use obs_size=84 normally as in the DQN nature paper
    path='models/{}'.format(env_key)
    window = Window(env_key)
      
    print('Loading ...')
    model = load(path, env)

    print('Visualizing ...')        
    max_episodes = 50000
    max_trajectory = 20
    with torch.no_grad():
        for episode in range(max_episodes):
            obs = env.reset()
            mission = tokenize(obs['mission'], model['vocab'])
            window.set_caption(obs['mission'])
            
            done = False
            for _ in range(max_trajectory):
                window.show_img(env.render('rgb_array'))
                action = select_action(model,obs['image'],mission)
                obs, reward, done, _ = env.step(action)
                
                if done or window.closed:
                    break
            if done:
                print("Mission success.\n")
            else:
                print("Mission failed.\n")
            
            if window.closed:
                break

