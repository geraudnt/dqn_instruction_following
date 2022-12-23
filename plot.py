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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--env_key',
    default="Minigrid-Custom-PickUpObj-12x12-v0", # "MiniGrid-DoorKey-16x16-v0"
    help="Environment"
)

if __name__ == '__main__':
    env = gym.make(args.env_key)
    if "Minigrid" in args.env_key:
        env = FullyObsWrapper(env) # Wrapper for Minigrid egocentric full observations
        env = RGBImgObsWrapper(env) # Wrapper for Minigrid pixel observations
    path='models/{}'.format(args.env_key)
    window = Window(args.env_key)
      
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

