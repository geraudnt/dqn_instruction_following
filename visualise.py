"""
Visualising Learned Tasks
"""

import torch
import numpy as np
import os
import itertools

import argparse
from library import *
import envs.envs
from envs.wrappers import *
from matplotlib import pyplot as plt
from gym_minigrid.window import Window
import imageio

parser = argparse.ArgumentParser()
parser.add_argument(
    '--env_key',
    default="Minigrid-PickUpObj-Custom-v0",
    help="Environment"
)
parser.add_argument(
    '--exp',
    default=None,
    help="Task expression"
)
parser.add_argument(
    '--num_dists',
    type=int,
    default=1,
    help="Number of distractors"
)
parser.add_argument(
    '--size',
    type=int,
    default=7,
    help="Grid size"
)
parser.add_argument(
    '--obs_size',
    type=int,
    default=None,
    help="Observation size"
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=None
)
parser.add_argument(
    '--save',
    default=False,
    help="draw what the agent sees",
    action='store_true'
)

args = parser.parse_args()
if "Custom" in args.env_key:
    env = gym.make(args.env_key, exp=args.exp, num_dists=args.num_dists, size=args.size, seed=args.seed)
else:
    env = gym.make(args.env_key)
env = FullyObsWrapper(env, egocentric=True) # Wrapper for egocentric full observations
env = RGBImgObsWrapper(env, obs_size=args.obs_size) # Wrapper for pixel observations


def fig_image(fig, pad=0, h_pad=None, w_pad=None, rect=(0,0,1,1)):
    # fig.tight_layout(pad=-0.3,h_pad=h_pad,w_pad=w_pad)
    fig.gca().margins(0)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


if __name__ == '__main__':
    path='models/{}'.format(args.env_key)
      
    print('Loading ...')
    model = load(path, env)

    if args.save:
        print('Saving video ...')    
        images = []
    else:
        print('Visualizing ...')     
        window = Window(args.env_key)   
    max_episodes = 4
    max_trajectory = 20
    with torch.no_grad():
        for episode in range(max_episodes):
            obs = env.reset()
            mission = tokenize(obs['mission'], model['vocab'])
            if not args.save:
                window.set_caption(obs['mission'])
            
            done = False
            for _ in range(max_trajectory):
                if not args.save:
                    window.show_img(env.render('rgb_array'))
                else:
                    image_allocentric = env.render("rgb_array", highlight=False)
                    image_egocentric = obs["image"]

                    fig, axs = plt.subplots(1, 2)
                    axs = axs.flatten()

                    axs[0].set_title("Allocentric obs", fontsize=20)
                    axs[0].set_xticks([])
                    axs[0].set_yticks([])
                    axs[0].imshow(image_allocentric)

                    axs[1].set_title("Egocentric obs", fontsize=20)
                    axs[1].set_xticks([])
                    axs[1].set_yticks([])
                    axs[1].imshow(image_egocentric)

                    fig.suptitle("Mission: "+obs["mission"], fontsize=20)
                    fig.tight_layout()
                    fig.subplots_adjust(top=1)
                    images.append(fig_image(fig))
                action = select_action(model,obs['image'],mission)
                obs, reward, done, _ = env.step(action)
                
                if not args.save and window.closed:
                    break

                if done:
                    break
            if done:
                print("Mission success.\n")
            else:
                print("Mission failed.\n")
            
            if not args.save and window.closed:
                break
    if args.save:
        imageio.mimsave("images/trained_agent_{}.gif".format(args.env_key),images,fps=10)

