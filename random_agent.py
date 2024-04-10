"""
Save gif of a random agent
"""

import argparse
import numpy as np
import envs.envs
from envs.wrappers import *
import imageio
from matplotlib import pyplot as plt


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
    default=9,
    help="Number of distractors"
)
parser.add_argument(
    '--size',
    type=int,
    default=12,
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
    '--agent_view',
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
env = RGBImgObsWrapper(env, tile_size=32, obs_size=args.obs_size) # Wrapper for pixel observations


def fig_image(fig, pad=0, h_pad=None, w_pad=None, rect=(0,0,1,1)):
    # fig.tight_layout(pad=-0.3,h_pad=h_pad,w_pad=w_pad)
    fig.gca().margins(0)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

images = []
for _ in range(5):
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

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

        if done:
            break

imageio.mimsave("images/random_agent.gif",images,fps=10)