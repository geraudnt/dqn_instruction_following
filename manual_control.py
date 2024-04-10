"""
Control the environment with keyboard
"""

import argparse
import numpy as np
import envs.envs
from envs.wrappers import *
from gym_minigrid.window import Window

class ManualControl:
    def __init__(
        self,
        env,
    ) -> None:
        self.env = env

        self.window = Window(args.env_key)
        self.window.reg_key_handler(self.key_handler)

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset()
        self.window.show(block=True)

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        print('reward=%.2f, action=%.2f' % (reward, action))

        if done:
            print("done!")
            self.reset()
        else:
            self.redraw(obs)

    def redraw(self, img):
        if not args.agent_view:
            img = self.env.render("rgb_array", highlight=False)
        else:
            img = img["image"]
        self.window.show_img(img)

    def reset(self):
        obs = self.env.reset()

        if hasattr(self.env, "mission"):
            print("Mission: %s" % self.env.mission)
            self.window.set_caption(self.env.mission)

        self.redraw(obs)

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.window.close()
            return
        if key == "backspace":
            self.reset()
            return

        if "Custom" in args.env_key:
            key_to_action = {
                "left": self.env.task_actions.left,
                "right": self.env.task_actions.right,
                "up": self.env.task_actions.forward,
                "enter": self.env.task_actions.done,
            }
        else:
            key_to_action = {
                "left": self.env.actions.left,
                "right": self.env.actions.right,
                "up": self.env.actions.forward,
                " ": self.env.actions.toggle,
                "pageup": self.env.actions.pickup,
                "pagedown": self.env.actions.drop,
                "enter": self.env.actions.done,
            }
        if key in key_to_action:
            action = key_to_action[key]
            self.step(action)


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

manual_control = ManualControl(env)
manual_control.start()