"""
Custom Environments
"""

import gym
from gym import spaces
from gym.utils import seeding
from gym_minigrid.minigrid import MiniGridEnv, Grid, Ball, Box, Key
from envs.babyai_utils.verifier import *
from envs.babyai_utils.levelgen import *

from gym.envs.registration import register

import os
import cv2
import random
import numpy as np
from enum import IntEnum
from itertools import product, chain, combinations
from sympy.logic import boolalg
from sympy import sympify, Symbol


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def exp_goals(all_goals, exp):         
    def convert(exp):
        if type(exp) == Symbol:
            goals = set()  
            for goal in all_goals:
                if str(exp) in goal:
                    goals.add(goal)
            compound = goals
        elif type(exp) == boolalg.Or:
            compound = convert(exp.args[0])
            for sub in exp.args[1:]:
                compound = compound | convert(sub)
        elif type(exp) == boolalg.And:
            compound = convert(exp.args[0])
            for sub in exp.args[1:]:
                compound = compound & convert(sub)
        else: # NOT
            compound = convert(exp.args[0])
            compound = all_goals - compound
        return compound
    
    goals = list(convert(exp))
    return goals

class Actions(IntEnum):
    # Move forward, rotate right, rotate left 
    left = 0
    right = 1 
    forward = 2
    done = 3

class PickUpObjEnv(MiniGridEnv):    
    """
    Environment in which the agent needs to pickup a desired object.
    """

    OBJ_TYPES = ["key", "ball", "box"]
    OBJ_COLORS = ["red", "blue", "green", "purple", "yellow"]
    
    def __init__(self, exp=None, dist_type=None, dist_color=None, num_dists=1, size=7, corner_mod=False, xor_dists=False, seed=None):
        self.dist_type = dist_type
        self.dist_color = dist_color
        self.num_dists = num_dists
        self.r = -0.1
        self.rmin = -10
        self.rmax = 10
        self.done = False
        self.corner_mod = corner_mod
        self.xor_dists = xor_dists
  
        self.all_goals = list(product(self.OBJ_COLORS, self.OBJ_TYPES))
        self._all_goals = frozenset(self.all_goals)
        
        # exp = "red & ball"
        self.exp = exp
        if self.exp:
            self.mission = self.gen_mission(exp)

            exp = sympify(exp)
            exp = boolalg.simplify_logic(exp)
            goals = exp_goals(self._all_goals, exp)
            self.goals =  goals
        
        super().__init__(
            grid_size=size,
            max_steps=float('inf'),
        )
        
        # Modify action_space
        self.task_actions = Actions
        self.action_space = spaces.Discrete(len(self.task_actions))
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)
        
        # Room corners
        corners = [(1,1), (1,self.width-2), (self.width-2,self.width-2), (self.width-2,1)]
        pos = self._rand_int(0, 4)

        # Add agent
        self.place_agent()
        if self.corner_mod:
            self.agent_pos = corners[pos]

        # Task
        exp = self.exp
        if not exp:
            exp = self.gen_exp()
            self.mission = self.gen_mission(exp)

            exp = sympify(exp)
            exp = boolalg.simplify_logic(exp)
            goals = exp_goals(self._all_goals, exp)
            self.goals =  goals
        
        # Make sure there's at least one goal object
        if len(self.goals)>0:
            obj_color, obj_type = self._rand_elem(self.goals)
            obj, _ = self.add_object(obj_type, obj_color)
            if self.corner_mod:
                self.grid.set(*_, None)
                self.grid.set(*corners[(pos+2)%4], obj)      

        # Add distractors
        if not self.xor_dists:
            self.add_distractors()
        else:
            i = 0
            while i < self.num_dists:
                dist_type = self.dist_type if self.dist_type else self._rand_elem(self.OBJ_TYPES)
                dist_color = self.dist_color if self.dist_color else self._rand_elem(self.OBJ_COLORS)

                if (obj_type == dist_type or obj_color == dist_color) and not (obj_type == dist_type and obj_color == dist_color):
                    _ = self.add_object(dist_type, dist_color)
                    i += 1
    
    def gen_exp(self):
        exp = "{} & {}".format(self._rand_elem(self.OBJ_COLORS),self._rand_elem(self.OBJ_TYPES))
        return exp
    
    def gen_mission(self, exp):
        mission = exp.replace(" ","")
        mission = "pick up a " + mission
        mission = mission.replace("(","") 
        mission = mission.replace(")","")
        mission = mission.replace("&"," ")
        mission = mission.replace("|",", or a ")
        mission = mission.replace("!"," and not a ")
        mission = mission.strip() + "."
        return mission

    def add_object(self, type, color):
        if type=='box':
            obj = Box(color)
        if type=='ball':
            obj = Ball(color)
        if type=='key':
            obj = Key(color)
        pos = self.place_obj(obj)   
        return obj, pos
    
    def add_distractors(self):
        for _ in range(self.num_dists):
            color = self.dist_color if self.dist_color else self._rand_elem(self.OBJ_COLORS)
            type = self.dist_type if self.dist_type else self._rand_elem(self.OBJ_TYPES)
            if type=='box':
                obj = Box(color)
            if type=='ball':
                obj = Ball(color)
            if type=='key':
                obj = Key(color)
            pos = self.place_obj(obj)   
        
    def reset(self):
        obs = super().reset()
        self.done = False
        return obs

    def step(self, action):      
        if action == self.task_actions.done:
            action = self.actions.pickup                
        obs, _, _, info = super().step(action)
        reward = self.r
        
        # If successfully picked an object
        obj = self.carrying
        if obj:
            goal=(obj.color, obj.type)
            if goal in self.goals:
                reward = self.rmax
            else:
                reward = self.rmin
            self.done = True

        return obs, reward, self.done, info
    
class PickUpObjEnv8x8(PickUpObjEnv):    
    def __init__(self, *args, **kwargs):
        
        kwargs["num_dists"] = 1
        kwargs["size"] = 8
        super().__init__(*args, **kwargs)
    
class PickUpObjEnv12x12(PickUpObjEnv):    
    def __init__(self, *args, **kwargs):
        
        kwargs["num_dists"] = 9
        kwargs["size"] = 12
        super().__init__(*args, **kwargs)


register(
    id='MiniGrid-Custom-PickUpObj-8x8-v0',
    entry_point='envs.envs:PickUpObjEnv8x8'
)

register(
    id='MiniGrid-Custom-PickUpObj-12x12-v0',
    entry_point='envs.envs:PickUpObjEnv12x12'
)