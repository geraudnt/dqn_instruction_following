"""
Useful Wrappers
"""

import gym
from gym import spaces
import numpy as np
import cv2
from collections import deque
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, Goal


class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
        Default: Regular topdown view
        Optional: Egocentric topdown view
    """

    def __init__(self, env, egocentric=False):
        super().__init__(env)

        self.egocentric = egocentric
        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):        
        env = self.unwrapped
        full_grid = env.grid.encode()     
        # full_grid[:,:,2] = OBJECT_TO_IDX["wall"]*(full_grid[:,:,0]==OBJECT_TO_IDX["wall"])

        if not self.egocentric:
            rgb_img = full_grid
            y, x = self.agent_pos
            rgb_img[y, x, :] = (OBJECT_TO_IDX["agent"], 0, self.agent_dir) 
        else:       
            s = full_grid.shape[0]
            y, x = self.agent_pos

            # Egocentric rotation
            agent_pos = full_grid[:,:,0]*0
            agent_pos[y,x] = 1
            k = 3 - self.agent_dir
            agent_pos = np.rot90(agent_pos, k=k)
            for i in range(3):
                full_grid[:,:,i] = np.rot90(full_grid[:,:,i], k=k)        
            x, y = np.where(agent_pos==1)
            x, y = x[0], y[0]

            # Egocentric position
            ox = s//2-x    
            rgb_img = full_grid.copy()
            if ox>=0:
                rgb_img[ox:s//2,:,:] = full_grid[:x,:,:]    
                rgb_img[s//2:,:,:] = full_grid[x:x+s//2+s%2,:,:]   
                rgb_img[:ox,:,:] = full_grid[x+s//2+s%2:,:,:]   
            else:
                ox = s+ox
                rgb_img[s//2:ox,:,:] = full_grid[x:,:,:]    
                rgb_img[:s//2,:,:] = full_grid[x-s//2:x,:,:]   
                rgb_img[ox:,:,:] = full_grid[:x-s//2,:,:]    
            full_grid = rgb_img.copy()
            rgb_img[:,s-(y+1):,:] = full_grid[:,:y+1,:] 
            rgb_img[:,:s-(y+1),:] = full_grid[:,y+1:,:] 

        return {
            'mission': obs['mission'],
            'image': rgb_img
        }

class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8, obs_size=None):
        super().__init__(env)

        self.tile_size = tile_size
        self.obs_size = obs_size

        obs_shape = env.observation_space.spaces['image'].shape
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype='uint8'
        )
        if self.obs_size:
            self.observation_space.spaces['image'] = spaces.Box(
                low=0,
                high=255,
                shape=(self.obs_size, self.obs_size, 3),
                dtype='uint8'
            )

    def observation(self, obs):
        env = self.unwrapped
        
        # Render the whole grid
        if self.egocentric:
            grid, _ = env.grid.decode(obs['image'])
            rgb_img = grid.render(
                self.tile_size,
                agent_pos=(obs['image'].shape[0] // 2, obs['image'].shape[1] - 1),
                agent_dir=3
            )
        else:
            rgb_img = env.render(
                mode='rgb_array',
                highlight=False,
                tile_size=self.tile_size
            )
        
        # Resize image
        if self.obs_size:
            rgb_img = cv2.resize(rgb_img, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA)

        return {
            'mission': obs['mission'],
            'image': rgb_img
        }
