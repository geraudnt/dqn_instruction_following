"""
Learning Tasks
"""

import numpy as np
import torch
import os
import sys
import itertools

import gym
import gym_minigrid
import envs.envs
from envs.wrappers import *

from library import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--env_key',
    default="MiniGrid-Custom-PickUpObj-12x12-v0", # "MiniGrid-DoorKey-8x8-v0"
    help="Environment"
)
args = parser.parse_args()

def evaluate(env, model, max_steps):
    returns, successes = (0, 0)
    obs = env.reset()
    mission = tokenize(obs['mission'],model['vocab'])
    for t in range(max_steps):
        action = select_action(model, obs['image'], mission)
        new_obs, reward, done, info = env.step(action)
        obs = new_obs
        returns += reward
        successes += (reward>0)
        if done:
            break
    return [returns, successes]

def train(env,
            model_path="model",
            logs_path="logs",
            eval_model=False,
            save_model=True,
            save_logs=True,
            load_model=False,
            max_episodes=int(1e6),
            learning_starts=int(1e4),
            replay_buffer_size=int(1e5),
            train_freq=4,
            target_update_freq=int(1e3),
            batch_size=32,
            gamma=0.95,
            learning_rate=1e-4,
            eps_initial=1.0,
            eps_final=0.1,
            max_steps = 100,
            mean_episodes=100,
            eps_timesteps=int(5e5),
            print_freq=10):
    
    if hasattr(env, "max_steps"):
        max_steps = env.max_steps
    
    ### Initialising
    eps_schedule = LinearSchedule(eps_timesteps, eps_final, eps_initial)
    replay_buffer = ReplayBuffer(replay_buffer_size, batch_size)
    
    agent = Agent(env, gamma=gamma, learning_rate=learning_rate, replay_buffer=replay_buffer, path=model_path)
    if load_model and os.path.exists(model_path):
        model = load(model_path, env)
        agent.vocab = model['vocab']
        agent.q_func.load_state_dict(model['params'].state_dict())
        agent.target_q_func.load_state_dict(model['params'].state_dict())
        print('RL model loaded')
    model = {'params': agent.q_func, 'vocab': agent.vocab}

    # Training  
    episode_returns = []
    episode_successes = []
    eval_returns = []
    eval_successes = []
    avg_return_best = 0
    success_rate_best = 0
    steps = 0
    for episode in range(max_episodes):
        obs = env.reset()
        mission = tokenize(obs['mission'],agent.vocab)

        episode_returns.append(0.0)
        episode_successes.append(0.0)
        done = False
        t = 0
        while not done and t<max_steps:
            ### Collecting experience
            if random.random() > eps_schedule(steps):
                action = select_action(model, obs['image'], mission)
            else:
                action =  env.action_space.sample()
            
            new_obs, reward, done, info = env.step(action)
            replay_buffer.add(mission, obs['image'], action, reward, new_obs['image'], done, info)
            obs = new_obs
            episode_returns[-1] += reward
            episode_successes[-1] = (reward>0)

            ### Updating agent    
            if steps > learning_starts:
                agent.update_td_loss()

            if steps > learning_starts and steps % target_update_freq == 0:
                agent.update_target_network()

            t += 1    
            steps += 1
        
        returns, successes = 0, 0
        if eval_model:
            returns, successes = evaluate(env, model, max_steps)
        eval_returns.append(returns)
        eval_successes.append(successes)
        
        ### Print and Save training progress
        if print_freq is not None and episode % print_freq == 0:
            avg_return = round(np.mean(episode_returns[-mean_episodes-1:-1]), 1)
            success_rate = np.mean(episode_successes[-mean_episodes-1:-1]) 
            eval_avg_return = round(np.mean(eval_returns[-mean_episodes-1:-1]), 1)
            eval_success_rate = np.mean(eval_successes[-mean_episodes-1:-1]) 
            if avg_return > avg_return_best:
                avg_return_best = avg_return
                success_rate_best = success_rate
                if save_model:
                    agent.save()
                    print("\nModel saved. ar: {}, sr: {}\n".format(avg_return_best, success_rate_best))    
            if save_logs:
                np.save(logs_path, [episode_returns, episode_successes, eval_returns, eval_successes]) 
            print("--------------------------------------------------------")
            print("steps {}".format(steps))
            print("episodes {}".format(episode))
            print("mission {}".format(obs['mission']))
            print("average return: current {}, best {}, eval {}".format(avg_return,avg_return_best,eval_avg_return))
            print("success rate: current {}, best {}, eval {}".format(success_rate,success_rate_best,eval_success_rate))
            print("% time spent exploring {}".format(int(100 * eps_schedule(steps))))
            print("--------------------------------------------------------")
    
    return agent, model, episode_returns, episode_successes, eval_returns, eval_successes


if __name__ == '__main__':    
    env = gym.make(args.env_key)
    if "MiniGrid" in args.env_key:
        env = FullyObsWrapper(env) # Wrapper for MiniGrid egocentric full observations
        env = RGBImgObsWrapper(env) # Wrapper for MiniGrid pixel observations
    
    model_path='models/{}'.format(args.env_key)
    logs_path='logs/{}'.format(args.env_key)
    train(env, model_path = model_path, logs_path = logs_path)

    
