import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import highway_env

env = gym.make('highway-v0')
default_config = env.unwrapped.config
print(default_config)
