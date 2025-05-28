import sys
import os
import numpy as np
from myosuite.utils import gym
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import minimize
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import time, subprocess

# Crear una instancia del entorno

env = gym.make('ExoLeg40MuscFlexoExtEnv-v0', target_qpos=0.5)
env.reset()
data = env.sim.data
dt = env.sim.model.opt.timestep
action_data = np.loadtxt('/Users/achs/Documents/PHD/code/NAIR_Code/code/RL/Pytorch/outputs/2024-09-02/09-55-30/datos.txt')
acumulate_reward =0
a = 0
for i in action_data:
    action = i # Listas para almacenar los datos
    obs, reward, *_ = env.step(action)
    a += 1
    acumulate_reward += reward
    if a % 200 == 0:
        print(acumulate_reward)
        acumulate_reward = 0
    env.mj_render()
    time.sleep(0.01)