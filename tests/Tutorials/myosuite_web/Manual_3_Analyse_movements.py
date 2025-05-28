# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:28:54 2023
Manual_data_analysis 
@author: andres
"""


import myosuite
from myosuite.utils import gym
import skvideo.io
import numpy as np
import os
import matplotlib.pyplot as plt

 
env = gym.make('myoElbowPose1D6MRandom-v0')
env.reset()

data_store = []
for _ in range(100):
    env.mj_render()
    a = env.action_space.sample()
    env.step(a) # take a random action
    
    data_store.append({"action":a.copy(), 
                        "jpos":env.sim.data.qpos.copy(), 
                        "mlen":env.sim.data.actuator_length.copy(), 
                        "act":env.sim.data.act.copy()})
env.close()


def VAF(W, H, A):
    """
    Args:
        W: ndarray, m x rank matrix, m-muscles x activation coefficients obtained from (# rank) nmf
        H: ndarray, rank x L matrix, basis vectors obtained from nmf where L is the length of the signal
        A: ndarray, m x L matrix, original time-invariant sEMG signal
    Returns:
        global_VAF: float, VAF calculated for the entire A based on the W&H
        local_VAF: 1D array, VAF calculated for each muscle (column) in A based on W&H
    """
    SSE_matrix = (A - np.dot(W, H))**2
    SST_matrix = (A)**2

    global_SSE = np.sum(SSE_matrix)
    global_SST = np.sum(SST_matrix)
    global_VAF = 100 * (1 - global_SSE / global_SST)

    local_SSE = np.sum(SSE_matrix, axis = 0)
    local_SST = np.sum(SST_matrix, axis = 0)
    local_VAF = 100 * (1 - np.divide(local_SSE, local_SST))

    return global_VAF, local_VAF



import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

act = np.array([dd['act'] for dd in data_store])

VAFstore=[]
SSE, SST = [], []

sample_points = [1,2,3,4,5,10,20,30]
for isyn in sample_points:
    nmf_model = NMF(n_components=isyn, init='random', random_state=0);
    W = nmf_model.fit_transform(act)
    H = nmf_model.components_

    global_VAF, local_VAF = VAF(W, H, act)

    VAFstore.append(global_VAF)

plt.plot(sample_points,VAFstore,'-o')
plt.xlabel('Number of Muscle Synergies')
plt.ylabel('Explained Variance R^2')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


