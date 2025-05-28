#=================================================
#	This model is generated with tacking the Myosuite conversion of [Rajagopal's full body gait model](https://github.com/opensim-org/opensim-models/tree/master/Models/RajagopalModel) as close
#reference.
#	Model	  :: Myo Leg 1 Dof 40 Musc Exo (MuJoCoV2.0)
#	Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
#	source	:: https://github.com/AndresChS/NAIR_Code
#	====================================================== -->
import numpy as np
from myosuite.utils import gym
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import minimize
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import time
import subprocess

def inspect_mjdata(data):
    """
    Inspects and prints all attributes of an mjData object.
    
    Parameters:
    data (mjData): The mjData object to inspect.
    """
    attributes = dir(data)
    for attr in attributes:
        # Ignore private and special attributes
        if attr.startswith('_'):
            continue
        try:
            value = getattr(data, attr)
            print(f"{attr}")
        except Exception as e:
            print(f"Could not access attribute {attr}: {e}")

# Example usage
env = gym.make('ExoLeg40MuscFlexoExtEnv-v0')
env.reset()
data = env.sim.data

inspect_mjdata(data)

client = MongoClient("mongodb://localhost:27017/")
db = client["0_env_data"]
# Experiment
experiment = 'knee_angles_raw'

def run_simulation(num_steps, torque):
    env = gym.make('ExoLeg40MuscFlexoExtEnv-v0')
    env.reset()
    data = env.sim.data
    actions = np.zeros(41)
    for step in range(num_steps):
        current_position = data.qpos[env.sim.model.joint_name2id('knee_angle_r')]
        
        if step >= 300: 
            #db.experiments_results.insert_one(data_dict)
            if step % 10 == 0:
                torque = torque + 1
                if torque >= 100: #90º max
                    torque = 100
        actions[0] = torque
        obs, *_ = env.step(actions)
        data_dict = {

            'id_experiment': experiment,
            'time_step': step,
            'time': data.time,
            'qpos': data.qpos.tolist(),
        
        }

    env.reset()

    env.close()
    
    return 

# Open and reset Environment
env = gym.make('ExoLeg40MuscFlexoExtEnv-v0')
env.reset()

#Simulation parameters
num_steps = 2400  # Steps for simulation
torque = -100   # Initial Torque

run_simulation(num_steps, torque)

#subprocess.run(["python3", "/Users/achs/Documents/PHD/code/myosuite/tests/angles_plot.py"])








