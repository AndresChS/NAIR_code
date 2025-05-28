from myosuite import gym
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import numpy as np
import os, time, multiprocessing
import json
from pymongo import MongoClient
import pandas as pd
import subprocess
import matplotlib.pyplot as plt

client = MongoClient("mongodb://localhost:27017/")
db = client['RL_Tests']  # Ajusta el nombre de la base de datos
# Experiment
experiment = 'PPO_spasticity_0_69'

# Define the environment name and model path
env_name = "ExoLegSpasticityFlexoExtEnv-v0"     #"ExoLeg40MuscFlexoExtEnv-v0"   
                                                #"ExoLegPassiveFlexoExtEnv-v0"
                                                #"ExoLegSpasticityFlexoExtEnv-v0"

log_path = f'/Users/achs/Documents/PHD/code/NAIR_Code/code/RL/SB3/outputs/ExoLegSpasticityFlexoExtEnv-v0/2024-11-05/20-02-02'

# Define the path to the best model saved
best_model_path = os.path.join(log_path, 'best_model.zip')  # Ensure the correct file extension

def save_weights_to_txt(angle_leg, filename):
    # Abre el archivo TXT en modo escritura
    with open(filename, 'w') as file:
        file.write(str(angle_leg))
    

def create_env():
    env = gym.make(env_name)
    #env.set_target_qpos(target_qpos=0.4)
    return env

def main():
    # Create the vectorized environment
    target = 0.2
    env = DummyVecEnv([create_env])
    
    # Load the best model
    try:
        model = SAC.load(best_model_path)
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found at {best_model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

    # Visualize Results
    save_dict = {}
    obs = env.reset()
    actions = []
    sum_rw = 0
    #print(model)
    for _ in range(256):
        action, _states = model.predict(obs)
        action[:, 1:] = 0 
        obs, rewards, dones, info = env.step(action)
        #env.render()
        actions.append(action[0])
    #print(model.policy)
    env_mj = gym.make(env_name)
    obs_mj = env_mj.reset()
    #action = np.zeros(41)
    #action[0] = 1
    #print(actions[100])
    sum_rw = 0
    step = 0
    for i in actions:
        obs_mj, rw, *_ = env_mj.step(np.array(i))
        sum_rw = rw + sum_rw
        #print(obs_mj)
        time.sleep(0.01)
        env_mj.mj_render()
        data = env_mj.sim.data
        step = step + 1
        data_dict = {

                'id_experiment': experiment,
                'step': step,
                'time': data.time,
                'reward':rw,
                'qpos': data.qpos.tolist(),
                'qvel': data.qvel.tolist(),
                'qacc': data.qacc.tolist(),
                'ctrl': data.ctrl.tolist(),
                'act': data.act.tolist() if data.act.size > 0 else [],
                'xpos': data.xpos.tolist(),
                'xquat': data.xquat.tolist(),
                'xmat': data.xmat.tolist(),
                'qfrc_actuator': data.qfrc_actuator.tolist(),
                'qfrc_constraint': data.qfrc_constraint.tolist(),
                'inter_force': obs_mj[-1].tolist(),
                'sensor': data.sensordata.tolist() if data.sensordata.size > 0 else [],
                'energy': data.energy.tolist() if hasattr(data, 'energy') else [],
                'contact': [contact.__dict__ for contact in data.contact[:data.ncon]] if data.ncon > 0 else [],
                'subtree_com': data.subtree_com.tolist(),
                'cinert': data.cinert.tolist(),
                'ten_length': data.ten_length.tolist(),
            }
        #print(data.actuator_force)
        db.SB3.insert_one(data_dict)
    
    env_mj.close()
    print(sum_rw)


if __name__ == "__main__":
    main()
#subprocess.run(["python3", "./plot_save_results.py"])