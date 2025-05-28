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
import time, subprocess

def open_mongodb_server(db_name='PID_Results', collection_name='experiments_results', experiment_name='PID_lv0_0', host="localhost", port=27017):
    """
    Initializes the connection to a MongoDB server and returns the specified database and collection.
    
    Args:
        db_name (str): Name of the database to connect to.
        collection_name (str): Name of the collection to access.
        experiment_name (str): Name of the experiment to store or query.
        host (str): MongoDB server hostname (default is 'localhost').
        port (int): MongoDB server port (default is 27017).
    
    Returns:
        collection: The MongoDB collection object for further operations.
        experiment: The experiment identifier to use within the collection.
    """
    # Connect to MongoDB server
    client = MongoClient(f"mongodb://{host}:{port}/")
    
    # Access the database and collection
    db = client[db_name]
    collection = db[collection_name]
    
    # Optionally set experiment name (could be used later in queries)
    experiment = experiment_name
    
    return collection, experiment, db

"""
client = MongoClient("mongodb://localhost:27017/")
db = client['PID_Results']  # Ajusta el nombre de la base de datos
collection = db['experiments_results']  # Ajusta el nombre de la colección
# Experiment
experiment = 'PID_lv0_0'
"""
def test(value):
    print(type(value))
    print(value)
    return value

# PID controller definition
class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute(self, current_position, dt, setpoint):
        # Setpoint if change during the simulation
        self.setpoint = setpoint
        # Error
        error = self.setpoint - current_position
        
        # Integral term
        self.integral += error * dt
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        
        # Output PID controller
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Limit output to range [-100, 100]
        output = np.clip(output, -1, 1) 

        # Actualization
        self.prev_error = error
        
        return output,error


def run_simulation(kp, ki, kd, setpoint, num_steps):
    env = gym.make('ExoLegSpasticityFlexoExtEnv-v0')
    obs_res = env.reset()
    data = env.sim.data
    dt = env.sim.model.opt.timestep
    test(obs_res)
    pid_controller = PIDController(kp, ki, kd, setpoint)
    # Listas para almacenar los datos
    positions = []
    torques = []
    times = []
    rw_sum=0
    for step in range(num_steps):
        current_position = data.qpos[env.sim.model.joint_name2id('knee_angle_r')]
        """if step >= 100:
            if step % 100 == 0:
                setpoint = setpoint + 0.0174
                if setpoint >= 1.57: #90º max
                    setpoint = 1.57
        """
        torque, PID_error = pid_controller.compute(current_position, dt, setpoint)
        actions = np.zeros(41)
        actions[0] = -torque
        obs, rw, *_= env.step(actions)
        rw_sum = rw + rw_sum
        #print(rw)
        #env.mj_render()
        #time.sleep(0.01)
        if optimized ==1:
            data_dict = {

                'id_experiment': experiment,
                'Parameters': params_list,
                'optimized': optimized,
                'setpoint': setpoint,
                'error': (-PID_error),
                'reward': rw,
                'step': step,
                'time': data.time,
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
                'inter_force': obs[-1].tolist(),
                'sensor': data.sensordata.tolist() if data.sensordata.size > 0 else [],
                'energy': data.energy.tolist() if hasattr(data, 'energy') else [],
                'contact': [contact.__dict__ for contact in data.contact[:data.ncon]] if data.ncon > 0 else [],
                'subtree_com': data.subtree_com.tolist(),
                'cinert': data.cinert.tolist(),
                'ten_length': data.ten_length.tolist(),
            }

            db.torque.insert_one(data_dict)
        
    env.close()
    print(rw_sum)
    return 

collection, experiment, db = open_mongodb_server(
    db_name='0_env_data', 
    collection_name='torque', 
    experiment_name='angles_torque_61'
    )

# Initial guess for PID parameters
num_steps = 256  # Número de pasos de simulación
kp = 7
ki = 50
kd = 0.03
setpoint = 0.2
params_list = [ki, kp, kd]
optimized = 1
# Perform the optimization
run_simulation(kp, ki, kd, setpoint, num_steps)
subprocess.run(["python3", "./plot_save_results.py"])








