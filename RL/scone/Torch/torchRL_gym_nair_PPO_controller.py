"""
    Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
    source	:: https://github.com/AndresChS/NAIR_Code
"""

import sys
import os
import gym
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import ObservationNorm
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules import ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule, TensorDictSequential
from omegaconf import DictConfig
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from tensordict import TensorDict
from sconetools import sconepy # type: ignore
import yaml

import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing

sconegym_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../NAIR_envs')) 
sys.path.append(sconegym_path)
import sconegym # type: ignore
today = datetime.now().strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
# ====================================================================
# Scone step simulation definition
# --------------------------------------------------------------------
def scone_step(model, muscles_actions, motor_torque, use_neural_delays=True, step=0):

	muscle_activations = muscles_actions
	motor_torque = np.array([motor_torque])
	#print("torque: ", motor_torque, "	mus_in: ", mus_in)
	mus_in = np.concatenate((muscle_activations,motor_torque))
	#print(mus_in)
	model.set_actuator_inputs(mus_in)
	
	model.advance_simulation_to(step)

	return model.com_pos(), model.time()

# ====================================================================
# RL controller definition
# --------------------------------------------------------------------
def TorchRL_controller(state, env_eval, step, timesteps, device):
	obs_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
	tensordict = TensorDict({"observation": obs_tensor}, batch_size=[1])

	# sample from the distribution
	action_tensordict = agent.sample(tensordict)
	action = action_tensordict["action"][0].detach().cpu().numpy()

	print("reward: ", env_eval._get_rew())
	return action, env_eval.step(action)
	#state, reward, terminated, info = env_eval.step(action)

# ====================================================================
# Scripts and paths administrations
# --------------------------------------------------------------------
trainning_path = "/home/achs/Documents/achs/code/NAIR_code/RL/scone/SKRL/outputs/nair_walk_h0918-v0/2025-06-03/14-01-02"
sys.path.append('training_path')
from nair_agents import get_models

with open(trainning_path+"/.hydra/config.yaml", "r") as file:
    config = yaml.safe_load(file)

config_env = config["env"]
config_logger = config["logger"]
config_optim = config["optim"]
config_hiperparameters = config["hiperparameters"]
env_id = config_env["env_name"]

# Path to the trained model checkpoint
best_model_path = trainning_path+"/runs/outputs/checkpoints/best_agent.pt"
from gym import spaces
import numpy as np

# ====================================================================
# Create vectorized environment for prediction
# --------------------------------------------------------------------
try:
	env = gym.vector.make(env_id, use_delayed_sensors=config_env["use_delayed_sensors"], num_envs=config_env["num_cpu"], asynchronous=False)
	print("observation space env: ", env.observation_space)
	print("action space env: ", env.action_space)
except gym.error.DeprecatedEnv as e:
	env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("nair")][0]
	print("sconewalk_h0918_osim-v1 not found. Trying {}".format(env_id))

base_env = GymWrapper(env)
env = TransformedEnv(base_env)
obs_dim = env.observation_spec["observation"].shape[-1]
act_dim = env.action_spec.shape[-1]
agent = "PPO"  # o "PPO", "DDPG"

# ====================================================================
# Model parameters loading and instantiation
# --------------------------------------------------------------------
actor, _ = get_models(agent, obs_dim, act_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = env.to(device)
actor.to(device)
agent = TensorDictModule(
	actor, in_keys=["observation"], out_keys=["loc", "scale"]
)
agent = ProbabilisticActor(
	module=agent,
	spec=env.action_spec,
	in_keys=["loc", "scale"],
	distribution_class=TanhNormal,
	distribution_kwargs={
		"low": env.action_spec.space.low,
		"high": env.action_spec.space.high,
	},
	return_log_prob=True,
	# we'll need the log-prob for the numerator of the importance weights
)

agent.load_state_dict(torch.load(best_model_path))


# ====================================================================
# Evaluation environment definition
# --------------------------------------------------------------------
# Reset environment and initialize variables for an episode
env_eval = gym.vector.make(config_env["env_name"], use_delayed_sensors=True, num_envs=1, asynchronous=False)
#gym.vector.make("sconewalk_h0918_osim-v1")
state = env_eval.reset()
print("observation space eval", env_eval.observation_space)
done = False
episode_reward =0
store_data = True
use_neural_delays = config_env["use_delayed_sensors"]
random_seed =1
min_com_height = 0 #minimun heigth to abort the simulation
# ====================================================================
# Sconepy model initialitation
# --------------------------------------------------------------------
model = sconepy.load_model(sconegym_path+"/sconegym/nair_envs/H0918_KneeExo/H0918_KneeExoRLV0.scone", sconegym_path+"/sconegym/nair_envs/H0918_KneeExo/par/gait.par")
model.reset()
model.set_store_data(store_data)
sconepy.evaluate_par_file(sconegym_path+"/sconegym/nair_envs/H0918_KneeExo/par/gait.par")

dof_positions = model.dof_position_array()
model.set_dof_positions(dof_positions)
model.init_state_from_dofs()

# Configuration  of time steps and simulation time
max_time = 5 # In seconds
timestep = 0.005
timesteps = int(max_time / timestep)
com_y_list = []
time_list = []
pos_list = []
# ====================================================================
# Controller loop and main function
# --------------------------------------------------------------------
for step in range(timesteps):

	actions = np.zeros(len(model.muscles()))
	motor_torque, (state, reward, terminated, info) = TorchRL_controller(state, env_eval, step*timestep, timesteps, device=device)
	model_com_pos, model_time = scone_step(model, motor_torque=motor_torque, muscles_actions=actions, use_neural_delays=False, step=step*timestep)
	episode_reward += reward
	com_y = model.com_pos().y
	dofs = model.dofs()
	pos_list.append(dofs[2].pos())
	com_y_list.append(model.com_pos().y)
	time_list.append(step*timestep)
	print(com_y)
	if com_y < min_com_height:
		print(f"Aborting simulation at t={model.time():.2f} com_y={com_y:.4f}")
		break

print(f"Episode completed in {step} steps with total reward: {episode_reward:.2f}")
env_eval.close()

if store_data:
	dirname = "sconerun_" + config_env["algorithm"] + "_" + model.name() + "_" + today
	filename = model.name() + f'_{model.time():0.2f}_{episode_reward:0.2f}'
    
	if use_neural_delays: dirname += "_delay"
	model.write_results(dirname, filename)
	print(f"Results written to {dirname}/{filename}", flush=True)

