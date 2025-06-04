import sys, os
import time
from datetime import datetime
import gym
import torch
from sconetools import sconepy
import yaml
# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils import set_seed

sconegym_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../NAIR_envs')) 
sys.path.append(sconegym_path)
import sconegym
today = datetime.now().strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
# ====================================================================
# Scone step simulation definition
# --------------------------------------------------------------------
def scone_step(model, actions, use_neural_delays=True, step=0):
	muscle_activations = actions
	model.init_muscle_activations(muscle_activations)
	dof_positions = model.dof_position_array()
	model.set_dof_positions(dof_positions)
	model.init_state_from_dofs()
	# The model inputs are computed here
	# We use an example controller that mimics basic monosynaptic reflexes
	# by simply adding the muscle force, length and velocity of a muscle
	# and feeding it back as input to the same muscle
	if use_neural_delays:
		# Read sensor information WITH neural delays
		mus_in = model.delayed_muscle_force_array()
		mus_in += model.delayed_muscle_fiber_length_array() - 1.2
		mus_in += 0.1 * model.delayed_muscle_fiber_velocity_array()
	else:
		# Read sensor information WITHOUT neural delays
		mus_in = model.muscle_force_array()
		mus_in += model.muscle_fiber_length_array() - 1
		mus_in += 0.2 * model.muscle_fiber_velocity_array()
	#motor_torque = np.array([0])
	#mus_in = np.concatenate((mus_in,motor_torque))
	# The muscle inputs (excitations) are set here
	if use_neural_delays:
		# Set muscle excitation WITH neural delays
		model.set_delayed_actuator_inputs(mus_in)
	else:
		# Set muscle excitation WITHOUT neural delays
		model.set_actuator_inputs(mus_in)
	
	model.advance_simulation_to(step)

	return model.com_pos(), model.time()

# ====================================================================
# RL controller definition
# --------------------------------------------------------------------
def SKRL_controller(state, env_eval, step, timesteps, device):
	obs_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
	action_tensor, _, _ = agent.act(obs_tensor, timestep=step, timesteps=timesteps)
	#actions, log_prob, outputs = agent.act(obs_tensor, timestep=step, timesteps=timesteps)
	action = action_tensor.squeeze(0).detach().cpu().numpy()
	print("reward: ", env_eval._get_rew())
	return action, env_eval.step(action)
	#state, reward, terminated, info = env_eval.step(action)

# ====================================================================
# Scripts and paths administrations
# --------------------------------------------------------------------
trainning_path = "/home/achs/Documents/achs/code/NAIR_code/RL/scone/SKRL/outputs/nair_walk_h0918-v0/2025-06-03/14-01-02"
sys.path.append('training_path')
from torch_gym_nair_H0918_ppo import Policy, Value, CustomPPO

with open(trainning_path+"/.hydra/config.yaml", "r") as file:
    config = yaml.safe_load(file)

config_env = config["env"]
config_logger = config["logger"]
config_optim = config["optim"]
config_hiperparameters = config["hiperparameters"]

set_seed()
# Path to the trained model checkpoint
best_model_path = trainning_path+"/runs/outputs/checkpoints/best_agent.pt"
from gym import spaces
import numpy as np

# ====================================================================
# Create vectorized environment for prediction
# --------------------------------------------------------------------
try:
    env = gym.vector.make(config_env["env_name"], use_delayed_sensors=True, num_envs=config_env["num_cpu"], asynchronous=False)
    print("observation space env", env.observation_space)
except gym.error.DeprecatedEnv as e:
    env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("nair")][0]
    print(config_env["env_name"], " not found. Trying {}".format(env_id))
# Wrap environmet for use torch tensors operations
env = wrap_env(env)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
from gym.spaces import Box
from skrl.utils import spaces as skrl_spaces
print("observation space wrap", env.observation_space)

# ====================================================================
# Model parameters loading and instantiation
# --------------------------------------------------------------------
# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
models["value"] = Value(env.observation_space, env.action_space, device)
print("Action space:",env.action_space, "   Observation space: ", env.observation_space)
# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = config_hiperparameters["batch_size"]  # memory_size
cfg["learning_epochs"] = config_hiperparameters["learning_epochs"]
cfg["mini_batches"] = config_hiperparameters["mini_batches"]
cfg["discount_factor"] = config_hiperparameters["discount_factor"]
cfg["lambda"] = config_hiperparameters["lambda_factor"]
cfg["learning_rate"] = config_optim["lr"]
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = config_optim["optimizer_kwargs"]
cfg["grad_norm_clip"] = config_hiperparameters["clip_norm"]
cfg["ratio_clip"] = config_hiperparameters["ratio_clip"]
cfg["value_clip"] = config_hiperparameters["value_clip"]
cfg["clip_predicted_values"] = config_hiperparameters["clip_predicted_values"]
cfg["entropy_loss_scale"] = config_hiperparameters["entropy_loss"]
cfg["value_loss_scale"] = config_hiperparameters["value_loss"]
cfg["kl_threshold"] = config_hiperparameters["kl_threshold"]

cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = config_logger["write_interval"]
cfg["experiment"]["checkpoint_interval"] = config_logger["check_interval"]
cfg["experiment"]["directory"] = config_env["torch_dir"]

agent = CustomPPO(models=models,
            memory=None,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
            )

agent.load(best_model_path)


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
model = sconepy.load_model("../../../../NAIR_envs/sconegym/nair_envs/H0918_KneeExo/H0918_KneeExoV0.scone")
model.reset()
model.set_store_data(store_data)

dirname = "sconerun_" + config_env["algorithm"] + "_" + model.name() + "_" + today

# Configuration  of time steps and simulation time
max_time = 2 # In seconds
timestep = 0.01
timesteps = int(max_time / timestep)

# ====================================================================
# Controller loop and main function
# --------------------------------------------------------------------
for step in range(timesteps):

	steps = step
	step= step*timestep
	actions, (state, reward, terminated, info) = SKRL_controller(state, env_eval, step, timesteps, device=device)
	model_com_pos, model_time = scone_step(model, actions, use_neural_delays=True, step=step)
	episode_reward += reward
	com_y = model.com_pos().y
	print(com_y)
	if com_y < min_com_height:
		print(f"Aborting simulation at t={model.time():.2f} com_y={com_y:.4f}")
		break

print(f"Episode completed in {steps} steps with total reward: {episode_reward:.2f}")
env_eval.close()
if store_data:
	filename = model.name() + f'_{model.time():0.2f}_{episode_reward:0.2f}'
    
	if use_neural_delays: dirname += "_delay"
	model.write_results(dirname, filename)
	print(f"Results written to {dirname}/{filename}", flush=True)

