"""	
This model is generated with tacking the Myosuite conversion of [Rajagopal's full body gait model](https://github.com/opensim-org/opensim-models/tree/master/Models/RajagopalModel) as close
reference.
	Model	  :: MyoLeg 1 Dof Exo (MuJoCoV2.0)
	Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
	source	:: https://github.com/AndresChS/NAIR_Code
"""

import hydra # type: ignore
from myosuite.utils import gym
import numpy as np
import time
from datetime import datetime
import shutil
import torch as th

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import numpy as np
import os

# Using time to define the unique naming
start_time = time.time()
today = datetime.now().strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
time_now = datetime.now().strftime('%H-%M-%S')  # Format: HH-MM-SS
tensorlog_path = f'code/RL/SB3/tensorlogs'

# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def make_env(env_id: str, num_env = 0, seed: int = 0):
    """
    Utility function for multiprocessed env.
    env_id: the environment ID
    num_env: index of the subprocess 
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + num_env)
        return env
    
    return _init


@hydra.main(config_path=".", config_name="config_SAC")
def main(cfg):  # noqa: F821

    # Define env and model architecture
    # Initializa N parallel envs and create them
    env_name = cfg.env.env_name                     #"ExoLeg40MuscFlexoExtEnv-v0"   
                                                    #"ExoLegPassiveFlexoExtEnv-v0"
                                                    #"ExoLegSpasticityFlexoExtEnv-v0"
    num_cpu = cfg.env.num_cpu # Number of processes to use
    # Create log dir
    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    envs = gym.make(env_name)
    env = Monitor(gym.make(env_name))  # Add Monitor here
    # Log Path
    log_path = f'.'
    
    
    # Creating normalized action space for policy
    n_actions = envs.action_space.shape[0]
    act_low = np.concatenate((np.array([envs.action_space.low[0]]), np.zeros(n_actions - 1)))
    act_high = np.concatenate((np.array([envs.action_space.high[0]]), np.zeros(n_actions - 1)))
    normalized_action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)
    action_noise = NormalActionNoise(mean=0.0, sigma=0.1 * th.ones(n_actions))  # Ruido de exploración

# ====================================================================
# Environment Callbacks
# --------------------------------------------------------------------

    # Define evaluation and checkpoint callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=cfg.logger.test_interval,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq= cfg.logger.check_interval,  # Save the model every 10000 training steps
        save_path= log_path + '/checkpoints/',  # Folder to save checkpoints
        name_prefix='rl_model',  # Prefix for saved model files
        verbose=0
    )

    callback_list = CallbackList([eval_callback, checkpoint_callback])
    policy_kwargs = dict(activation_fn=th.nn.GELU,
                     net_arch=dict(pi=[64, 64], qf=[64, 64]))
    
# ====================================================================
# Model Definition
# --------------------------------------------------------------------

    # Define model with the custom policy class
    model = SAC(
                policy="MlpPolicy", 
                env=envs, 
                learning_rate=cfg.optim.lr, 
                buffer_size=cfg.collector.buffer_size,
                batch_size=cfg.hiperparameters.batch_size,
                tau=cfg.hiperparameters.tau,
                gamma=cfg.hiperparameters.gamma,
                train_freq=(cfg.hiperparameters.train_freq,"episode"),
                gradient_steps=cfg.hiperparameters.gradient_steps,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                verbose=0,
                tensorboard_log=log_path,
                device="cuda")
    
    # Define action space for the model
    model.action_space = normalized_action_space
    # Copy environment.py on outputs
    shutil.copy(cfg.env.path, log_path)

    # Training
    print("---- Begin training -----")
    model.learn(total_timesteps=cfg.collector.num_steps, callback=callback_list, progress_bar=True)
    



    # Visualize Results
    obs = envs.reset()
    actions = []
    print(model.policy)
    for _ in range(512):
        action, _states = model.predict(obs)
        action[:, 1:] = 0 
        obs, rewards, dones, info = envs.step(action)
        #envs.render()
        actions.append(action[0])
    #print(model.policy)

    env_mj = gym.make(env_name)
    obs_mj = env_mj.reset()
    for i in actions:
        env_mj.step(np.array(i))
        time.sleep(0.01)
        env_mj.mj_render()
    
    env_mj.close()

if __name__ == "__main__":
    main()
# WARN: env.mj_render to get variables from other wrappers is deprecated 
# and will be removed in v1.0, to get this variable you can do `env.unwrapped.mj_render` 
# for environment variables or `env.get_wrapper_attr('mj_render')` that will search the reminding wrappers.
"""
from myosuite.utils import gym
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import time
from datetime import datetime
import torch as th
import torch.nn as nn
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, CheckpointCallback


# Using time to define the unique naming
start_time = time.time()
today = datetime.now().strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
time_now = datetime.now().strftime('%H-%M-%S')  # Format: HH-MM-SS

if __name__ == "__main__":

    # Define env and model architecture
    # Initializa N parallel envs and create them
    env_name = "ExoLegPassiveFlexoExtEnv-v0"     #"ExoLeg40MuscFlexoExtEnv-v0"   
                                                #"ExoLegPassiveFlexoExtEnv-v0"
    num_cpu = 4  # Number of processes to use

    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    envs = DummyVecEnv([make_env(env_name,i) for i in range(num_cpu)])
    
    # Log Path
    log_path = f'./outputs/{env_name}/{today}/{time_now}'
    
    # Creating normalized action space for policy
    n_actions = envs.action_space.shape[0]
    act_low = np.concatenate((np.array([envs.action_space.low[0]]), np.zeros(n_actions - 1)))
    act_high = np.concatenate((np.array([envs.action_space.high[0]]), np.zeros(n_actions - 1)))
    normalized_action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)
    
    # The noise objects for DDPG
    n_actions = envs.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
"""

