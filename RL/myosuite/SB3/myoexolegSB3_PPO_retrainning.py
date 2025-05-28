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
from torchrl.record.loggers import generate_exp_name, get_logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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


@hydra.main(config_path=".", config_name="config_PPO")
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
    envs = DummyVecEnv([make_env(env_name,i) for i in range(num_cpu)])
    env = Monitor(gym.make(env_name))  # Add Monitor here
    # Log Path
    log_path = f'.'
    
    
    # Creating normalized action space for policy
    n_actions = envs.action_space.shape[0]
    act_low = np.concatenate((np.array([envs.action_space.low[0]]), np.zeros(n_actions - 1)))
    act_high = np.concatenate((np.array([envs.action_space.high[0]]), np.zeros(n_actions - 1)))
    normalized_action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)
    
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
    wandb_callback = WandbCallback(
        log_interval=cfg.logger.check_interval
    )

    plotting_callback = PlottingCallback()
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    policy_kwargs = dict(activation_fn=th.nn.GELU,
                     net_arch=dict(pi=[32], vf=[32]))
    # Define model with the custom policy class
    model = PPO.load("/Users/achs/Documents/PHD/code/NAIR_Code/code/RL/SB3/outputs/ExoLegSpasticityFlexoExtEnv-v0/2024-10-23/16-51-23")
                
    # Copy environment.py on outputs
    shutil.copy(cfg.env.path, log_path)
    # Define action space for the model
    print(model)
    model.set_env(envs)
    # Create logger
    #logger = None
    #if cfg.logger.backend:
    #    exp_name = generate_exp_name("PPO", f"{cfg.logger.exp_name}_{cfg.env.env_name}")
    #    logger = get_logger(
    #        cfg.logger.backend, logger_name="ppo", experiment_name=exp_name
    #    )
    
    # Training
    print("---- Begin training -----")
    model.learn(total_timesteps=cfg.collector.total_frames, callback=callback_list, progress_bar=True)
    
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
