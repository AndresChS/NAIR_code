#====================================================
#	This model is generated with tacking the Myosuite conversion of [Rajagopal's full body gait model](https://github.com/opensim-org/opensim-models/tree/master/Models/RajagopalModel) as close
#   reference.
#	Model	  :: MyoLeg 1 Dof 40 Musc Exo (MuJoCoV2.0)
#	Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
#	source	:: https://github.com/AndresChS/NAIR_Code
#====================================================
from myosuite.utils import gym
import numpy as np
import hydra
import time, os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# ====================================================================
# Unique naming and functions definition
# --------------------------------------------------------------------
start_time = time.time()
today = datetime.now().strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
time_now = datetime.now().strftime('%H-%M-%S')  # Format: HH-MM-SS
tensorlog_path = f'/Users/achs/Documents/PHD/code/NAIR_Code/code/RL/SB3/tensorlogs'
set_seed()
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

class NormalizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(NormalizedActionWrapper, self).__init__(env)
        n_actions = env.action_space.shape[0]
        act_low = np.concatenate((np.array([env.action_space.low[0]]), np.zeros(n_actions - 1)))
        act_high = np.concatenate((np.array([env.action_space.high[0]]), np.zeros(n_actions - 1)))
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

    def action(self, action):
        # Add normalization logic here if needed, for example:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action
    
class ResetWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        # Remove 'seed' from kwargs if it exists
        if 'seed' in kwargs:
            print("Removing 'seed' from reset kwargs")  # Optional: Debugging line
            kwargs.pop('seed')
        return self.env.reset(**kwargs)
# ====================================================================
# Model Class Definition
# --------------------------------------------------------------------    

# define models (deterministic models) using mixin
class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.action_layer(x)), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}

# ====================================================================
# Main loop and hydra preconfiguration
# --------------------------------------------------------------------
@hydra.main(config_path=".", config_name="config_DDPG")
def main(cfg):  # noqa: F821

    # Define env and model architecture
    # Initialize N parallel envs and create them
    env_name = cfg.env.env_name                     #"ExoLeg40MuscFlexoExtEnv-v0"   
                                                    #"ExoLegPassiveFlexoExtEnv-v0"
                                                    #"ExoLegSpasticityFlexoExtEnv-v0"
    num_cpu = cfg.env.num_cpu # Number of processes to use
    # Create log dir
    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    try:
        envs = gym.vector.make(env_name, num_envs=num_cpu, asynchronous=False)
    except gym.error.DeprecatedEnv as e:
        env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith(env_name)][0]
        print(env_name," not found. Trying {}".format(env_id))
        envs = gym.vector.make(env_id, num_envs=num_cpu, asynchronous=False)
    envs = wrap_env(envs)
    envs = NormalizedActionWrapper(envs)
    envs = ResetWrapper(envs)
    device = envs.device
    # Log Path
    log_path = f'.'   
    # instantiate a memory as experience replay
    memory = RandomMemory(memory_size=100000, num_envs=envs.num_envs, device=device, replacement=False)


# ====================================================================
# Defined and normalized action & observation space
# --------------------------------------------------------------------
    #Observation Space
    # Creating normalized action space for policy
    n_actions = envs.action_space.shape[0]
    act_low = np.concatenate((np.array([envs.action_space.low[0]]), np.zeros(n_actions - 1)))
    act_high = np.concatenate((np.array([envs.action_space.high[0]]), np.zeros(n_actions - 1)))
    normalized_action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)
    
# ====================================================================
# Models instantiation
# --------------------------------------------------------------------
    # instantiate the agent's models (function approximators).
    # DDPG requires 4 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
    models = {}
    models["policy"] = Actor(envs.observation_space, envs.action_space, device)
    models["target_policy"] = Actor(envs.observation_space, envs.action_space, device)
    models["critic"] = Critic(envs.observation_space, envs.action_space, device)
    models["target_critic"] = Critic(envs.observation_space, envs.action_space, device)

    # initialize models' parameters (weights and biases)
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
    cfg = DDPG_DEFAULT_CONFIG.copy()
    cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=1.0, device=device)
    cfg["batch_size"] = 100
    cfg["random_timesteps"] = 100
    cfg["learning_starts"] = 100
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 1000
    cfg["experiment"]["checkpoint_interval"] = 1000
    cfg["experiment"]["directory"] = "runs/torch/Pendulum"

    agent = DDPG(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=envs.observation_space,
                action_space=envs.action_space,
                device=device)
 
# ====================================================================
# Trainning
# --------------------------------------------------------------------

    print("---- Begin training -----")    
    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 15000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=envs, agents=[agent])
    # start training
    trainer.train()

    # evaluate the agent(s)
    trainer.eval()

if __name__ == "__main__":
    main()