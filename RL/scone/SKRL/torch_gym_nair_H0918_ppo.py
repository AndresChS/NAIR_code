"""	
	Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
	source	:: https://github.com/AndresChS/NAIR_Code
"""

import gym
import sconegym
import torch
import torch.nn as nn
import hydra 
import time
from datetime import datetime
import shutil

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# Using time to define the unique naming
start_time = time.time()
today = datetime.now().strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
time_now = datetime.now().strftime('%H-%M-%S')  # Format: HH-MM-SS

# ====================================================================
# Make environment
# --------------------------------------------------------------------
def make_env(env_id: str, num_env = 0, seed: int = 0, delayed_sensors=True):
    """
    Utility function for multiprocessed env.
    env_id: the environment ID
    num_env: index of the subprocess 
    """
    def _init():
        try:
            env = gym.vector.make(env_id, use_delayed_sensors=delayed_sensors, num_envs=num_env, asynchronous=False)
        except gym.error.DeprecatedEnv as e:
            env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("nair")][0]
            print("sconewalk_h0918_osim-v1 not found. Trying {}".format(env_id))
        env = gym.vector.make(env_id, num_envs=4, asynchronous=False)
        env = wrap_env(env)
        device = env.device
        return env
    
    return _init

# ====================================================================
# Policy and value definition
# --------------------------------------------------------------------

class CustomPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post_interaction(self, timestep: int, timesteps: int):
        super().post_interaction(timestep, timesteps)
        #print("acciones", env.observations)
        if timestep % 1000 == 0:
            print(f"[Custom PPO] Paso {timestep} de {timesteps}")
            #print("reward", self.env._rewards)

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


@hydra.main(config_path=".", config_name="config_PPO")
def main(cfg_hydra): 

    env_id = cfg_hydra.env.env_name
    num_cpu = cfg_hydra.env.num_cpu
    delays = cfg_hydra.env.use_delayed_sensors

    # meake and wrap the gym environment.
    try:
        env = gym.vector.make(env_id, use_delayed_sensors=delays, num_envs=num_cpu, asynchronous=False)
        print("observation space env", env.observation_space)
    except gym.error.DeprecatedEnv as e:
        env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("nair")][0]
        print("sconewalk_h0918_osim-v1 not found. Trying {}".format(env_id))
    env = wrap_env(env)
    print("observation space wrap", env.observation_space)
    # note: the environment version may change depending on the gym version
    device = env.device
    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=cfg_hydra.hiperparameters.batch_size, num_envs=env.num_envs, device=device)

    # ====================================================================
    # Model Definition
    # --------------------------------------------------------------------
    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
    models["value"] = Value(env.observation_space, env.action_space, device)
    print("Action space:",env.action_space, "   Observation space: ", env.observation_space)

    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = cfg_hydra.hiperparameters.batch_size  # memory_size
    cfg["learning_epochs"] = cfg_hydra.hiperparameters.learning_epochs
    cfg["mini_batches"] = cfg_hydra.hiperparameters.mini_batches
    cfg["discount_factor"] = cfg_hydra.hiperparameters.discount_factor
    cfg["lambda"] = cfg_hydra.hiperparameters.lambda_factor
    cfg["learning_rate"] = cfg_hydra.optim.lr
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = cfg_hydra.optim.optimizer_kwargs
    cfg["grad_norm_clip"] = cfg_hydra.hiperparameters.clip_norm
    cfg["ratio_clip"] = cfg_hydra.hiperparameters.ratio_clip
    cfg["value_clip"] = cfg_hydra.hiperparameters.value_clip
    cfg["clip_predicted_values"] = cfg_hydra.hiperparameters.clip_predicted_values
    cfg["entropy_loss_scale"] = cfg_hydra.hiperparameters.entropy_loss
    cfg["value_loss_scale"] = cfg_hydra.hiperparameters.value_loss
    cfg["kl_threshold"] = cfg_hydra.hiperparameters.kl_threshold

    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = cfg_hydra.logger.write_interval
    cfg["experiment"]["checkpoint_interval"] = cfg_hydra.logger.check_interval
    cfg["experiment"]["experiment_name"] = "outputs"
    cfg["experiment"]["directory"] = ""
    cfg["experiment"]["create"] = False

    agent = CustomPPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device
                )


    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": cfg_hydra.collector.num_steps, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

    shutil.copy(cfg_hydra.env.path, ".")
    shutil.copy("/home/achs/Documents/AChS/PHD/code/NAIR_Code/envs/sconegym/torch_gym_nair_H0918_ppo.py", ".")
    # start training
    trainer.train()
    env.close()

if __name__ == "__main__":
    main()