import optuna
import sys, os
import gym
import hydra
import torch
from sconetools import sconepy
import yaml
import numpy as np
# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer, Trainer
from skrl.utils import set_seed

# ====================================================================
# Scripts and paths administrations
# --------------------------------------------------------------------
from torch_gym_nair_H0918_ppo import Policy, Value

sconegym_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../NAIR_envs')) 
sys.path.append(sconegym_path)
import sconegym

# ====================================================================
# Policy and value definition
# --------------------------------------------------------------------
class CustomPPO(PPO):

    def __init__(self, *args, env=None, **kwargs):
        super().__init__(*args, **kwargs)
        if env is not None:
            self.envs = env  # si es un único entorno
            # self.envs = env   # si es vectorizado
set_seed()

# ====================================================================
# Model parameters loading and instantiation
# --------------------------------------------------------------------
# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
with open("config_PPO.yaml", "r") as file:
    config = yaml.safe_load(file)
print(config)
config_env = config["env"]
config_logger = config["logger"]
config_optim = config["optim"]
config_hiperparameters = config["hiperparameters"]

cfg = PPO_DEFAULT_CONFIG.copy()
#cfg["rollouts"] = config_hiperparameters["batch_size"]  # memory_size
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

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = config_logger["write_interval"]
cfg["experiment"]["checkpoint_interval"] = config_logger["check_interval"]
cfg["experiment"]["directory"] = config_env["torch_dir"]


def objective(trial):
    # Suggesting hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # Optuna Parameters config PPO 
    cfg["mini_batches"] = batch_size
    cfg["lambda"] = gamma
    cfg["learning_rate"] = learning_rate
    agent_config = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "batch_size": batch_size,
        # añade más si es necesario
    }

    # ====================================================================
    # Create vectorized environment 
    # --------------------------------------------------------------------
    try:
        env = gym.vector.make(config_env["env_name"], use_delayed_sensors=True, num_envs=config_env["num_cpu"], asynchronous=False)
    except gym.error.DeprecatedEnv as e:
        env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("nair")][0]
        print(config_env["env_name"], " not found. Trying {}".format(env_id))
    # Wrap environmet for use torch tensors operations
    env = wrap_env(env)
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
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    
    agent = CustomPPO(models=models,
            memory=None,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            env=env
            )
    
    cfg_trainer = {"timesteps": 10000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
    trainer.train()

    # Evalúa el rendimiento
    avg_reward = env._rewards #evaluate_agent(agent, env)  # define esta función también
    #avg_reward = trainer.eval(env, agent, episodes=5, deterministic=True)

    env.close()
    return float(np.mean(avg_reward))

study = optuna.create_study(direction="maximize")  # o "minimize"
study.optimize(objective, n_trials=50)

print("Best trial:")
print(study.best_trial.params)