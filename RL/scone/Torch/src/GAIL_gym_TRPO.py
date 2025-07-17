# From Musculoco 
"""
from time import perf_counter
from contextlib import contextmanager
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length, parse_dataset
from mushroom_rl.core.logger.logger import Logger
from imitation_lib.imitation import GAIL_TRPO
from imitation_lib.utils import FullyConnectedNetwork, DiscriminatorNetwork, NormcInitializer, \
                         GailDiscriminatorLoss
from imitation_lib.utils import BestAgentSaver

from mushroom_rl.core.serialization import *

from imitation_lib.musculoco_il.algorithms.GAIL_KL_objective import TargetEntropyUniform2NormalKLDGAIL, Uniform2NormalKLDGAIL, \
    TargetEntropyUniform2MultivariateGaussianKLDGAIL, Uniform2MultivariateGaussianKLDGAIL
from imitation_lib.musculoco_il.policy.gaussian_torch_policy import OptionalGaussianTorchPolicy
from imitation_lib.musculoco_il.policy.latent_exploration_torch_policy import LatentExplorationPolicy
from imitation_lib.musculoco_il.util.preprocessors import StateSelectionPreprocessor
from imitation_lib.musculoco_il.util.rewards import OutOfBoundsActionCost
from imitation_lib.musculoco_il.util.standardizer import Standardizer
"""
# From previous works
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import hydra
import gym
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.objectives import ClipPPOLoss, PPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules import ProbabilisticActor, ValueOperator, MLP, TanhNormal, SafeModule
from tensordict.nn import TensorDictModule, TensorDictSequential
from omegaconf import DictConfig
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

# NAIR library
from utils.validation import validate_batch
from utils.training import BestAgentSaver
from utils.dataset import ExpertDatasetFromSTO
from lib.nair.agents.nair_agents import get_models  # Import algorithms models from nair PPO, SAC, DDPG
# Add scone gym
sconegym_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../NAIR_envs'))
sys.path.append(sconegym_path)
import sconegym # type: ignore

init_dir = os.getcwd()

# -------------------------------
# DEFINE DISCRIMINATOR
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

def sample_from(dataset, batch_size=128):
    indices = torch.randint(0, len(dataset), (batch_size,))
    obs = []
    act = []
    for idx in indices:
        o, a = dataset[idx]
        obs.append(o)
        act.append(a)
    return torch.stack(obs), torch.stack(act)

# -------------------------------
# Config imitation learning and env with hydra
# -------------------------------
@hydra.main(config_path="./config", config_name="config_GAIL_TRPO", version_base="1.1")
def main(cfg_hydra: DictConfig):

    # Files and paths administration
    start_time = time.time()
    today = datetime.now().strftime('%Y-%m-%d')
    time_now = datetime.now().strftime('%H-%M-%S')
    #shutil.copy(os.path.dirname(__file__) + "../../../../NAIR_envs/sconegym/nair_gaitgym.py", ".")
    #shutil.copy(init_dir+"/IL_train.py", ".")
    checkpoint_dir = os.path.join(os.getcwd(), f"outputs/checkpoints")
    # Tensorlogs
    logs_dir = os.path.join(os.getcwd(), f"outputs/TF_logs")
    writer = SummaryWriter(log_dir=logs_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Charge expert dataset 
    sto_dir = init_dir+"/datasets"
    sto_files = [os.path.join(sto_dir, f) for f in os.listdir(sto_dir) if f.endswith(".sto")]
    input_cols =  cfg_hydra.env.input_cols
    target_cols = cfg_hydra.env.target_cols

    # Environment definition
    env_id = cfg_hydra.env.env_name
    num_cpu = cfg_hydra.env.num_cpu
    delays = cfg_hydra.env.use_delayed_sensors
    
# -------------------------------
# Create environment
# -------------------------------
    try:
        env = gym.vector.make(env_id, use_delayed_sensors=delays, num_envs=num_cpu, asynchronous=False)
        print("observation space env: ", env.observation_space)
        print("action space env: ", env.action_space)
    except gym.error.DeprecatedEnv as e:
        env_id_suggest = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("nair")][0]
        print(env_id, " not found. Trying {}".format(env_id_suggest))

    base_env = GymWrapper(env)
    env = TransformedEnv(base_env)
    #env.append_transform(ObservationNorm(in_keys=["observation"]))
    
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    
# -------------------------------
# Create Agent
# -------------------------------    
    agent = cfg_hydra.env.algorithm  # PPO, DDPG, MPO, SAC, TRPO
    actor, critic = get_models(agent, obs_dim, act_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = env.to(device)
    actor.to(device)
    critic.to(device)
    print("obs_dim:", obs_dim, "act_dim:", act_dim)
    print("observation_spec:", env.observation_spec)
    print("action_spec:", env.action_spec)

    policy = TensorDictModule(
        actor,
        in_keys=["observation"],
        out_keys=["loc", "scale"]
    )

    
    policy = ProbabilisticActor(
        module=policy,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True
    )
    
    value = nn.Sequential(
        nn.Linear(obs_dim, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 1),
    ).to(device)  # <- Aquí mueves la red al dispositivo


    value = ValueOperator(
        module=value,
        in_keys=["observation"],
    )
    #print("action_spec:", env.action_spec)
    #print("Running policy:", policy(env.reset()))
    #print("Running value:", value(env.reset()))

    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=cfg_hydra.collector.frames_per_batch,
        total_frames=cfg_hydra.collector.total_frames,
        split_trajs=False,
        device=device,
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg_hydra.collector.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=cfg_hydra.hiperparameters.gamma,    
        lmbda=cfg_hydra.hiperparameters.lmbda, 
        value_network=value, 
        average_gae=True, 
        device=device,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value,
        clip_epsilon=cfg_hydra.hiperparameters.clip_epsilon,
        entropy_bonus=bool(cfg_hydra.hiperparameters.entropy_coef),
        entropy_coef=cfg_hydra.hiperparameters.entropy_coef,
        # these keys match by default but we set this for completeness
        critic_coef=cfg_hydra.hiperparameters.critic_coef,
        loss_critic_type="smooth_l1",
    )
    
    # Buffers
    agent_buffer = ReplayBuffer()
    expert_buffer = ExpertDatasetFromSTO(sto_files, input_cols, target_cols)
    # Discriminador
    discriminator = Discriminator(obs_dim, act_dim)
    discriminator.to(device)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    # RL optimizer (ej: PPO)
    rl_optimizer = torch.optim.Adam(policy.parameters(), cfg_hydra.optim.lr)

# -------------------------------
# Trainning loop
# -------------------------------
    n_iters = 1000 
    disc_updates = 1   #discriminator updates
    global_step = 0
    best_reward = -float("inf")
    logs = defaultdict(list)
    pbar = tqdm(total=cfg_hydra.collector.total_frames)

    for iteration in range(n_iters):
        for batch in collector:
            batch_size = batch.numel()
            global_step += batch_size
            pbar.update(batch_size)

            agent_obs = batch["observation"]
            agent_act = batch["action"]

            # === Train discriminator ===
            for _ in range(disc_updates):
                exp_obs, exp_act = sample_from(expert_buffer, batch_size=cfg_hydra.hiperparameters.batch_size)
                exp_obs, exp_act = exp_obs.to(device), exp_act.to(device)
                ag_obs = agent_obs[:cfg_hydra.hiperparameters.batch_size].to(device)
                ag_act = agent_act[:cfg_hydra.hiperparameters.batch_size].to(device)

                exp_preds = discriminator(exp_obs, exp_act)
                ag_preds = discriminator(ag_obs, ag_act)
                loss_disc = -torch.mean(torch.log(exp_preds + 1e-8) + torch.log(1 - ag_preds + 1e-8))

                disc_optimizer.zero_grad()
                loss_disc.backward()
                disc_optimizer.step()

            logs["loss_disc"].append(loss_disc.item())

            # === Assign synthetic reward ===
            with torch.no_grad():
                agent_rewards = -torch.log(1 - discriminator(agent_obs, agent_act) + 1e-8).view(-1, 1)
                batch.set("reward", agent_rewards)
                batch["next"]["reward"] = agent_rewards.clone()

            # === Compute advantage ===
            advantage_module(batch)

            # === Policy update ===
            loss_values = loss_module(batch)
            total_loss = loss_values["loss_objective"]
            rl_optimizer.zero_grad()
            total_loss.backward()
            rl_optimizer.step()

            logs["loss_policy"].append(total_loss.item())
            if global_step % 1_000 < batch_size:
                writer.add_scalar("loss/discriminator", loss_disc.item(), global_step)
                writer.add_scalar("loss/ppo", total_loss.item(), global_step)
                writer.add_scalar("reward/fake_env", agent_rewards.mean().item(), global_step)
            # === Optional: save best reward ===
            avg_reward = agent_rewards.mean().item()
            logs["avg_reward"].append(avg_reward)
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_path = os.path.join(checkpoint_dir, f"best_agent.pt")
                torch.save({
                    "step": global_step,
                    "actor": policy.state_dict(),
                    "critic": value.state_dict(),
                    "disc": discriminator.state_dict(),
                    "optimizer_policy": rl_optimizer.state_dict(),
                    "optimizer_disc": disc_optimizer.state_dict(),
                }, best_path)
                print(f"[Iter {iteration}] Loss Disc: {loss_disc.item():.4f} | Loss PPO: {total_loss.item():.4f} | Reward: {agent_rewards.mean().item():.4f}")

            # === Save regular checkpoints ===
            if global_step % 10_000 < batch_size:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "actor": policy.state_dict(),
                    "critic": value.state_dict(),
                    "disc": discriminator.state_dict(),
                    "optimizer_policy": rl_optimizer.state_dict(),
                    "optimizer_disc": disc_optimizer.state_dict(),
                }, checkpoint_path)
                print(f"[Iter {iteration}] Loss Disc: {loss_disc.item():.4f} | Loss PPO: {total_loss.item():.4f} | Reward: {agent_rewards.mean().item():.4f}")


    pbar.close()

if __name__ == "__main__":
    main()
