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
from torch.utils.data import Dataset, DataLoader
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
# Add scone gym
sconegym_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../NAIR_envs'))
sys.path.append(sconegym_path)
import sconegym # type: ignore
# Import algorithms models from nair PPO, SAC, DDPG
from nair_agents import get_models
init_dir = os.getcwd()

# -------------------------------
# EXPERT DATASET
# -------------------------------
class ExpertDatasetFromSTO(Dataset):
    def __init__(self, sto_files, obs_cols, act_cols, skip_rows=6):
        """
        Expert dataset built from Scone .sto log files.

        Args:
            sto_files (list[str]): list of paths to .sto files
            obs_cols (list[str]): column names used as observations
            act_cols (list[str]): column names used as actions
            skip_rows (int): number of rows to skip (default: 6, for .sto headers)
        """
        self.obs_data = []
        self.act_data = []

        for file_path in sto_files:
            df = pd.read_csv(file_path, sep='\t', comment='%', skiprows=skip_rows)
            
            # Drop rows with NaN values
            df = df.dropna()

            # Extract observation and action arrays
            obs = df[obs_cols].values.astype(np.float32)
            act = df[act_cols].values.astype(np.float32)

            self.obs_data.append(obs)
            self.act_data.append(act)

        # Concatenate all episodes into single arrays
        self.obs_data = torch.tensor(np.vstack(self.obs_data))
        self.act_data = torch.tensor(np.vstack(self.act_data))

    def __len__(self):
        return len(self.obs_data)

    def __getitem__(self, idx):
        # Return a single (observation, action) pair
        return self.obs_data[idx], self.act_data[idx]

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
# Validation of batch
# -------------------------------
from tensordict import TensorDict

def validate_batch(batch, required_keys=None):
    if required_keys is None:
        required_keys = ["observation", "reward", "action", "done", "advantage", "state_value", "value_target"]

    print("\n" + "="*10 + " VALIDANDO TENSORDICT " + "="*10)
    errors = False

    # Check if batch is empty
    if batch.batch_size == torch.Size([]) or len(batch.batch_size) == 0:
        print("[ERROR] Batch está vacío (batch_size=[])")
        errors = True

    # Check required keys
    for key in required_keys:
        if key not in batch:
            print(f"[ERROR] Falta la clave '{key}' en el batch.")
            errors = True
            continue

        val = batch.get(key)
        if val is None:
            print(f"[ERROR] batch['{key}'] es None.")
            errors = True
        elif not isinstance(val, torch.Tensor):
            print(f"[ERROR] batch['{key}'] no es un tensor.")
            errors = True
        elif torch.isnan(val).any():
            print(f"[ERROR] batch['{key}'] contiene NaN.")
            errors = True
        elif torch.isinf(val).any():
            print(f"[ERROR] batch['{key}'] contiene Inf.")
            errors = True
        else:
            print(f"[OK] '{key}' shape: {val.shape}, min: {val.min().item():.4f}, max: {val.max().item():.4f}")

    # Check nested 'next' tensordict
    next_td = batch.get("next", None)
    if next_td is not None:
        if isinstance(next_td, TensorDict):
            if "observation" in next_td:
                next_obs = next_td.get("observation")
                if next_obs is None or not isinstance(next_obs, torch.Tensor):
                    print("[ERROR] batch['next']['observation'] es inválido.")
                    errors = True
                else:
                    print(f"[OK] 'next.observation' shape: {next_obs.shape}")
            else:
                print("[ERROR] Falta 'observation' en batch['next'].")
                errors = True
        else:
            print("[ERROR] batch['next'] no es un TensorDict.")
            errors = True
    else:
        print("[ERROR] batch no contiene la clave 'next'.")
        errors = True

    print("="*10 + " VALIDACIÓN COMPLETA " + "="*10 + "\n")
    return not errors

# -------------------------------
# Config imitation learning and env with hydra
# -------------------------------
@hydra.main(config_path="./config", config_name="config_IL", version_base="1.1")
def main(cfg_hydra: DictConfig):
    start_time = time.time()
    today = datetime.now().strftime('%Y-%m-%d')
    time_now = datetime.now().strftime('%H-%M-%S')
    #shutil.copy(os.path.dirname(__file__) + "../../../../NAIR_envs/sconegym/nair_gaitgym.py", ".")
    shutil.copy(init_dir+"/IL_train.py", ".")
    checkpoint_dir = os.path.join(os.getcwd(), f"outputs/checkpoints")
    logs_dir = os.path.join(os.getcwd(), f"outputs/TF_logs")
    os.makedirs(checkpoint_dir, exist_ok=True)

    env_id = cfg_hydra.env.env_name
    num_cpu = cfg_hydra.env.num_cpu
    delays = cfg_hydra.env.use_delayed_sensors
    # Charge expert dataset 
    sto_dir = init_dir+"/datasets"
    sto_files = [os.path.join(sto_dir, f) for f in os.listdir(sto_dir) if f.endswith(".sto")]
    
    input_cols =  cfg_hydra.env.input_cols
    target_cols = cfg_hydra.env.target_cols

    #logdir = os.path.join(os.getcwd(), "runs", f"GAIL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=logs_dir)
    #Create env
    try:
        env = gym.vector.make(env_id, use_delayed_sensors=delays, num_envs=num_cpu, asynchronous=False)
        print("observation space env: ", env.observation_space)
        print("action space env: ", env.action_space)
    except gym.error.DeprecatedEnv as e:
        env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("nair")][0]
        print("sconewalk_h0918_osim-v1 not found. Trying {}".format(env_id))

    base_env = GymWrapper(env)
    env = TransformedEnv(base_env)
    #env.append_transform(ObservationNorm(in_keys=["observation"]))
    
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
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

"""
# -------------------------------
# MODELO TRANSFORMER
# -------------------------------
class GaitTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output = nn.Linear(64, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.activation(self.output(x))
    


# -------------------------------
# ENTRENAMIENTO
# -------------------------------
def main_2():
    dataset = MultipleSTODataset(sto_files, input_cols, target_cols, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = len(input_cols)
    output_dim = len(target_cols)

    teacher = GaitTransformer(input_dim, output_dim).to(DEVICE)

    student = GaitTransformer(input_dim, output_dim).to(DEVICE)

    teacher.load_state_dict(student.state_dict())
    teacher.eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    mse_loss = nn.MSELoss()
    losses = []

    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.no_grad():
                y_teacher = teacher(x)

            y_student = student(x)

            loss_mse = mse_loss(y_student, y)
            kl = F.kl_div(y_student.log(), y_teacher, reduction='batchmean')
            loss = loss_mse + LAMBDA_KL * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")

    # Plot final
    plt.plot(losses)
    plt.title("Pérdida total (MSE + λ·KL)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
"""

