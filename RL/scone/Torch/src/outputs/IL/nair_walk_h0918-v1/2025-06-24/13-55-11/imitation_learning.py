# Reescribimos el código para que utilice MultipleSTODataset en lugar de un único DataFrame

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
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
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules import ProbabilisticActor, ValueOperator
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


import torch
import torch.nn as nn

# -------------------------------
# DATASET SECUENCIAL MULTIPLE
# -------------------------------
class MultipleSTODataset(Dataset):
    def __init__(self, sto_files, input_cols, target_cols, seq_len=50):
        self.seq_len = seq_len
        self.input_seqs = []
        self.target_seqs = []

        for file_path in sto_files:
            df = pd.read_csv(file_path, sep='\t', comment='%', skiprows=6)
            inputs = df[input_cols].values
            targets = df[target_cols].values

            for i in range(len(df) - seq_len):
                self.input_seqs.append(torch.tensor(inputs[i:i+seq_len], dtype=torch.float32))
                self.target_seqs.append(torch.tensor(targets[i:i+seq_len], dtype=torch.float32))

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        return self.input_seqs[idx], self.target_seqs[idx]

class MusclePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Sigmoid()  # activaciones musculares entre 0 y 1
        )
    
    def forward(self, obs):
        return self.net(obs)
    
# -------------------------------
# Config imitation learning and env with hydra
# -------------------------------
@hydra.main(config_path="./config", config_name="config_IL", version_base="1.1")
def main(cfg_hydra: DictConfig):
    start_time = time.time()
    today = datetime.now().strftime('%Y-%m-%d')
    time_now = datetime.now().strftime('%H-%M-%S')
    #shutil.copy(os.path.dirname(__file__) + "../../../../NAIR_envs/sconegym/nair_gaitgym.py", ".")
    shutil.copy(init_dir+"/imitation_learning.py", ".")
    output_dir = os.path.join(os.getcwd(), f"outputs/checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    env_id = cfg_hydra.env.env_name
    num_cpu = cfg_hydra.env.num_cpu
    delays = cfg_hydra.env.use_delayed_sensors

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

    policy = MusclePolicy(obs_dim=..., act_dim=...)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    loss_fn = nn.MSELoss()
    sto_dir = "datasets"
    sto_files = [os.path.join(sto_dir, f) for f in os.listdir(sto_dir) if f.endswith(".sto")]
    SEQ_LEN = 50
    BATCH_SIZE = 32
    LAMBDA_KL = 0.1
    EPOCHS = 10
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_cols = [
        'pelvis_tilt', 'pelvis_tilt_u', 'pelvis_tx', 'pelvis_tx_u', 'pelvis_ty',
        'pelvis_ty_u', 'hip_flexion_r', 'hip_flexion_r_u', 'knee_angle_r',
        'knee_angle_r_u', 'ankle_angle_r', 'ankle_angle_r_u', 'hip_flexion_l',
        'hip_flexion_l_u', 'knee_angle_l', 'knee_angle_l_u', 'ankle_angle_l',
        'ankle_angle_l_u', 'upperexo_hinge', 'upperexo_hinge_u', 'upperexo_slide',
        'upperexo_slide_u', 'exo_rotation', 'exo_rotation_u', 'lowerexo_hinge',
        'lowerexo_hinge_u'
    ]

    target_cols = [
        'hamstrings_r.activation', 'bifemsh_r.activation', 'glut_max_r.activation',
        'iliopsoas_r.activation', 'vas_int_r.activation', 'gastroc_r.activation',
        'soleus_r.activation', 'tib_ant_r.activation',
        'hamstrings_l.activation', 'bifemsh_l.activation', 'glut_max_l.activation',
        'iliopsoas_l.activation', 'vas_int_l.activation', 'gastroc_l.activation',
        'soleus_l.activation', 'tib_ant_l.activation'
    ]
    dataset = MultipleSTODataset(sto_files, input_cols, target_cols, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        for obs, target_act in dataloader:
            pred = policy(obs)
            loss = loss_fn(pred, target_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

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

