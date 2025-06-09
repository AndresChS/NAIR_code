"""	
	Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
	source	:: https://github.com/AndresChS/NAIR_Code
"""
import sys
import os
import gym
import hydra 
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# Import algorithms models from nair PPO, SAC, DDPG
from nair_agents import get_models
# add scone gym using a relative path
sconegym_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../NAIR_envs')) 
sys.path.append(sconegym_path)
import sconegym
# Using time to define the unique naming
start_time = time.time()
today = datetime.now().strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
time_now = datetime.now().strftime('%H-%M-%S')  # Format: HH-MM-SS




agent = "PPO"  # o "PPO", "DDPG"
obs_dim = 96
act_dim = 18

actor, critic = get_models(agent, obs_dim, act_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor.to(device)
critic.to(device)

print(f"{agent.upper()} charged. Actor in {next(actor.parameters()).device}")

# === Hiperparámetros ===
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCH = 4
BATCH_SIZE = 64
ENTROPY_COEF = 0.01
MAX_EPISODES = 100
MAX_STEPS = 1000

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
            print(env_id, " not found. Trying {}".format(env_id))
        env = gym.vector.make(env_id, num_envs=2, asynchronous=False)
        env = gym.wrap_env(env)
        return env
    
    return _init


# ====================================================================
# Policy and value definition
# --------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(64, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))  # log(std) entrenable
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd)
        value = self.critic(x)
        return mean, std, value

# === Memoria PPO ===
class Memory:
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.logprobs, self.dones, self.values = [], [], []

    def store(self, state, action, reward, logprob, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logprobs.append(logprob)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

# === GAE simplificado ===
def compute_returns_advantages(rewards, dones, values, gamma=GAMMA, device='cpu'):
    returns = []
    G = 0
    for r, d in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1 - d)
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    advantages = returns - values
    return returns, advantages


@hydra.main(config_path=".", config_name="config_PPO", version_base="1.1")
def main(cfg_hydra): 

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
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = float(env.action_space.low[0])
    act_high = float(env.action_space.high[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    policy = ActorCritic(obs_dim, act_dim).to(device)
    memory = Memory()
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # === Bucle principal ===
    for ep in range(MAX_EPISODES):
        state = env.reset()
        ep_reward = 0

        for _ in range(MAX_STEPS):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            mean, std, value = policy(state_tensor)
            dist = Normal(mean, std)
            action = dist.sample()
            logprob = dist.log_prob(action).sum()

            # clip acción dentro de los límites del entorno
            action_clipped = action.clamp(act_low, act_high)
            next_state, reward, done, _ = env.step(action_clipped.detach().cpu().numpy())

            memory.store(state, action.detach().cpu().numpy(), reward, logprob.item(), done, value.item())

            state = next_state
            ep_reward += reward
            if done:
                break

        # === Actualización PPO ===
        returns, advantages = compute_returns_advantages(memory.rewards, memory.dones, memory.values, device=device)

        for _ in range(K_EPOCH):
            for i in range(0, len(memory.states), BATCH_SIZE):
                s = s = torch.tensor(np.array(memory.states[i:i+BATCH_SIZE]), dtype=torch.float32, device=device)
                a = torch.tensor(np.array(memory.actions[i:i+BATCH_SIZE]), dtype=torch.float32, device=device)
                old_logprobs = torch.tensor(memory.logprobs[i:i+BATCH_SIZE], device=device)
                adv = advantages[i:i+BATCH_SIZE].to(device)
                ret = returns[i:i+BATCH_SIZE].to(device)

                mean, std, values = policy(s)
                dist = Normal(mean, std)
                logprobs = dist.log_prob(a).sum(dim=1)
                entropy = dist.entropy().sum(dim=1).mean()

                ratios = torch.exp(logprobs - old_logprobs)
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * adv

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (ret - values.squeeze()).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        memory.clear()
        print(f"[EP {ep}] Recompensa total: {ep_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()