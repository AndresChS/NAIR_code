from myosuite import gym
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import numpy as np
import os, time, multiprocessing
import json
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm


# Define the environment name and model path
env_name = "ExoLegSpasticityFlexoExtEnv-v0"     #"ExoLeg40MuscFlexoExtEnv-v0"   
                                                #"ExoLegPassiveFlexoExtEnv-v0"
                                                #"ExoLegSpasticityFlexoExtEnv-v0"
log_path = f'/home/nair-group/achs/NAIR_Code/code/RL/SB3/code/RL/SB3/outputs/ExoLegSpasticityFlexoExtEnv-v0/2024-11-21/10-17-05'

# Define the path to the best model saved
best_model_path = os.path.join(log_path, 'best_model.zip')  # Ensure the correct file extension

def create_env():
    env = gym.make(env_name)

    return env
def calculate_asymmetric_deviation(data):
    """
    Calcula la desviación asimétrica hacia arriba y hacia abajo respecto a la media,
    para cada paso de tiempo.

    Args:
        data (np.ndarray): Array de datos con forma (n_simulaciones, n_steps).

    Returns:
        tuple: (media, desviación_hacia_arriba, desviación_hacia_abajo), todos con forma (n_steps,).
    """
    mean = np.mean(data, axis=0)  # Media para cada paso de tiempo
    above_mean = np.where(data >= mean, data - mean, 0)  # Solo valores por encima de la media
    below_mean = np.where(data < mean, mean - data, 0)  # Solo valores por debajo de la media

    deviation_up = np.mean(above_mean, axis=0)
    deviation_down = np.mean(below_mean, axis=0)

    return mean, deviation_up, deviation_down


# Create the vectorized environment
env = DummyVecEnv([create_env])

# Load the best model
try:
    model = SAC.load(best_model_path)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {best_model_path}")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")
n_steps = 1024
n_it_lv = 10
reward_matrix = np.zeros((4*n_it_lv, n_steps))
error_matrix = np.zeros((4*n_it_lv, n_steps))
control_matrix = np.zeros((4*n_it_lv, n_steps))
inter_matrix = np.zeros((4*n_it_lv, n_steps))
vel_matrix = np.zeros((4*n_it_lv, n_steps))

# Colores para cada nivel de espasticidad
colors = ['blue', 'orange', 'green', 'red']
labels = [f"Level {i}" for i in range(4)]  # Etiquetas para los niveles

# Bucle para obtener datos para cada nivel de espasticidad
for lv in range(4):  # Suponiendo 4 niveles de espasticidad
    for i in tqdm(range(n_it_lv), desc=f"Nivel de espasticidad {lv}"):  # n_it_lv simulaciones por nivel de espasticidad
        env.envs[0].set_spasticity_level(spasticity_level=lv)
        obs = env.reset()
        total_reward = 0
        for j in range(n_steps):
            action, _states = model.predict(obs)
            action[:, 1:] = 0 
            obs, rewards, dones, info = env.step(action)
            reward_matrix[i + lv * n_it_lv, j] = rewards[0]
            total_reward += rewards[0]
            data = env.envs[0].sim.data
            
            error_matrix[i + lv * n_it_lv, j] = data.qpos[6] - 0.2
            vel_matrix[i + lv * n_it_lv, j] = data.qvel[6]#data.ctrl[13]
            control_matrix[i + lv * n_it_lv, j] = data.ctrl[0]
            inter_matrix[i + lv * n_it_lv, j] = -obs[0][-1]#data.qfrc_actuator[6]#-obs[0][-1]
 # Crear figuras
fig, axs = plt.subplots(3, 1, figsize=(20, 15))

# Graficar cada métrica por nivel de espasticidad
for lv in range(4):
    start_idx = lv * n_it_lv
    end_idx = (lv + 1) * n_it_lv

    # Calcular desviaciones asimétricas
    error_mean, error_dev_up, error_dev_down = calculate_asymmetric_deviation(
        error_matrix[start_idx:end_idx, :]
    )
    control_mean, control_dev_up, control_dev_down = calculate_asymmetric_deviation(
        control_matrix[start_idx:end_idx, :]
    )
    vel_mean, vel_dev_up, vel_dev_down = calculate_asymmetric_deviation(
        vel_matrix[start_idx:end_idx, :]
    )
    inter_mean, inter_dev_up, inter_dev_down = calculate_asymmetric_deviation(
        inter_matrix[start_idx:end_idx, :]
    )

    # Graficar Error
    axs[0].plot(range(n_steps), error_mean, label=f'Nivel {lv}', color=colors[lv])
    axs[0].fill_between(
        range(n_steps),
        error_mean - error_dev_down,
        error_mean + error_dev_up,
        color=colors[lv],
        alpha=0.3,
        label='_nolegend_',
    )

    # Graficar Control
    axs[1].plot(range(n_steps), control_mean, label=f'Nivel {lv}', color=colors[lv])
    axs[1].fill_between(
        range(n_steps),
        control_mean - control_dev_down,
        control_mean + control_dev_up,
        color=colors[lv],
        alpha=0.3,
        label='_nolegend_',
    )
    """
    # Graficar Velocidad
    axs[2].plot(range(n_steps), vel_mean, label=f'Nivel {lv}', color=colors[lv])
    axs[2].fill_between(
        range(n_steps),
        vel_mean - vel_dev_down,
        vel_mean + vel_dev_up,
        color=colors[lv],
        alpha=0.3,
        label='_nolegend_',
    )
    """
    # Graficar Interacción
    axs[2].plot(range(n_steps), inter_mean, label=f'Nivel {lv}', color=colors[lv])
    axs[2].fill_between(
        range(n_steps),
        inter_mean - inter_dev_down,
        inter_mean + inter_dev_up,
        color=colors[lv],
        alpha=0.3,
        label='_nolegend_',
    )



axs[0].set_ylabel("Error position")
axs[1].set_ylabel("Control input")
#axs[2].set_ylabel("Velocity (rads/s)")
axs[2].set_ylabel("Interaction (Nm)")

# Colocar la leyenda global en la parte superior
fig.legend(
    labels,  # Etiquetas de la leyenda
    loc='upper center',  # Posición en la parte superior central
    ncol=4,  # Número de columnas
    frameon=False,  # Sin borde
    bbox_to_anchor=(0.5, 1),  # Ajustar posición vertical de la leyenda
)

plt.tight_layout(rect=[0, 0.01, 1, 0.97])  # Ajustar el espacio para que la leyenda no sobreponga las gráficas
plt.savefig("spasticity_levels_plot_asymmetric.png")
plt.show()