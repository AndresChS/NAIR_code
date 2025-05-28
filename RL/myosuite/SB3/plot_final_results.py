#=================================================
#	This model is generated with tacking the Myosuite conversion of [Rajagopal's full body gait model](https://github.com/opensim-org/opensim-models/tree/master/Models/RajagopalModel) as close
#reference.
#	Model	  :: Myo Leg 1 Dof 40 Musc Exo (MuJoCoV2.0)
#	Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
#	source	:: https://github.com/AndresChS/NAIR_Code
#	====================================================== -->

from myosuite import gym
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


# Define the environment name and model path
env_name = "ExoLegSpasticityFlexoExtEnv-v0"     #"ExoLeg40MuscFlexoExtEnv-v0"   
                                                #"ExoLegPassiveFlexoExtEnv-v0"
                                                #"ExoLegSpasticityFlexoExtEnv-v0"
log_path = f'/home/nair-group/achs/NAIR_Code/code/RL/SB3/code/RL/SB3/outputs/ExoLegSpasticityFlexoExtEnv-v0/2024-11-15/14-21-38'

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
n_steps = 1536
n_it_lv = 1
reward_matrix = np.zeros((4*n_it_lv, n_steps))
error_matrix = np.zeros((4*n_it_lv, n_steps))
control_matrix = np.zeros((4*n_it_lv, n_steps))
inter_matrix = np.zeros((4*n_it_lv, n_steps))

# Colores para cada nivel de espasticidad
colors = ['blue', 'ab1a28', 'green', 'red']

# Bucle para obtener datos para cada nivel de espasticidad
for lv in range(4):  # Suponiendo 4 niveles de espasticidad
    for i in tqdm(range(n_it_lv), desc=f" RL Nivel de espasticidad {lv}"):  # n_it_lv simulaciones por nivel de espasticidad
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
            control_matrix[i + lv * n_it_lv, j] = data.ctrl[0]
            inter_matrix[i + lv * n_it_lv, j] = data.qfrc_actuator[6]#-obs[0][-1]


#======================================================
# PID results
#====================================================== -->
import numpy as np
from myosuite.utils import gym
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def test(value):
    print(type(value))
    print(value)
    return value

# PID controller definition
class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute(self, current_position, dt, setpoint):
        # Setpoint if change during the simulation
        self.setpoint = setpoint
        # Error
        error = self.setpoint - current_position
        
        # Integral term
        self.integral += error * dt
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        
        # Output PID controller
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Limit output to range [-100, 100]
        output = np.clip(output, -1, 1) 

        # Actualization
        self.prev_error = error
        
        return output,error

# Initial guess for PID parameters
kp = 7
ki = 0
kd = 0.05
setpoint = 0.2
params_list = [ki, kp, kd]
optimized = 1
# Perform the optimization


env = gym.make('ExoLegSpasticityFlexoExtEnv-v0')
env.set_spasticity_level(spasticity_level=1)
data = env.sim.data
dt = env.sim.model.opt.timestep    
pid_error_matrix = np.zeros((4*n_it_lv, n_steps))
pid_control_matrix = np.zeros((4*n_it_lv,n_steps))
pid_inter_matrix = np.zeros((4*n_it_lv, n_steps))
pid_controller = PIDController(kp, ki, kd, setpoint)

# Bucle para obtener datos para cada nivel de espasticidad
for lv in range(4):  # Suponiendo 4 niveles de espasticidad
    for i in tqdm(range(n_it_lv), desc=f"PID Nivel de espasticidad {lv}"):  # n_it_lv simulaciones por nivel de espasticidad
        env.set_spasticity_level(spasticity_level=lv)
        obs = env.reset()
        total_reward = 0
        for j in range(n_steps):
            actions = np.zeros(41)
            current_position = data.qpos[env.sim.model.joint_name2id('knee_angle_r')]
            torque, PID_error = pid_controller.compute(current_position, dt, setpoint)
            actions[0] = -torque
            obs, rewards, *_ = env.step(actions)
            
            data = env.sim.data
            
            pid_error_matrix[i + lv * n_it_lv, j] = -PID_error
            pid_control_matrix[i + lv * n_it_lv, j] = actions[0]
            pid_inter_matrix[i + lv * n_it_lv, j] = data.qfrc_actuator[6]


# Calcular límites globales para cada métrica (mínimo y máximo en todos los datos)
error_min = min(np.min(error_matrix), np.min(pid_error_matrix))
error_max = max(np.max(error_matrix), np.max(pid_error_matrix))

control_min = -1.1 #min(np.min(control_matrix), np.min(pid_control_matrix))
control_max = 1.1 #max(np.max(control_matrix), np.max(pid_control_matrix))

inter_min = -200 #min(np.min(inter_matrix), np.min(pid_inter_matrix))
inter_max = max(np.max(inter_matrix), np.max(pid_inter_matrix))

colors = ['orange', 'black']  # Colores personalizados
# Crear figura de 3x4 para las gráficas
fig, axs = plt.subplots(3, 4, figsize=(20, 12))

# Bucle para graficar cada nivel de espasticidad en una columna
for lv in range(4):
    start_idx = lv * n_it_lv
    end_idx = (lv + 1) * n_it_lv

    # Calcular desviaciones asimétricas y medias para SAC (RL)
    error_mean, error_dev_up, error_dev_down = calculate_asymmetric_deviation(
        error_matrix[start_idx:end_idx, :]
    )
    control_mean, control_dev_up, control_dev_down = calculate_asymmetric_deviation(
        control_matrix[start_idx:end_idx, :]
    )
    inter_mean, inter_dev_up, inter_dev_down = calculate_asymmetric_deviation(
        inter_matrix[start_idx:end_idx, :]
    )

    # Calcular desviaciones asimétricas y medias para PID
    pid_error_mean, pid_error_dev_up, pid_error_dev_down = calculate_asymmetric_deviation(
        pid_error_matrix[start_idx:end_idx, :]
    )
    pid_control_mean, pid_control_dev_up, pid_control_dev_down = calculate_asymmetric_deviation(
        pid_control_matrix[start_idx:end_idx, :]
    )
    pid_inter_mean, pid_inter_dev_up, pid_inter_dev_down = calculate_asymmetric_deviation(
        pid_inter_matrix[start_idx:end_idx, :]
    )

    # Graficar Error (fila 1)
    axs[0, lv].plot(range(n_steps), pid_error_mean, label='PID', linewidth=2, color=colors[1])
    axs[0, lv].fill_between(
        range(n_steps),
        pid_error_mean - pid_error_dev_down,
        pid_error_mean + pid_error_dev_up,
        color=colors[1],
        alpha=0.3,
    )
    axs[0, lv].plot(range(n_steps), error_mean, label='SAC', linewidth=2, color=colors[0])
    axs[0, lv].fill_between(
        range(n_steps),
        error_mean - error_dev_down,
        error_mean + error_dev_up,
        color=colors[0],
        alpha=0.3,
    )
    axs[0, lv].set_ylim(error_min, error_max)
    if lv == 0:
        axs[0, lv].set_ylabel("Error")
        #axs[0, lv].legend()
    else:
        axs[0, lv].tick_params(labelleft=False)  # Ocultar etiquetas del eje y en otras columnas
    axs[0, lv].set_title(f'Level {lv}')

    # Graficar Control (fila 2)
    axs[1, lv].plot(range(n_steps), pid_control_mean, label='PID', linewidth=2, color=colors[1])
    axs[1, lv].fill_between(
        range(n_steps),
        pid_control_mean - pid_control_dev_down,
        pid_control_mean + pid_control_dev_up,
        color=colors[1],
        alpha=0.3,
    )
    axs[1, lv].plot(range(n_steps), control_mean, label='SAC', linewidth=2, color=colors[0])
    axs[1, lv].fill_between(
        range(n_steps),
        control_mean - control_dev_down,
        control_mean + control_dev_up,
        color=colors[0],
        alpha=0.3,
    )
    axs[1, lv].set_ylim(control_min, control_max)
    if lv == 0:
        axs[1, lv].set_ylabel("Control")
    else:
        axs[1, lv].tick_params(labelleft=False)  # Ocultar etiquetas del eje y en otras columnas

    # Graficar Interacción (fila 3)
    axs[2, lv].plot(range(n_steps), pid_inter_mean, label='PID', linewidth=2, color=colors[1])
    axs[2, lv].fill_between(
        range(n_steps),
        pid_inter_mean - pid_inter_dev_down,
        pid_inter_mean + pid_inter_dev_up,
        color=colors[1],
        alpha=0.3,
    )
    axs[2, lv].plot(range(n_steps), inter_mean, label='SAC', linewidth=2, color=colors[0])
    axs[2, lv].fill_between(
        range(n_steps),
        inter_mean - inter_dev_down,
        inter_mean + inter_dev_up,
        color=colors[0],
        alpha=0.3,
    )
    axs[2, lv].set_ylim(inter_min, inter_max)
    if lv == 0:
        axs[2, lv].set_ylabel("Interaction (N/m)")
    else:
        axs[2, lv].tick_params(labelleft=False)  # Ocultar etiquetas del eje y en otras columnas

    # Eliminar etiquetas del eje x en todas las subgráficas
    axs[0, lv].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[1, lv].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Colocar la leyenda global en la parte superior
fig.legend(
    ["Mean PID", "STD PID", "Mean SAC", "STD SAC"],  # Etiquetas de la leyenda
    loc='upper center',
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1),
)

plt.tight_layout(rect=[0, 0.01, 1, 0.95])
plt.savefig("comparison_plot_pid_vs_rl_asymmetric.png")
plt.show()

# Guardar matrices de resultados
output_dir = "output_data2"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "reward_matrix.npy"), reward_matrix)
np.save(os.path.join(output_dir, "error_matrix.npy"), error_matrix)
np.save(os.path.join(output_dir, "control_matrix.npy"), control_matrix)
np.save(os.path.join(output_dir, "inter_matrix.npy"), inter_matrix)

np.save(os.path.join(output_dir, "pid_error_matrix.npy"), pid_error_matrix)
np.save(os.path.join(output_dir, "pid_control_matrix.npy"), pid_control_matrix)
np.save(os.path.join(output_dir, "pid_inter_matrix.npy"), pid_inter_matrix)

# Guardar datos procesados para boxplots
boxplot_data = {
    "spasticity_level": [],
    "metric": [],
    "value": [],
}

for lv in range(4):
    start_idx = lv * n_it_lv
    end_idx = (lv + 1) * n_it_lv

    # Agregar datos de error
    boxplot_data["spasticity_level"].extend([lv] * n_it_lv * n_steps)
    boxplot_data["metric"].extend(["Error"] * n_it_lv * n_steps)
    boxplot_data["value"].extend(error_matrix[start_idx:end_idx, :].flatten())

    # Agregar datos de control
    boxplot_data["spasticity_level"].extend([lv] * n_it_lv * n_steps)
    boxplot_data["metric"].extend(["Control"] * n_it_lv * n_steps)
    boxplot_data["value"].extend(control_matrix[start_idx:end_idx, :].flatten())

    # Agregar datos de interacción
    boxplot_data["spasticity_level"].extend([lv] * n_it_lv * n_steps)
    boxplot_data["metric"].extend(["Interaction"] * n_it_lv * n_steps)
    boxplot_data["value"].extend(inter_matrix[start_idx:end_idx, :].flatten())

       # Agregar datos de interacción
    boxplot_data["spasticity_level"].extend([lv] * n_it_lv * n_steps)
    boxplot_data["metric"].extend(["Interaction"] * n_it_lv * n_steps)
    boxplot_data["value"].extend(pid_inter_matrix[start_idx:end_idx, :].flatten())

# Convertir a DataFrame y guardar como CSV
import pandas as pd

boxplot_df = pd.DataFrame(boxplot_data)
boxplot_df.to_csv(os.path.join(output_dir, "boxplot_data.csv"), index=False)

print(f"Datos guardados en la carpeta {output_dir}")
