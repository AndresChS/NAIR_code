#=================================================
#	This model is generated with tacking the Myosuite conversion of [Rajagopal's full body gait model](https://github.com/opensim-org/opensim-models/tree/master/Models/RajagopalModel) as close
#reference.
#	Model	  :: Myo Leg 1 Dof 40 Musc Exo (MuJoCoV2.0)
#	Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
#	source	:: https://github.com/AndresChS/NAIR_Code
#	====================================================== -->
import numpy as np
from myosuite.utils import gym
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import time, subprocess
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


def run_simulation(kp, ki, kd, setpoint, num_steps):
    
    env = gym.make('ExoLegSpasticityFlexoExtEnv-v0')
    env.set_spasticity_level(spasticity_level=1)
    data = env.sim.data
    dt = env.sim.model.opt.timestep    
    n_it_lv = 10
    n_steps = num_steps
    reward_matrix = np.zeros((4*n_it_lv, n_steps))
    error_matrix = np.zeros((4*n_it_lv, n_steps))
    control_matrix = np.zeros((4*n_it_lv, n_steps))
    inter_matrix = np.zeros((4*n_it_lv, n_steps))
    vel_matrix = np.zeros((4*n_it_lv, n_steps))
    pid_controller = PIDController(kp, ki, kd, setpoint)


    # Colores para cada nivel de espasticidad
    colors = ['blue', 'orange', 'green', 'red']
    labels = [f"Level {i}" for i in range(4)]  # Etiquetas para los niveles

    # Bucle para obtener datos para cada nivel de espasticidad
    for lv in range(4):  # Suponiendo 4 niveles de espasticidad
        for i in tqdm(range(n_it_lv), desc=f"Nivel de espasticidad {lv}"):  # n_steps simulaciones por nivel de espasticidad
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
                
                error_matrix[i + lv * n_it_lv, j] = -PID_error
                vel_matrix[i + lv * n_it_lv, j] = data.qvel[6]
                control_matrix[i + lv * n_it_lv, j] = actions[0]
                inter_matrix[i + lv * n_it_lv, j] = data.qfrc_actuator[6]

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


    # Guardar matrices de resultados
    output_dir = "output_data_pd1"
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "pid_error_matrix.npy"), error_matrix)
    np.save(os.path.join(output_dir, "pid_control_matrix.npy"), control_matrix)
    np.save(os.path.join(output_dir, "pid_inter_matrix.npy"), inter_matrix)

    print(f"Datos guardados en la carpeta {output_dir}")

# Initial guess for PID parameters
num_steps = 1536 # Número de pasos de simulación
kp = 12
ki = 0
kd = 0.05
setpoint = 0.2
params_list = [ki, kp, kd]
optimized = 1
# Perform the optimization
run_simulation(kp, ki, kd, setpoint, num_steps)
#subprocess.run(["python3", "./plot_save_results.py"])
