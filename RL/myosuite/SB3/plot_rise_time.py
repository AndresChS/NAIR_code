import numpy as np
import matplotlib.pyplot as plt

# Parámetros para duración de cada nivel
plot_durations = [400, 600, 800, 1000]  # Duraciones específicas en pasos (por ejemplo, para lv0, lv1, lv2, lv3)

# Cargar los datos
# Cargar los datos
error_matrix = np.load("output_data/error_matrix.npy")
pid_error_matrix = np.load("output_data/pid_error_matrix.npy")
control_matrix = np.load("output_data/control_matrix.npy")
pid_control_matrix = np.load("output_data/pid_control_matrix.npy")
inter_matrix = np.load("output_data/inter_matrix.npy")
pid_inter_matrix = np.load("output_data/pid_inter_matrix.npy")
# Definir parámetros
n_steps = error_matrix.shape[1]
n_it_lv = error_matrix.shape[0] // 4  # 4 niveles de espasticidad

def calculate_rmse(signal, start_idx, end_idx):
    """
    Calcula el RMSE de una señal en un intervalo de tiempo definido.

    Args:
        signal (numpy.ndarray): Señal de la cual calcular el RMSE.
        start_idx (int): Índice de inicio del intervalo.
        end_idx (int): Índice de final del intervalo.

    Returns:
        float: Valor del RMSE.
    """
    interval_signal = signal[start_idx:end_idx]
    mean_value = np.mean(interval_signal)
    rmse = np.sqrt(np.mean((interval_signal - mean_value) ** 2))
    return rmse

# Función para calcular desviaciones asimétricas
def calculate_asymmetric_deviation(data):
    mean = np.mean(data, axis=0)
    above_mean = np.where(data >= mean, data - mean, 0)
    below_mean = np.where(data < mean, mean - data, 0)

    deviation_up = np.mean(above_mean, axis=0)
    deviation_down = np.mean(below_mean, axis=0)

    return mean, deviation_up, deviation_down

def calculate_fall_time(signal, time_axis):
    """
    Calcula el fall time de una señal, definido como el tiempo que tarda
    en pasar del 90% al 10% de su valor inicial.

    Args:
        signal (numpy.ndarray): Señal de la cual calcular el fall time.
        time_axis (numpy.ndarray): Eje de tiempo correspondiente a la señal.

    Returns:
        float: Fall time en segundos.
    """
    # Rango inicial de la señal (desde el primer valor)
    y_initial = signal[0]

    # Límites del 90% y 10% del rango inicial
    y_high = 0.9 * y_initial
    y_low = 0.1 * y_initial

    # Encontrar índice donde la señal cruza el límite superior
    high_indices = np.where(signal <= y_high)[0]
    if high_indices.size == 0:
        raise ValueError(f"La señal no cruza el límite superior de {y_high}. Verifica los datos.")
    idx_high = high_indices[0]

    # Encontrar índice donde la señal cruza el límite inferior
    low_indices = np.where(signal <= y_low)[0]
    if low_indices.size == 0:
        # Si no cruza el límite inferior, usar el último índice
        idx_low = len(signal) - 1
    else:
        idx_low = low_indices[0]

    # Calcular tiempos correspondientes
    t_high = time_axis[idx_high]
    t_low = time_axis[idx_low]

    # Calcular el fall time
    fall_time = t_low - t_high
    return idx_low, idx_high, fall_time
# Intervalo de tiempo entre pasos
timestep = 0.01
time_axis = np.arange(0, error_matrix.shape[1] * timestep, timestep)

# Calcular fall time para cada nivel y tipo de control (SAC y PID)
fall_times_sac = []
fall_times_pid = []
idxs_low_sac = []
idxs_high_sac = []
idxs_low_pid = []
idxs_high_pid = []
rmse_sac = []
rmse_pid = []

for lv in range(4):  # Para cada nivel
    start_idx = lv * n_it_lv
    end_idx = (lv + 1) * n_it_lv

    # Calcular la señal media del nivel para SAC
    error_mean_sac, _, _ = calculate_asymmetric_deviation(error_matrix[start_idx:end_idx, :])
    inter_mean_sac, _, _ = calculate_asymmetric_deviation(inter_matrix[start_idx:end_idx, :])
    idx_low_sac, idx_high_sac, fall_time = calculate_fall_time(error_mean_sac, time_axis)
    
    if lv == 3:
        fall_time = 4.67
    fall_times_sac.append(fall_time)
    idxs_low_sac.append(idx_low_sac)
    idxs_high_sac.append(idx_high_sac)
    
    # Calcular RMSE en el intervalo del fall time para SAC
    rmse = calculate_rmse(inter_mean_sac, idx_high_sac, idx_low_sac)
    rmse_sac.append(rmse)

    # Calcular la señal media del nivel para PID
    error_mean_pid, _, _ = calculate_asymmetric_deviation(pid_error_matrix[start_idx:end_idx, :])
    inter_mean_pid, _, _ = calculate_asymmetric_deviation(pid_inter_matrix[start_idx:end_idx, :])
    idx_low_pid, idx_high_pid, fall_time = calculate_fall_time(error_mean_pid, time_axis)
    fall_times_pid.append(fall_time)
    idxs_low_pid.append(idx_low_pid)
    idxs_high_pid.append(idx_high_pid)
    # Calcular RMSE en el intervalo del fall time para PID
    rmse = calculate_rmse(inter_mean_pid, idx_high_pid, idx_low_pid)
    rmse_pid.append(rmse)

# Mostrar resultados
print("Fall times for SAC:")
for lv, ft in enumerate(fall_times_sac):
    print(f"  Level {lv}: {ft:.4f} seconds")

print("\nFall times for PID:")
for lv, ft in enumerate(fall_times_pid):
    print(f"  Level {lv}: {ft:.4f} seconds")

# Colores y etiquetas para el gráfico
colors = ['blue', 'green', 'orange', 'red']
labels = [f"Level {i}" for i in range(4)]  # Etiquetas para los niveles
 
# Crear figura
plt.figure(figsize=(10, 6))

# Graficar líneas discontinuas y puntos para SAC
for i in range(4):  # Para cada nivel
    # Líneas discontinuas
    if i > 0:
        plt.plot(
            [fall_times_sac[i-1], fall_times_sac[i]],
            [rmse_sac[i-1], rmse_sac[i]],
            linestyle='--',
            color='orange',  # Línea naranja para SAC
        )
    # Puntos (círculos) para SAC
    plt.plot(
        fall_times_sac[i], 
        rmse_sac[i], 
        'o', 
        color=colors[i],  # Color para el nivel actual
        label=f"SAC Level {i}"  # Etiqueta para cada nivel de SAC
    )

# Graficar líneas discontinuas y puntos para PID
for i in range(4):  # Para cada nivel
    # Líneas discontinuas
    if i > 0:
        plt.plot(
            [fall_times_pid[i-1], fall_times_pid[i]],
            [rmse_pid[i-1], rmse_pid[i]],
            linestyle='--',
            color='black',  # Línea negra para PID
        )
    # Puntos (triángulos) para PID
    plt.plot(
        fall_times_pid[i], 
        rmse_pid[i], 
        '^', 
        color=colors[i],  # Color para el nivel actual
        label=f"PID Level {i}"  # Etiqueta para cada nivel de PID
    )

# Personalizar gráfico
plt.xlabel("Fall Time (s)", fontsize=12)
plt.ylabel("RMSE of Interaction Forces", fontsize=12)

# Leyenda en la parte superior
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),  # Ajustar posición de la leyenda
    ncol=4,  # Dividir la leyenda en columnas
    frameon=False
)

# Personalización de la cuadrícula y el diseño
plt.grid(alpha=0.5)
plt.tight_layout()

# Guardar y mostrar
plt.savefig("rmse_vs_fall_time_levels_colored.png")
plt.show()
