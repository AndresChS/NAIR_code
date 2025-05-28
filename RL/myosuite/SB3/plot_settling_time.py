import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configurar fuentes sin usar LaTeX
rcParams['text.usetex'] = False
rcParams['font.family'] = 'serif'  # Cambiar a una fuente compatible como Times New Roman
rcParams['pdf.fonttype'] = 42  # Usar fuentes TrueType en PDFs

# Parámetros para duración de cada nivel
plot_durations = [1500, 1500, 1500, 1500]  # Duraciones específicas en pasos (por ejemplo, para lv0, lv1, lv2, lv3)

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

def calculate_settling_time(signal, time_axis, tolerance=0.05, window=100):
    """
    Calcula el settling time basado en un promedio móvil para determinar estabilidad.

    Args:
        signal (numpy.ndarray): Señal para calcular el settling time.
        time_axis (numpy.ndarray): Eje de tiempo correspondiente a la señal.
        tolerance (float): Umbral de tolerancia para la estabilidad (porcentaje del valor final).
        window (int): Tamaño de la ventana para el promedio móvil (en pasos de tiempo).

    Returns:
        int: Índice del settling time.
        float: Tiempo correspondiente.
        str: Estado del cálculo ('success' o 'failed').
    """
    final_value = np.mean(signal[-window:])  # Usar el promedio móvil como valor final aproximado
    lower_bound = final_value * (1 - tolerance)
    upper_bound = final_value * (1 + tolerance)

    # Calcular un promedio móvil para suavizar la señal
    moving_avg = np.convolve(signal, np.ones(window) / window, mode='valid')

    # Encontrar índice donde el promedio móvil entra dentro del rango de tolerancia
    within_bounds = np.where((moving_avg >= lower_bound) & (moving_avg <= upper_bound))[0]
    
    if within_bounds.size == 0:
        # Si no se encuentra estabilidad, devolver fallo
        return -1, -1, 'failed'
    
    # Índice del settling time
    settling_start_idx = within_bounds[0] + (window - 1)  # Ajustar por el tamaño de la ventana
    settling_time = time_axis[settling_start_idx]
    
    return settling_start_idx, settling_time, 'success'

def calculate_practical_settling_time(signal, time_axis, std_threshold=0.02, window=500):
    """
    Calcula un tiempo de estabilización práctico basado en la variabilidad de la señal.

    Args:
        signal (numpy.ndarray): Señal para calcular el settling time.
        time_axis (numpy.ndarray): Eje de tiempo correspondiente a la señal.
        std_threshold (float): Umbral de desviación estándar dentro de la ventana para considerar estabilidad.
        window (int): Tamaño de la ventana (en pasos de tiempo).

    Returns:
        int: Índice del tiempo de estabilización.
        float: Tiempo correspondiente.
        str: Estado del cálculo ('success' o 'failed').
    """
    # Iterar sobre la señal con una ventana deslizante
    for i in range(len(signal) - window):
        # Calcular desviación estándar dentro de la ventana
        window_std = np.std(signal[i:i + window])
        if window_std <= std_threshold:
            settling_time = time_axis[i]
            return i, settling_time, 'success'
    
    # Si no se encuentra estabilización, devolver fallo
    return -1, -1, 'failed'


def calculate_rmse(signal, start_idx, end_idx):

    interval_signal = signal[start_idx:end_idx]
    rms = np.sqrt(np.mean(interval_signal ** 2))
    return rms

# Función para calcular desviaciones asimétricas
def calculate_asymmetric_deviation(data):
    mean = np.mean(data, axis=0)
    above_mean = np.where(data >= mean, data - mean, 0)
    below_mean = np.where(data < mean, mean - data, 0)

    deviation_up = np.mean(above_mean, axis=0)
    deviation_down = np.mean(below_mean, axis=0)

    return mean, deviation_up, deviation_down

def calculate_fall_time(signal, time_axis):
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

settling_times_sac = []
settling_times_pid = []
rmse_sac = []
rmse_pid = []

for lv in range(4):  # Para cada nivel
    start_idx = lv * n_it_lv
    end_idx = (lv + 1) * n_it_lv

    # Señal promedio para el error (SAC)
    error_mean_sac, _, _ = calculate_asymmetric_deviation(error_matrix[start_idx:end_idx, :])
    inter_mean_sac, _, _ = calculate_asymmetric_deviation(inter_matrix[start_idx:end_idx, :])
    
    # Calcular el settling time para SAC
    settling_start_idx, settling_time, status = calculate_practical_settling_time(error_mean_sac, time_axis)
    if status == 'failed':
        print(f"Settling time no calculado para SAC nivel {lv}. Usando índice completo.")
        settling_start_idx = len(time_axis) - 1  # Usa todo el rango
        settling_time = time_axis[-1]
    settling_times_sac.append(settling_time)
    
    # Calcular RMSE hasta el settling time para SAC
    rmse = calculate_rmse(inter_mean_sac, 0, settling_start_idx)
    rmse_sac.append(rmse)

    # Señal promedio para el error (PID)
    error_mean_pid, _, _ = calculate_asymmetric_deviation(pid_error_matrix[start_idx:end_idx, :])
    inter_mean_pid, _, _ = calculate_asymmetric_deviation(pid_inter_matrix[start_idx:end_idx, :])
    
    # Calcular el tiempo de estabilización práctica para PID
    settling_start_idx, settling_time, status = calculate_practical_settling_time(error_mean_pid, time_axis)
    if status == 'failed':
        print(f"Settling time no calculado para PID nivel {lv}. Usando índice completo.")
        settling_start_idx = len(time_axis) - 1  # Usa todo el rango
        settling_time = time_axis[-1]
    settling_times_pid.append(settling_time)
    
    # Calcular RMSE hasta el tiempo de estabilización para PID
    rmse = calculate_rmse(inter_mean_pid, 0, settling_start_idx)
    rmse_pid.append(rmse)



print(settling_times_sac)
print(settling_times_pid)
print()
# Colores y etiquetas para el gráfico
colors = ['g', '#FFDD44', 'orange', '#FF4500']
labels = [f"Level {i}" for i in range(4)]  # Etiquetas para los niveles
plt.figure(figsize=(10, 6))

# Graficar líneas discontinuas y puntos para SAC
for i in range(4):  # Para cada nivel
    if i > 0:
        plt.plot(
            [settling_times_sac[i-1], settling_times_sac[i]],
            [rmse_sac[i-1], rmse_sac[i]],
            linestyle='--',
            linewidth=2,
            color='orange',
        )
    plt.plot(
        settling_times_sac[i], 
        rmse_sac[i], 
        'o', 
        color=colors[i],
         markersize=15,
        label=f"SAC Level {i}"
    )

# Graficar líneas discontinuas y puntos para PID
for i in range(4):  # Para cada nivel
    if i > 0:
        plt.plot(
            [settling_times_pid[i-1], settling_times_pid[i]],
            [rmse_pid[i-1], rmse_pid[i]],
            linestyle='--',
            linewidth=2,
            color='black',
        )
    plt.plot(
        settling_times_pid[i], 
        rmse_pid[i], 
        '^', 
        color=colors[i],
         markersize=15,
        label=f"PID Level {i}"
    )

plt.xlabel("Settling Time (s)", fontsize=20)
plt.ylabel("Interaction Torque (RMS)", fontsize=20)
plt.title(r'$\blacktriangle$ PID       $\bullet$ SAC', fontsize=24, color="black")

# Añadir manualmente el color naranja al círculo utilizando el símbolo de círculo sólido
plt.gcf().text(0.546, 0.91, r'$\bullet$', color="orange", fontsize=50)
#lt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, fontsize=14)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("rmse_vs_settling_time_levels_colored.png")
plt.show()
