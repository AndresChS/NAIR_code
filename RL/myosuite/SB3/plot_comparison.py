import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configurar fuentes sin usar LaTeX
rcParams['text.usetex'] = False
rcParams['font.family'] = 'serif'  # Cambiar a una fuente compatible como Times New Roman
rcParams['pdf.fonttype'] = 42  # Usar fuentes TrueType en PDFs

# Parámetros para duración de cada nivel
plot_durations = [400, 600, 800, 1000]  # Duraciones específicas en pasos (por ejemplo, para lv0, lv1, lv2, lv3)

# Cargar los datos
error_matrix = np.load("output_data/error_matrix.npy")
pid_error_matrix = np.load("output_data/pid_error_matrix.npy")
control_matrix = np.load("output_data/control_matrix.npy")
pid_control_matrix = np.load("output_data/pid_control_matrix.npy")
inter_matrix = np.load("output_data/inter_matrix.npy")
pid_inter_matrix = np.load("output_data/pid_inter_matrix.npy")

# Calcular límites globales para las gráficas
error_min = min(np.min(error_matrix), np.min(pid_error_matrix))
error_max = max(np.max(error_matrix), np.max(pid_error_matrix))

control_min = -1.1
control_max = 1.1

inter_min = -100
inter_max = max(np.max(inter_matrix), np.max(pid_inter_matrix))

colors = ['orange', 'black']  # Colores personalizados
timestep = 0.01  # Intervalo de tiempo por paso (en segundos)

# Definir parámetros
n_steps = error_matrix.shape[1]
n_it_lv = error_matrix.shape[0] // 4  # 4 niveles de espasticidad
time_axis = np.arange(0, n_steps * timestep, timestep)  # Eje temporal en segundos

# Función para calcular desviaciones asimétricas
def calculate_asymmetric_deviation(data):
    mean = np.mean(data, axis=0)
    above_mean = np.where(data >= mean, data - mean, 0)
    below_mean = np.where(data < mean, mean - data, 0)

    deviation_up = np.mean(above_mean, axis=0)
    deviation_down = np.mean(below_mean, axis=0)

    return mean, deviation_up, deviation_down

# Crear figura de 3x4 para las gráficas
fig, axs = plt.subplots(3, 4, figsize=(20, 12))

# Bucle para graficar cada nivel de espasticidad en una columna
for lv in range(4):
    start_idx = lv * n_it_lv
    end_idx = (lv + 1) * n_it_lv
    plot_duration = plot_durations[lv]  # Duración específica para este nivel
    time_limit = plot_duration * timestep  # Convertir pasos a segundos

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
    axs[0, lv].plot(time_axis[:plot_duration], pid_error_mean[:plot_duration], label='PID', linewidth=2, color=colors[1])
    axs[0, lv].fill_between(
        time_axis[:plot_duration],
        pid_error_mean[:plot_duration] - pid_error_dev_down[:plot_duration],
        pid_error_mean[:plot_duration] + pid_error_dev_up[:plot_duration],
        color=colors[1],
        alpha=0.3,
    )
    axs[0, lv].plot(time_axis[:plot_duration], error_mean[:plot_duration], label='SAC', linewidth=2, color=colors[0])
    axs[0, lv].fill_between(
        time_axis[:plot_duration],
        error_mean[:plot_duration] - error_dev_down[:plot_duration],
        error_mean[:plot_duration] + error_dev_up[:plot_duration],
        color=colors[0],
        alpha=0.3,
    )
    axs[0, lv].set_ylim(error_min, error_max)
    if lv == 0:
        axs[0, lv].set_ylabel("Position Error (rads)", fontsize=16)
    else:
        axs[0, lv].tick_params(labelleft=False)  # Ocultar etiquetas del eje y en otras columnas

    axs[0, lv].set_title(f'Level {lv} (0-{time_limit:.1f}s)')

    # Graficar Control (fila 2)
    axs[1, lv].plot(time_axis[:plot_duration], pid_control_mean[:plot_duration], label='PID', linewidth=2, color=colors[1])
    axs[1, lv].fill_between(
        time_axis[:plot_duration],
        pid_control_mean[:plot_duration] - pid_control_dev_down[:plot_duration],
        pid_control_mean[:plot_duration] + pid_control_dev_up[:plot_duration],
        color=colors[1],
        alpha=0.3,
    )
    axs[1, lv].plot(time_axis[:plot_duration], control_mean[:plot_duration], label='SAC', linewidth=2, color=colors[0])
    axs[1, lv].fill_between(
        time_axis[:plot_duration],
        control_mean[:plot_duration] - control_dev_down[:plot_duration],
        control_mean[:plot_duration] + control_dev_up[:plot_duration],
        color=colors[0],
        alpha=0.3,
    )
    axs[1, lv].set_ylim(control_min, control_max)
    if lv == 0:
        axs[1, lv].set_ylabel("Control", fontsize=16)
    else:
        axs[1, lv].tick_params(labelleft=False)  # Ocultar etiquetas del eje y en otras columnas

    # Graficar Interacción (fila 3)
    axs[2, lv].plot(time_axis[:plot_duration], pid_inter_mean[:plot_duration], label='PID', linewidth=2, color=colors[1])
    axs[2, lv].fill_between(
        time_axis[:plot_duration],
        pid_inter_mean[:plot_duration] - pid_inter_dev_down[:plot_duration],
        pid_inter_mean[:plot_duration] + pid_inter_dev_up[:plot_duration],
        color=colors[1],
        alpha=0.3,
    )
    axs[2, lv].plot(time_axis[:plot_duration], inter_mean[:plot_duration], label='SAC', linewidth=2, color=colors[0])
    axs[2, lv].fill_between(
        time_axis[:plot_duration],
        inter_mean[:plot_duration] - inter_dev_down[:plot_duration],
        inter_mean[:plot_duration] + inter_dev_up[:plot_duration],
        color=colors[0],
        alpha=0.3,
    )
    axs[2, lv].set_ylim(inter_min, inter_max)
    if lv == 0:
        axs[2, lv].set_ylabel("Interaction (N/m)", fontsize=16)
    else:
        axs[2, lv].tick_params(labelleft=False)  # Ocultar etiquetas del eje y en otras columnas
        # Eliminar etiquetas del eje x en todas las subgráficas
    
    axs[0, lv].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[1, lv].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[2, lv].set_xlabel("Time (s)", fontsize=16)

# Colocar la leyenda global en la parte superior
fig.legend(
    ["Mean PID", "STD PID", "Mean SAC", "STD SAC"],  # Etiquetas de la leyenda
    loc='upper center',
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1),
    fontsize= 20
)

plt.tight_layout(rect=[0, 0.01, 1, 0.93])
plt.savefig("comparison_plot_pid_vs_rl_duration_specific.png")
plt.show()
