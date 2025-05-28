 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configurar fuentes sin usar LaTeX
rcParams['text.usetex'] = False
rcParams['font.family'] = 'serif'  # Cambiar a una fuente compatible como Times New Roman
rcParams['pdf.fonttype'] = 42  # Usar fuentes TrueType en PDFs

# Cargar los datos
error_matrix = np.load("output_data/error_matrix.npy")

control_matrix = np.load("output_data/control_matrix.npy")

inter_matrix = np.load("output_data/inter_matrix.npy")

# Definir parámetros
n_steps = error_matrix.shape[1]
n_it_lv = error_matrix.shape[0] // 4  # 4 niveles de espasticidad

timestep = 0.01  # Intervalo de tiempo por paso (modifica según tu simulación)

# Cambiar el rango de pasos a segundos reales
time_axis = np.arange(0, n_steps * timestep, timestep)

# Colores para cada nivel de espasticidad
colors = ['g', '#FFDD44', 'orange', '#FF4500']
labels = [f"Level {i}" for i in range(4)]  # Etiquetas para los niveles
time_plot =800
# Definir parámetros
n_steps = error_matrix.shape[1]
n_it_lv = error_matrix.shape[0] // 4  # 4 niveles de espasticidad

# Función para calcular desviaciones asimétricas
def calculate_asymmetric_deviation(data):
    mean = np.mean(data, axis=0)
    above_mean = np.where(data >= mean, data - mean, 0)
    below_mean = np.where(data < mean, mean - data, 0)

    deviation_up = np.mean(above_mean, axis=0)
    deviation_down = np.mean(below_mean, axis=0)

    return mean, deviation_up, deviation_down

# Crear figuras
fig, axs = plt.subplots(3, 1, figsize=(12, 8))

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
    inter_mean, inter_dev_up, inter_dev_down = calculate_asymmetric_deviation(
        inter_matrix[start_idx:end_idx, :]
    )

    # Graficar Error
    axs[0].plot(time_axis[0:time_plot], error_mean[0:time_plot], label=f'Nivel {lv}', color=colors[lv])
    axs[0].fill_between(
        time_axis[0:time_plot],
        error_mean[0:time_plot] - error_dev_down[0:time_plot],
        error_mean[0:time_plot] + error_dev_up[0:time_plot],
        color=colors[lv],
        alpha=0.3,
        label='_nolegend_',
    )

    # Graficar Control
    axs[1].plot(time_axis[0:time_plot], control_mean[0:time_plot], label=f'Nivel {lv}', color=colors[lv])
    axs[1].fill_between(
        time_axis[0:time_plot],
        control_mean[0:time_plot] - control_dev_down[0:time_plot],
        control_mean[0:time_plot] + control_dev_up[0:time_plot],
        color=colors[lv],
        alpha=0.3,
        label='_nolegend_',
    )
    # Graficar Interacción
    axs[2].plot(time_axis[0:time_plot], inter_mean[0:time_plot], label=f'Nivel {lv}', color=colors[lv])
    axs[2].fill_between(
        time_axis[0:time_plot],
        inter_mean[0:time_plot] - inter_dev_down[0:time_plot],
        inter_mean[0:time_plot] + inter_dev_up[0:time_plot],
        color=colors[lv],
        alpha=0.3,
        label='_nolegend_',
    )



axs[0].set_ylabel("Pos Error (rad)", fontsize = 18)
axs[1].set_ylabel("Control input", fontsize = 18)
#axs[2].set_ylabel("Velocity (rads/s)")
axs[2].set_ylabel("Interaction (Nm)", fontsize = 18)
axs[2].set_xlabel("Time (s)", fontsize = 22)
# Colocar la leyenda global en la parte superior
fig.legend(
    labels,  # Etiquetas de la leyenda
    loc='upper center',  # Posición en la parte superior central
    ncol=4,  # Número de columnas
    frameon=False,  # Sin borde
    fontsize = 24,
    bbox_to_anchor=(0.53, 1),  # Ajustar posición vertical de la leyenda
)

plt.tight_layout(rect=[0, 0.01, 1, 0.92])  # Ajustar el espacio para que la leyenda no sobreponga las gráficas
plt.savefig("spasticity_levels_plot_asymmetric.png")
plt.show()