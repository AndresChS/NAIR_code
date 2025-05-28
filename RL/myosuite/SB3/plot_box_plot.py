import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configurar fuentes sin usar LaTeX
rcParams['text.usetex'] = False
rcParams['font.family'] = 'serif'  # Cambiar a una fuente compatible como Times New Roman
rcParams['pdf.fonttype'] = 42  # Usar fuentes TrueType en PDFs


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

# Definir parámetros
n_steps = error_matrix.shape[1]
n_it_lv = error_matrix.shape[0] // 4  # 4 niveles de espasticidad

# =======================================================================
#           BOX PLOT 
# =======================================================================

# Calcular los máximos por nivel para SAC y PID
sac_inter_max_values = []
pid_inter_max_values = []

for lv in range(4):  # Suponiendo 4 niveles de espasticidad
    start_idx = lv * n_it_lv
    end_idx = (lv + 1) * n_it_lv
    
    # Máximos de interacción para SAC y PID en este nivel
    sac_max = np.max(inter_matrix[start_idx:end_idx, :], axis=1)
    pid_max = np.max(pid_inter_matrix[start_idx:end_idx, :], axis=1)
    
    sac_inter_max_values.append(sac_max)
    pid_inter_max_values.append(pid_max)

# Preparar datos para el box plot
data = []
colors = []  # Lista para asignar colores personalizados
labels = []

for lv in range(4):
    
    data.append(pid_inter_max_values[lv])
    data.append(sac_inter_max_values[lv])
    labels.append(f"PID Level {lv}")
    labels.append(f"SAC Level {lv}")
    colors.append("#A9A9A9")     # Color para PID
    colors.append("#FFC200")  # Color para SAC

# Crear el box plot con colores personalizados
plt.figure(figsize=(10, 6))
box = plt.boxplot(
    data, 
    patch_artist=True, 
    showmeans=True,
    meanprops=dict(marker='o', markerfacecolor='orange', markeredgecolor='black', markersize=4),
    medianprops=dict(color='orange', linewidth=1)
)

# Aplicar colores a las cajas
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Personalizar las etiquetas del eje X
plt.xticks(
    ticks=np.arange(1.5, len(data) + 1.5, 2),  # Centrar cada nivel
    labels=[f"Level {lv}" for lv in range(4)], 
    rotation=0, 
    fontsize=20
)

# Crear la leyenda superior
plt.legend(
    [box["boxes"][0], box["boxes"][1], box["means"][0], box["medians"][0]],
    ["PID", "SAC", "Mean", "Median"],
    loc='upper center',
    columnspacing=1.5,
    handletextpad=0.5,
    bbox_to_anchor=(0.5, 1.15),
    ncol=4,
    frameon=False,
    fontsize=24
)

# Ajustar etiquetas
plt.ylabel("Maximum Interaction Torque (N/m)", fontsize=20)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Guardar y mostrar
plt.tight_layout()
plt.savefig("box_plot_interaction_forces_all_legend.png")
plt.show()