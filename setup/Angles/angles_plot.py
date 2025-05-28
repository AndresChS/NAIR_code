from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

# Conecta a tu servidor de MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Ajusta la URL según sea necesario
db = client['0_env_data']  # Ajusta el nombre de la base de datos
collection = db['experiments_results']  # Ajusta el nombre de la colección

# Define los filtros para la consulta
id_filter = 'knee_angles_raw'  # Ajusta el valor del ID

# Realiza la consulta para obtener los documentos filtrados por id
query = {'id_experiment': id_filter}
documents = collection.find(query, {'_id': 0, 'time_step': 1, 'time': 1, 'qpos': 1})  # Excluye _id y selecciona las variables necesarias

# Extrae los datos y crea la tabla
data = list(documents)  # Convierte el cursor a una lista

# Verifica si se han encontrado documentos
if not data:
    print("No se encontraron documentos con los filtros especificados.")
else:
    df = pd.DataFrame(data)

def test(value):
    print(type(value))
    return value
# Agrupar los datos por timestep y asegurar consistencia
grouped_df = df.groupby('time_step').agg(
    time = ('time', 'first'),  # Suponiendo que 'time' es constante por timestep
    exo_flexion = ('qpos', lambda x:  (360/(2*3.14))*x.iloc[0][2]),  # Accede al elemento en la posición 6 de qpos si existe
    knee_angle_r_t1 = ('qpos', lambda x:  x.iloc[0][4]),  # Accede al elemento en la posición 6 de qpos si existe
    knee_angle_r_t2 = ('qpos', lambda x:  x.iloc[0][5]),  # Accede al elemento en la posición 6 de qpos si existe
    knee_angle_r = ('qpos', lambda x:  (-360/(2*3.14))*x.iloc[0][6]),  # Accede al elemento en la posición 6 de qpos si existe
    knee_angle_r_r2 = ('qpos', lambda x:  x.iloc[0][7]),  # Accede al elemento en la posición 6 de qpos si existe
    knee_angle_r_r3 = ('qpos', lambda x:  x.iloc[0][8])  # Accede al elemento en la posición 6 de qpos si existe
).reset_index()

print(grouped_df)

# Crear una figura y ejes

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)



# Asegurarse de que 'time' sea una lista con la misma longitud que los otros datos
time = grouped_df['time'].tolist()
exo_flexion = grouped_df['exo_flexion'].tolist()
knee_angle_r_t1 = grouped_df['knee_angle_r_t1'].tolist()
knee_angle_r_t2 = grouped_df['knee_angle_r_t2'].tolist()
knee_angle_r = grouped_df['knee_angle_r'].tolist()
knee_angle_r_r2 = grouped_df['knee_angle_r_r2'].tolist()
knee_angle_r_r3 = grouped_df['knee_angle_r_r3'].tolist()

# Graficar qpos en la posición [0]
axs[0].plot(time, exo_flexion, label='exo_flexion')
axs[0].plot(time, knee_angle_r, label='knee_flexion')
# Configurar la gráfica
axs[0].set_title('qpos vs. time')
axs[0].set_xlabel('time')
axs[0].set_ylabel('angle (rad)')
axs[0].legend()

# Graficar qpos en la posición [0]
axs[1].plot(time, knee_angle_r_t1, label='knee_t1')
axs[1].plot(time, knee_angle_r_t2, label='knee_t2')
axs[1].plot(time, knee_angle_r_r2, label='knee_r2')
axs[1].plot(time, knee_angle_r_r3, label='knee_r3')
# Configurar la gráfica
axs[1].set_title('Knee angles vs. time')
axs[1].set_xlabel('time')
axs[1].set_ylabel('ctrl_power')
axs[1].legend()

# Mostrar la gráfica
plt.tight_layout()
plt.show()

