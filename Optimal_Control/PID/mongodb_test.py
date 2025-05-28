from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

# Conecta a tu servidor de MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Ajusta la URL según sea necesario
db = client['test']  # Ajusta el nombre de la base de datos
collection = db['experiments_results']  # Ajusta el nombre de la colección

# Define los filtros para la consulta
id_filter = 'PID_experiment_66'  # Ajusta el valor del ID

# Realiza la consulta para obtener los documentos filtrados por id
query = {'id_experiment': id_filter, 'optimized': 1}
documents = collection.find(query, {'_id': 0, 'time_step': 1, 'time': 1, 'qpos': 1, 'error': 1, 'setpoint': 1, 'ctrl': 1})  # Excluye _id y selecciona las variables necesarias

# Extrae los datos y crea la tabla
data = list(documents)  # Convierte el cursor a una lista

# Verifica si se han encontrado documentos
if not data:
    print("No se encontraron documentos con los filtros especificados.")
else:
    df = pd.DataFrame(data)
    print(df)

def test(value):
    print(type(value))
    return value

# Agrupar los datos por timestep y asegurar consistencia
grouped_df = df.groupby('time_step').agg({
    'time': 'first',  # Suponiendo que 'time' es constante por timestep
    'qpos': lambda x:  x.iloc[0][6],  # Accede al elemento en la posición 6 de qpos si existe
    'ctrl': lambda x:  x.iloc[0][0],
    'error': lambda x:  x.iloc[0],
    'setpoint': lambda x:  x.iloc[0]
}).reset_index()

print(grouped_df)

# Crear una figura y ejes

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)



# Asegurarse de que 'time' sea una lista con la misma longitud que los otros datos
time = grouped_df['time'].tolist()

# Graficar qpos en la posición [0]
qpos_values = grouped_df['qpos'].tolist()
error_values = grouped_df['error'].tolist()
setpoint_values = grouped_df['setpoint'].tolist()
axs[0].plot(time, qpos_values, label='joint pos')
axs[0].plot(time, error_values, label='error')
axs[0].plot(time, setpoint_values, label='setpoint')
# Configurar la gráfica
axs[0].set_title('qpos[0] vs. time')
axs[0].set_xlabel('time')
axs[0].set_ylabel('angle (rad)')
axs[0].legend()

# Graficar qpos en la posición [0]
ctrl_values = grouped_df['ctrl'].tolist()
axs[1].plot(time, ctrl_values, label='qpos[0]')
# Configurar la gráfica
axs[1].set_title('ctrl[0] vs. time')
axs[1].set_xlabel('time')
axs[1].set_ylabel('ctrl_power')
axs[1].legend()

# Mostrar la gráfica
plt.tight_layout()
plt.show()

