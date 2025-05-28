"""	
This model is generated with tacking the Myosuite conversion of [Rajagopal's full body gait model](https://github.com/opensim-org/opensim-models/tree/master/Models/RajagopalModel) as close
reference.
	Model	  :: MyoLeg 1 Dof Exo (MuJoCoV2.0)
	Author	:: Andres Chavarrias (andreschavarriassanchez@gmail.com), David Rodriguez, Pablo Lanillos 
	source	:: https://github.com/AndresChS/NRG_Code
"""

from pymongo import MongoClient
import pandas as pd
import numpy as np
from myosuite.utils import gym
import pickle
# Conecta a tu servidor de MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Ajusta la URL según sea necesario
db = client['0_env_data']  # Ajusta el nombre de la base de datos
collection = db['experiments_results']  # Ajusta el nombre de la colección
# Experiment
experiment = 'knee_angles_init'
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
    print(value)
    return value

# Agrupar los datos por timestep y asegurar consistencia
grouped_df = df.groupby('time_step').agg(
    time = ('time', 'first'),  # Suponiendo que 'time' es constante por timestep
    qpos = ('qpos', lambda x:  x.iloc[0]),
    exo_flexion = ('qpos', lambda x:  -(360/(2*3.14))*x.iloc[0][2]),  # Accede al elemento en la posición 6 de qpos si existe
    knee_angle_r = ('qpos', lambda x:  (360/(2*3.14))*x.iloc[0][6]),  # Accede al elemento en la posición 6 de qpos si existe
).reset_index()


# Asegurarse de que 'time' sea una lista con la misma longitud que los otros datos
time = grouped_df['time'].tolist()
qpos = grouped_df['qpos'].tolist()
exo_flexion = grouped_df['exo_flexion'].tolist()
knee_angle_r = grouped_df['knee_angle_r'].tolist()

exact_degrees = list(range(0, 91))  # Lista de grados exactos de 1 a 90
closest_values = []  # Lista para almacenar los valores más cercanos
closest_indices = []  # Lista para almacenar los índices de los valores más cercanos
angles_dict = {}
for degree in exact_degrees:
    closest_index = min(range(len(knee_angle_r)), key=lambda i: (abs(knee_angle_r[i] - degree), knee_angle_r[i]))
    closest_value = knee_angle_r[closest_index]
    #print(qpos[closest_index])
    data_dict = {
        'id_experiment': experiment,
        'knee_angle_r': knee_angle_r[closest_index],
        'qpos': qpos[closest_index],
        'flexion_degrees': closest_value
    }
    angles_dict[degree] = np.concatenate((np.array([closest_value]),qpos[closest_index]))
    #print(angles_dict)
    #db.experiments_results.insert_one(data_dict)

# Save angles dict with pickle
with open('angles_dict.pkl', 'wb') as file:
    pickle.dump(angles_dict, file)
# abrir el visualizador de mujoco:   python -m mujoco.viewer
# Open and reset Environment
#print(angles_dict)
#env = gym.make('ExoLeg40MuscFlexoExtEnv-v0')
#env.reset()

#obs = env.set_env_state(qpos[24])

# ***************************************************************
#                       Future works
# ***************************************************************
# Function with 0 to 90 degrees. (now 89-90 has the same joint init)


