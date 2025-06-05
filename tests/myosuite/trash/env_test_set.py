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
from scipy.optimize import minimize
from pymongo import MongoClient
import pandas as pd
import subprocess


# Conecta a tu servidor de MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Ajusta la URL según sea necesario
db = client['0_env_data']  # Ajusta el nombre de la base de datos
collection = db['experiments_results']  # Ajusta el nombre de la colección

# Define los filtros para la consulta
id_filter = 'knee_angles_5'  # Ajusta el valor del ID

# Realiza la consulta para obtener los documentos filtrados por id
query = {'id_experiment': id_filter}
documents = collection.find(query, {'_id': 0, 'time_step': 1, 'time': 1, 'qpos': 1, 'qvel': 1, 'act': 1, 'xpos': 1, 'xquat': 1})  # Excluye _id y selecciona las variables necesarias

# Extrae los datos y crea la tabla
data = list(documents)  # Convierte el cursor a una lista

# Verifica si se han encontrado documentos
if not data:
    print("No se encontraron documentos con los filtros especificados.")
else:
    df = pd.DataFrame(data)

def test(value):
    print(value)
    print(type(value))
    return value

# Asegurarse de que qpos, qvel y act sean numpy arrays
df['qpos'] = df['qpos'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
df['qvel'] = df['qvel'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
df['act'] = df['act'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
df['xpos'] = df['xpos'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
df['xquat'] = df['xquat'].apply(lambda x: np.array(x) if isinstance(x, list) else x)

# Agrupar los datos por timestep y asegurar consistencia
grouped_df = df.groupby('time_step').agg(
    time = ('time', 'first'),  # Suponiendo que 'time' es constante por timestep
    knee_angle_r = ('qpos', lambda x:  (-360/(2*3.14))*x.iloc[0][6]),  # Accede al elemento en la posición 6 de qpos si existe
    qpos = ('qpos', lambda x:  x),  # Accede al elemento en la posición 6 de qpos si existe
    qvel = ('qvel', lambda x:  x),  # Accede al elemento en la posición 6 de qpos si existe
    act = ('act', lambda x:  x),  # Accede al elemento en la posición 6 de qpos si existe
    xpos = ('xpos', lambda x:  x),
    xquat = ('xquat', lambda x:  x)
).reset_index()

# Crear el diccionario de estado
exo_flexion = grouped_df['knee_angle_r'].tolist()
time = grouped_df['time'].tolist()
qp = grouped_df['qpos'].tolist()
test(qp[0][0])
qv = grouped_df['qvel'].tolist()
act = grouped_df['act'].tolist()
xpos = grouped_df['xpos'].tolist()
xquat = grouped_df['xquat'].tolist()
state_dict = {'exo_flexion': exo_flexion[0], 'time': time[0], 'qpos': qp[0][0], 'qvel': qv[0], 'act': act[0], 'body_pos': xpos[0], 'body_quat': xquat[0]}
test(state_dict['qpos'])

env = gym.make('ExoLeg40MuscFlexoExtEnv-v0')
obs = env.set_env_state(state_dict)

