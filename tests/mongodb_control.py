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
import time

client = MongoClient("mongodb://localhost:27017/")
db = client["test"]
collection = db['experiments_results']  # Ajusta el nombre de la colección

# Open and reset Environment
env = gym.make('ExoLeg40MuscFlexoExtEnv-v0')
env.reset()

# Define los filtros para la consulta
id_filter = 'PID_experiment_48'  # Ajusta el valor del ID

# Realiza la consulta para obtener los documentos filtrados por id
query = {'id_experiment': id_filter, 'optimized': 1}
documents = collection.find(query, {'_id': 0, 'time_step': 1, 'time': 1, 'qpos': 1, 'error': 1, 'ctrl': 1})  # Excluye _id y selecciona las variables necesarias

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
    'ctrl': lambda x:  x.iloc[0][0]
}).reset_index()

ctrl_values = grouped_df['ctrl'].tolist()
actions = np.zeros(41)
experiment = 'pid_results'

for _ in range(1000):
    actions[0]=ctrl_values[_]
    env.mj_render()
    env.step(actions) # take a random action
    time.sleep(0.02)
    data_dict = {

        'id_experiment': experiment,
        'time': data.time,
        'qpos': data.qpos.tolist(),
        'qvel': data.qvel.tolist(),
        'qacc': data.qacc.tolist(),
        'ctrl': data.ctrl.tolist(),
        
    }

    db.experiments_results.insert_one(
        data_dict)
env.close()