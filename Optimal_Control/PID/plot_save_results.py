import numpy as np
import matplotlib.pyplot as plt
import json
from pymongo import MongoClient
import pandas as pd


# Conecta a tu servidor de MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Ajusta la URL según sea necesario
db = client['0_env_data']  # Ajusta el nombre de la base de datos
collection = db['torque']  # Ajusta el nombre de la colección

# Define los filtros para la consulta
id_filter = 'angles_torque_61'  # Ajusta el valor del ID

# Realiza la consulta para obtener los documentos filtrados por id
query = {'id_experiment': id_filter}
documents = collection.find(query, {'_id': 0, 'time': 1, 'error':1, 'step': 1, 'qpos': 1, 'qvel':1, 'qacc':1, 'inter_force': 1, 'qfrc_actuator': 1, 'qfrc_constraint': 1, 'ctrl':1, 'ten_length':1, 'reward':1})  # Excluye _id y selecciona las variables necesarias

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
grouped_df = df.groupby('step').agg(
    time = ('time', 'first'),  # Suponiendo que 'time' es constante por timestep
    reward = ('reward', 'first'),  # Suponiendo que 'time' es constante por timestep
    knee_force = ('qfrc_actuator', lambda x:  x.iloc[0][6]),  # Accede al elemento en la posición 6 de qpos si existe
    knee_constraint = ('qfrc_constraint', lambda x:  x.iloc[0][6]),  # Accede al elemento en la posición 6 de qpos si existe
    inter_force = ('inter_force', 'first'),
    exo_control = ('ctrl', lambda x:  x.iloc[0][0]),  # Accede al elemento en la posición 0 de ctrl si existe
    semimem_spas = ('ctrl', lambda x:  x.iloc[0][32]),  # Accede al elemento en la posición 32 de ctrl si existe
    knee_angle = ('qpos', lambda x:  (-360/(2*3.14))*x.iloc[0][6]),  # Accede al elemento en la posición 6 de qpos si existe
    pos_error = ('error',  lambda x:  (-360/(2*3.14))*x),  # Accede al elemento en la posición 6 de qpos si existe
    knee_vel = ('qvel', lambda x:  x.iloc[0][6]),  # Accede al elemento en la posición 6 de qvel si existe
    knee_acc = ('qacc', lambda x:  x.iloc[0][6]),
    ten_length_semimem = ('ten_length', lambda x:  x.iloc[0][31]),
    ten_length_semiten = ('ten_length', lambda x:  x.iloc[0][32]),
    ten_length_bfsl = ('ten_length', lambda x:  x.iloc[0][6]),
    ten_length_bfsh = ('ten_length', lambda x:  x.iloc[0][7]),
    ten_length_gaslat = ('ten_length', lambda x:  x.iloc[0][13]),
    ten_length_gasmed = ('ten_length', lambda x:  x.iloc[0][12]),
    ten_length_gracc = ('ten_length', lambda x:  x.iloc[0][23]),
    ten_length_vasmed = ('ten_length', lambda x:  x.iloc[0][39]),
    ten_length_vaslat = ('ten_length', lambda x:  x.iloc[0][38]),
    ten_length_vasint = ('ten_length', lambda x:  x.iloc[0][37]),
    ten_length_recfem = ('ten_length', lambda x:  x.iloc[0][29])
).reset_index()

#print(grouped_df)

# Crear una figura y ejes
def main():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    # Asegurarse de que 'time' sea una lista con la misma longitud que los otros datos
    time = grouped_df['time'].tolist()
    reward = grouped_df['reward'].tolist()
    exo_control = grouped_df['exo_control'].tolist()
    semimem_spas = grouped_df['semimem_spas'].tolist()
    knee_vel = grouped_df['knee_vel'].tolist()
    knee_acc = grouped_df['knee_acc'].tolist()
    knee_angle = grouped_df['knee_angle'].tolist()
    pos_error = grouped_df['pos_error'].tolist()
    print(type(pos_error), len(pos_error), pos_error[-1]*(2*np.pi)/360)
    knee_force = grouped_df['knee_force'].tolist()
    knee_constraint = grouped_df['knee_constraint'].tolist()
    inter_force = grouped_df['inter_force'].tolist()
    ten_length_semimem = grouped_df['ten_length_semimem'].tolist()
    ten_length_semiten = grouped_df['ten_length_semiten'].tolist()
    ten_length_bfsh = grouped_df['ten_length_bfsh'].tolist()
    ten_length_bfsl = grouped_df['ten_length_bfsl'].tolist()
    ten_length_gaslat = grouped_df['ten_length_gaslat'].tolist()
    ten_length_gasmed = grouped_df['ten_length_gasmed'].tolist()
    ten_length_gracc = grouped_df['ten_length_gracc'].tolist()
    ten_length_vasmed = grouped_df['ten_length_vasmed'].tolist()
    ten_length_vaslat = grouped_df['ten_length_vaslat'].tolist()
    ten_length_vasint = grouped_df['ten_length_vasint'].tolist()
    ten_length_recfem = grouped_df['ten_length_recfem'].tolist()

    # Graficar qpos en la posición [0]
    #axs[0][0].plot(time[0:150], knee_angle[0:150], label='knee_flexion')
    axs[0][0].plot(time[0:150], pos_error[0:150], label='error')
    # Configurar la gráfica
    axs[0][0].set_title('qpos vs. time')
    axs[0][0].set_xlabel('time (s)')
    axs[0][0].set_ylabel('angle (º)')
    axs[0][0].legend()
    
    # Graficar qpos en la posición [0]
    axs[1][0].plot(time[0:150], exo_control[0:150], label='exo_ctrl')
    axs[1][0].plot(time[0:150], semimem_spas[0:150], label='spasticity')
    axs[1][0].plot(time[0:150], reward[0:150], label='dense_reward')
    # Configurar la gráfica
    axs[1][0].set_title('exo_control/Spasticity vs. time')
    #axs[1][0].set_ylim([-1, 1])
    axs[1][0].set_xlabel('time')
    axs[1][0].set_ylabel('Ctrl/Spasticity')
    axs[1][0].legend()

    # Graficar qpos en la posición [0]
    #print(inter_force)
    #axs[1][1].plot(time[0:150], knee_force[0:150], label='knee_force')
    #axs[1][1].plot(time[0:150], knee_constraint[0:150], label='knee_constraint')
    axs[1][1].plot(time[0:150], inter_force[0:150], label='inter_force')
    # Configurar la gráfica
    axs[1][1].set_title('knee/joint_torque vs. time')
    axs[1][1].set_xlabel('time')
    axs[1][1].set_ylabel('torque')
    axs[1][1].legend()

    # Graficar qpos en la posición [0]
    axs[0][1].plot(time[0:150], knee_vel[0:150], label='knee_vel')
    #axs[0][1].plot(time[0:150], knee_acc[0:150], label='knee_acc')
    # Configurar la gráfica
    axs[0][1].set_title('knee_vel vs. time')
    axs[0][1].set_xlabel('time')
    axs[0][1].set_ylabel('vel')
    axs[0][1].legend()
    """
    # Graficar ten_length en la posición [0]
    axs[3].plot(time[0:75], ten_length_semimem[0:75], label='semimem')
    axs[3].plot(time[0:75], ten_length_semiten[0:75], label='semiten')
    axs[3].plot(time[0:75], ten_length_bfsh[0:75], label='bfsh')
    axs[3].plot(time[0:75], ten_length_bfsl[0:75], label='bfsl')
    axs[3].plot(time[0:75], ten_length_gaslat[0:75], label='gaslat')
    axs[3].plot(time[0:75], ten_length_gasmed[0:75], label='gasmed')
    axs[3].plot(time[0:75], ten_length_gracc[0:75], label='gracc')
    axs[3].plot(time[0:75], ten_length_vasint[0:75], label='vasint')
    axs[3].plot(time[0:75], ten_length_vasmed[0:75], label='vasmed')
    axs[3].plot(time[0:75], ten_length_vaslat[0:75], label='vaslat')
    axs[3].plot(time[0:75], ten_length_recfem[0:75], label='recfem')
    # Configurar la gráfica
    axs[3].set_title('ten_length vs. time')
    axs[3].set_xlabel('time (s)')
    axs[3].set_ylabel('length (m)')
    axs[3].legend()
    """
    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()

    """
    # Read data or files
    filepath = "/Users/achs/Documents/PHD/code/NAIR_Code/code/RL/SB3/results/SB3_PPO_1_000_000_q_passive"
    with open(filepath, 'r') as file:
        # Lee el contenido completo del archivo y elimina los caracteres '[' y ']'
        data = file.read().strip().replace('[', '').replace(']', '')

    with open('/Users/achs/Documents/PHD/code/NAIR_Code/code/RL/SB3/results/SB3_PPO_1_000_000_spas.json', 'r') as file:
        diccionario_cargado = json.load(file)
    print(diccionario_cargado)
    # Convertir los datos a una lista de números flotantes
    data = [float(i) for i in data.split(',')]

    # Crear el eje x con una diferencia de 0.01 segundos
    x = np.arange(0, len(data) * 0.01, 0.01)

    # Graficar los datos
    plt.plot(x, data)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Valores')
    plt.title('Gráfica de datos desde archivo')
    plt.grid(True)

    # Guardar la gráfica como imagen
    plt.savefig('grafica.png')

    # Mostrar la gráfica
    plt.show()
    """
if __name__ == "__main__":
    main()