import numpy as np
from myosuite.utils import gym


# Crear el entorno
# abrir el visualizador de mujoco:   python -m mujoco.viewer
#env = gym.make('probConnection-v0')
#env = gym.make('motorFingerReachRandom-v0')
env = gym.make('ExoLeg40MuscFlexoExtEnv-v0')

obs = env.reset(reset_type="random")
a = env.sim.data.qpos[:].copy()
# Mostrar los datos de salida
print("Nueva observación después de la acción:")
#print(a)

# Cerrar el entorno
env.close()
