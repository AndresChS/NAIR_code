# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:30:22 2024

@author: andre
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Modelo de espasticidad muscular con fuerza de inercia
def espasticity_apply(self, **kwargs, obs_dict, muscles):

    obs_dict_squeeze = self.squeeze_obs_dict(obs_dict)
    qvel = obs_dict_squeeze['qvel'][muscles]
    qacc = obs_dict_squeeze['qacc'][muscles]    #Angle acceleration
    # Spasticity parameters
    K = 100     #Stiffness 
    B = 50      #Damping
    S = 0.5     #Activation trheshold
    mass = 1    #Limb mass
    L = 1       #limb length

    
        angulo, velocidad, aceleracion, K, B, S, masa, longitud_brazo):
    

    # Limitar el ángulo entre 0 y 130 grados (en radianes)
    angulo_limitado = max(0, min(angulo, np.radians(130)))

    # Cálculo del torque
    if angulo_limitado > S:
        torq = -K * (angulo_limitado - S) - B * qvel + mass * L**2 * qacc
    else:
        torq = 0

    return torq

# Simulación
angulos = np.linspace(0, np.radians(130), 100)
velocidades = np.linspace(-5, 5, 100)

# Parámetros del modelo
K = 100  # Rigidez
B = 50   # Damping
S = 0.5  # Umbral de activación
masa = 1  # Masa del brazo
longitud_brazo = 1  # Longitud del brazo
aceleracion = 5  # Aceleración angular (asumida constante en este ejemplo)

# Crear una matriz de torques
torques = np.zeros((len(angulos), len(velocidades)))

for i, angulo in enumerate(angulos):
    for j, velocidad in enumerate(velocidades):
        torques[i, j] = espasticidad(angulo, velocidad, aceleracion, K, B, S, masa, longitud_brazo)

# Gráfico de superficie
Angulo, Velocidad = np.meshgrid(angulos, velocidades)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(np.degrees(Angulo), Velocidad, torques, cmap='viridis')
ax.set_xlabel('Ángulo (grados)')
ax.set_ylabel('Velocidad Angular')
ax.set_zlabel('Torque')
ax.set_title('Modelo de Espasticidad Muscular con Fuerza de Inercia')

plt.show()
