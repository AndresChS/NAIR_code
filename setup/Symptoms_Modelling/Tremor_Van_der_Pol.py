# -*- coding: utf-8 -*-
"""
Van der Pol Oscillator as model for Spasticity 

Created on Wed Apr  3 09:44:41 2024

@author: andre
"""

import numpy as np
import matplotlib.pyplot as plt

# Definición de la función del oscilador de Van der Pol
def van_der_pol(x, mu):
    dx1 = x[1]
    dx2 = mu * (1 - x[0] ** 2) * x[1] - x[0]
    return np.array([dx1, dx2])

# Método de integración numérica de Euler
def euler_integration(func, x0, mu, t, dt):
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i - 1] + func(x[i - 1], mu) * dt
    return x

# Parámetros
mu = 0.5  # parámetro de no-linealidad
x0 = [1.0, 0.0]  # condiciones iniciales
t = np.arange(0, 50, 0.01)  # vector de tiempo
dt = t[1] - t[0]

# Integración numérica
x = euler_integration(van_der_pol, x0, mu, t, dt)

# Graficar
plt.plot(t, x[:, 0], label='Posición')
plt.plot(t, x[:, 1], label='Velocidad')
plt.title('Oscilador de Van der Pol')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()