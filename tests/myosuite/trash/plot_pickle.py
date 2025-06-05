import pickle
import matplotlib.pyplot as plt

# Nombre del archivo donde se guardaron los datos
filename = 'optimized_simulation_data.pkl'

# Cargar los datos desde el archivo pickle
with open(filename, 'rb') as f:
    loaded_data = pickle.load(f)

# Extraer los datos cargados
positions = loaded_data['positions']
torques = loaded_data['torques']
times = loaded_data['times']
print(positions)
# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(times, positions, label='Position')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.title('Position vs Time')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(times, torques, label='Torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque')
plt.title('Torque vs Time')
plt.legend()

plt.tight_layout()
plt.show()