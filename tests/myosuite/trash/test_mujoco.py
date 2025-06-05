import numpy as np
import mujoco

# Cargar el modelo XML del entorno en MuJoCo
model = mujoco.load_model_from_path("/Users/achs/Documents/PHD/code/myosuite/envs/myoleg_1dofexo/myoleg_1dofexo_V0.xml")  # Reemplaza "path/to/your/model.xml" con la ruta a tu archivo XML del modelo

# Crear una instancia de la simulación del entorno
sim = mujoco.MjSim(model)

# Bucle principal para la simulación y renderizado
while True:
    # Obtener la observación actual (opcional, dependiendo de tu entorno)
    obs = sim.get_state()  # Esto puede variar según cómo esté estructurado tu entorno

    # Renderizar el entorno
    viewer = mujoco.MjViewer(sim)
    viewer.render()

    # Modificar los valores de los actuadores (opcional, dependiendo de tu entorno)
    # Aquí puedes agregar tu lógica para modificar los valores de los actuadores

    # Avanzar la simulación
    sim.step()

    # Salir del bucle si se cierra la ventana de renderizado
    if viewer.closed:
        break

# Cerrar la ventana de renderizado y finalizar la simulación
viewer.finish()
