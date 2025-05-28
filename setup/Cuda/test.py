import torch
import torch.nn as nn
import torch.optim as optim

# Configuración para ejecutar en GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Generación de datos sintéticos
# Variables de entrada (x) y salida (y)
torch.manual_seed(0)
x = torch.rand(100000, 1) * 10  # Datos de entrada (1000 muestras, 1 característica)
y = 3 * x + 4 + torch.randn(100000, 1) * 0.5  # Datos de salida con algo de ruido

# Mover los datos al dispositivo (GPU)
x, y = x.to(device), y.to(device)

# Definir el modelo simple de regresión lineal
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 1 entrada, 1 salida

    def forward(self, x):
        return self.linear(x)

# Inicializar el modelo, la función de pérdida y el optimizador
model = LinearRegressionModel().to(device)
criterion = nn.MSELoss()  # Pérdida de error cuadrático medio
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Optimizador de descenso de gradiente estocástico

# Entrenamiento del modelo
epochs = 100000
for epoch in range(epochs):
    model.train()

    # Paso hacia adelante
    predictions = model(x)
    loss = criterion(predictions, y)

    # Retropropagación
    optimizer.zero_grad()
    loss.backward()

    # Actualización de parámetros
    optimizer.step()

    # Mostrar el progreso cada 100 épocas
    if (epoch + 1) % 100 == 0:
        print(f"Época {epoch + 1}/{epochs}, Pérdida: {loss.item():.4f}")

# Prueba del modelo (para ver si aprendió correctamente)
test_input = torch.tensor([[7.0]]).to(device)  # Un valor de entrada de prueba
predicted_output = model(test_input)
print(f"Para una entrada de {test_input.item()}, la salida predicha es {predicted_output.item():.2f}")
