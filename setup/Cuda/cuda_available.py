import torch
print("CUDA disponible:", torch.cuda.is_available())
print("Número de GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Nombre de la GPU {i}: {torch.cuda.get_device_name(i)}")
