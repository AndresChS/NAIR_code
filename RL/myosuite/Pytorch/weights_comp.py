


def read_file(filename):
    with open(filename, 'r') as file:
        content = file.readlines()
    return content

def save_weights_to_txt(actor, filename):
    # Abre el archivo TXT en modo escritura
    with open(filename, 'w') as file:
        
        for name, param in actor.named_parameters():
            # Parameter name
            file.write(f"Parameter name: {name}\n")
            # Weights
            file.write(f"Weights:\n{param.data.tolist()}\n\n")

def compare_files(file1, file2):
    # Leer el contenido de ambos archivos
    content1 = read_file(file1)
    content2 = read_file(file2)

    # Comparar si ambos contenidos son iguales
    if content1 == content2:
        print("Same weights")
    else:
        print("Differents weights:")
        # Print differences
        for i, (line1, line2) in enumerate(zip(content1, content2)):
            if line1 != line2:
                print(f"Difference in line {i+1}:")
                print(f"File 1: {line1.strip()}")
                print(f"File 2: {line2.strip()}")

