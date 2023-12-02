import pickle

def stepped_values(min: float, max: float, step: float) -> [float]:
    value = min
    all_values = []

    while value < max:
        all_values.append(value)

        value += step

    all_values.append(max)

    return all_values



def save_model(model, name):
    # Guarda el modelo entrenado como un .pkl
    with open('./output/' + name, 'wb') as archivo:
        pickle.dump(model, archivo)
    print("Model saved at './output/modelo.pkl'")


def load_model(name):
    # Cargar el modelo .pkl
    with open('./output/' + name, 'rb') as archivo:
        return pickle.load(archivo)
