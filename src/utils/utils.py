import os
import pickle


def stepped_values(min_value: float, max_value: float, step: float) -> [float]:
    """
    Returns an array of values between [min, max] every step passed.

    Example:
        stepped_values(0, 1, 0.5) # Returns [0, 0.5, 1]
        stepped_values(0, 5, 3) # Returns [0, 3, 5]
    """
    value = min_value
    all_values = []

    while value < max_value:
        all_values.append(value)

        value += step

    all_values.append(max_value)

    return all_values


def save_model(model, name: str, directory: str):
    """
    Stores the trained model into a .pkl file at the specified location.
    """
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name), 'wb') as archivo:
        pickle.dump(model, archivo)

    print(f"Model saved at '{os.path.join(directory, name)}'")


def load_model(name: str, directory: str):
    """
    Loads the model from a .pkl file and returns it.
    """
    with open(os.path.join(directory, name), 'rb') as archivo:
        return pickle.load(archivo)
