
def stepped_values(min: float, max: float, step: float) -> [float]:
    value = min
    all_values = []

    while value < min:
        all_values.append(value)

        value += step

    all_values.append(max)

    return all_values
