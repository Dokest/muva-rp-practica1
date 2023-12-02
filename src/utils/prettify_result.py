
def print_pretty_result(results: dict) -> str:
    acc = results["accuracy"]
    confusion_matrix = results["confusion_matrix"]
    f1 = results["f1"]
    precision = results["precision"]
    recall = results["recall"]

    print("Results:")
    print(f"\tAccuracy: {acc}")
    print(f"\tMatrix:")
    print(f"\t\t{confusion_matrix[0, 0]} | {confusion_matrix[0, 1]}")
    print(f"\t\t{confusion_matrix[1, 0]} | {confusion_matrix[1, 1]}")
    print(f"\tF1-score: {f1}")
    print(f"\tPrecision: {precision}")
    print(f"\tRecall: {recall}")

