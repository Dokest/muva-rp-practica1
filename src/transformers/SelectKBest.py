from sklearn.feature_selection import SelectKBest

from src.utils import stepped_values


def select_k_best_transformer():
    return {
        "transformer": [SelectKBest()],
        "transformer__k": stepped_values(50, 500, 10),
    }
