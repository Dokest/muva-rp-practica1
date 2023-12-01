from sklearn.feature_selection import SelectKBest

from src.utils.utils import stepped_values


def select_k_best_transformer():
    """Docs: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest"""
    return {
        "transformer": [SelectKBest()],
        "transformer__k": stepped_values(50, 500, 10),
    }
