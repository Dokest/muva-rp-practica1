from sklearn.neighbors import KNeighborsClassifier

from src.utils.utils import stepped_values


def classifier_knn():
    """
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
    """
    return {
        "classifier": [KNeighborsClassifier()],
        "classifier__n_neighbors": stepped_values(4, 10, 1),
    }
