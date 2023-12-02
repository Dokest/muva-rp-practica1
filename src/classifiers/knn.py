from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def classifier_knn():
    """Docs: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier"""
    return Pipeline([
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])
