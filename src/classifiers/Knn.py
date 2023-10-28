from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def classifier_knn():
    return Pipeline([
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])