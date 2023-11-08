from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


def classifier_k_means():
    """
    Returns an error because of some library incompatibility
    :return:
    """
    return Pipeline([
        ("KMeans", KMeans()),
    ])
