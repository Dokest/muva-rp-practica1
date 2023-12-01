from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline


def classifier_kmeans():
    """Docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans"""
    return Pipeline([
        ("K-Means", KMeans(n_clusters=2))
    ])