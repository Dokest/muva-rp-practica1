from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline


def classifier_kmeans():
    return Pipeline([
        ("K-Means", KMeans(n_clusters=2))
    ])