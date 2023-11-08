from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def classifier_kmeans():
    return Pipeline([
        ("K-Means", KMeans(n_clusters=2))
    ])