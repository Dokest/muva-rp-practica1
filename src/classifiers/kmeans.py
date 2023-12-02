from sklearn.cluster import KMeans


def classifier_kmeans():
    """
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    """
    return {
        "classifier": [KMeans()],
        "classifier__n_clusters": [2],
    }
