from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA


def pca_transformer():
    """
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
    """
    pipeline = FeatureUnion([
        ("StandardScaler", StandardScaler()),
        ("PCA", PCA()),
    ])

    return {
        "transformer": [pipeline],
        "transformer__PCA__n_components": [0.99],  # stepped_values(0.5, 0.99, 0.05),
        "transformer__PCA__whiten": [True],  # [True, False],
        "transformer__PCA__svd_solver": ["auto"],  # ["auto", "full"],
    }
