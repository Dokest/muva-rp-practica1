from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures

from src.utils.utils import stepped_values


def polynomial_features_transformer():
    """
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures
    """
    return {
        "transformer": [PolynomialFeatures()],
        "transformer__degree": stepped_values(2, 5, 1),
        "transformer__interaction_only": [True, False],
        "transformer__include_bias": [True, False],
    }


def polynomial_features_with_pca_transformer():
    pipeline = FeatureUnion([
        ("PCA", PCA(n_components=0.99)),
        ("PolynomialFeatures", PolynomialFeatures()),
    ])

    return {
        "transformer": [pipeline],
        "transformer__PolynomialFeatures__degree": [2, 3],
        "transformer__PolynomialFeatures__interaction_only": [True], # [True, False],
        "transformer__PolynomialFeatures__include_bias": [False], # [True, False],
    }
