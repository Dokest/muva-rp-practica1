from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import PolynomialFeatures


def transformer_select_k_best_poly_v1():
    """
    Really slow (when PolynomialFeatures is active) and not very good.

    Example:
    'f1': 0.5194805194805194
    'precision': 0.6349206349206349
    'recall': 0.43956043956043955
    'mse': 0.3915343915343915
    """
    preprocess = Pipeline([
        ("SelectKBest", SelectKBest(k=500)),
    ])

    features = FeatureUnion([
        ('PolynomialFeatures', PolynomialFeatures(degree=2))
    ])

    return Pipeline([
        ("preprocess", preprocess),
        ("features", features),
    ])

