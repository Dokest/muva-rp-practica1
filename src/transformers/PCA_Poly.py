from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from src.utils import stepped_values


def transformer_pca_poly_v1():
    """
    Really powerful, but still takes too much time because of the PolynomialFeatures.

    Example:
    'f1': 0.5909909909909911
    'precision': 0.5815602836879432
    'recall': 0.6007326007326007
    'mse': 0.400352733686067
    """
    preprocess = Pipeline([
        ("StandardScaler", StandardScaler()),
        ("PCA", PCA(n_components=0.95)),
    ])

    features = FeatureUnion([
        ('PolynomialFeatures', PolynomialFeatures(degree=2))
    ])

    return Pipeline([
        ("preprocess", preprocess),
        ("features", features),
    ])


def pca_params():
    return {
        "n_components": stepped_values(0.5, 1, 0.05),
    }


def transformer_pca_poly_v2():
    """
    Almost as powerful as v1, but much faster.

    Example:
    'f1': 0.5503875968992248
    'precision': 0.5843621399176955
    'recall': 0.5201465201465202
    'mse': 0.409171075837
    """
    preprocess = Pipeline([
        ("StandardScaler", StandardScaler()),
        ("PCA", PCA(n_components=0.99)),
    ])

    return Pipeline([
        ("preprocess", preprocess),
    ])
