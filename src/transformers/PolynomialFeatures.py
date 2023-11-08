from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.utils import stepped_values


def polynomial_features_transformer():
    pipeline = Pipeline([
        ('PolynomialFeatures', PolynomialFeatures())
    ])

    return pipeline, {
        "degree": stepped_values(2, 5, 1),
        "interaction_only": [True, False],
        "include_bias": [True, False],
    }
