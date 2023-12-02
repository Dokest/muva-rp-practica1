from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler


def minmax_transformer():
    """Docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler"""
    pipeline = Pipeline([
        ("StandardScaler", StandardScaler()),
        ("MinMax", MinMaxScaler()),
    ])

    return {
        "transformer": [pipeline],
    }
