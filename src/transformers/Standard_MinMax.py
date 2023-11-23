from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import stepped_values
from sklearn.preprocessing import MinMaxScaler


def minmax_transformer():
    pipeline = Pipeline([
        ("StandardScaler", StandardScaler()),
        ("MinMax", MinMaxScaler()),
    ])

    return {
        "transformer": [pipeline],
    }
