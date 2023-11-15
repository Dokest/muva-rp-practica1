from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import stepped_values
from sklearn.decomposition import PCA


def pca_transformer():
    pipeline = Pipeline([
        ("StandardScaler", StandardScaler()),
        ("PCA", PCA()),
    ])

    return {
        "transformer": [pipeline],
        "transformer__PCA__n_components": [0.99], # stepped_values(0.5, 0.99, 0.05),
        "transformer__PCA__whiten": [True], # [True, False],
        "transformer__PCA__svd_solver": ["auto"], #["auto", "full"],
    }
