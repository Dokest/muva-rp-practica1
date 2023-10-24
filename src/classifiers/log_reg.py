from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def classifier_log_reg():
    return Pipeline([
        ("LogisticRegression", LogisticRegression(max_iter=10000)),
    ])
