from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def classifier_sgd():
    return Pipeline([
        ("Gradient descent", SGDClassifier(max_iter=1000, tol=1e-3))
    ])