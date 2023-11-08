from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def classifier_perceptron():
    return Pipeline([
        ("perceptron", Perceptron(alpha=0.1))
    ])