from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline


def classifier_perceptron():
    return Pipeline([
        ("perceptron", Perceptron(alpha=0.1))
    ])