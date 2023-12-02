from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline


def classifier_perceptron():
    """Docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron"""
    return Pipeline([
        ("perceptron", Perceptron(alpha=0.1))
    ])