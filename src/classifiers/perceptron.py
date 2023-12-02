from sklearn.linear_model import Perceptron


def classifier_perceptron():
    """
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
    """
    return {
        "classifier": [Perceptron()],
        "classifier__alpha": [0.1],
    }
