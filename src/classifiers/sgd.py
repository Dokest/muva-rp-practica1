from sklearn.linear_model import SGDClassifier

from src.utils.utils import stepped_values


def classifier_sgd():
    """
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
    """
    return {
        "classifier": [SGDClassifier()],
        "classifier__loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron", "squared_error",
                             "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
        "classifier__alpha": stepped_values(0.0001, 2, 0.10),
        "classifier__penalty": ["l2", "l1", "elasticnet"],
        "classifier__n_jobs": [-1],
        "classifier__random_state": [1]
    }
