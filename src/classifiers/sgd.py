from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from src.utils import stepped_values


def classifier_sgd():
    return {
        "classifier": [SGDClassifier()],
        "classifier__loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron", "squared_error",
                             "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
        "classifier__alpha": stepped_values(0.0001, 2, 0.10),
        "classifier__penalty": ["l2", "l1", "elasticnet"],
        "classifier__n_jobs": [-1],
        "classifier__random_state": [1]
    }
