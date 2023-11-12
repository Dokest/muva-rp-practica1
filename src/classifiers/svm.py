from sklearn.svm import SVC

from src.utils import stepped_values


def svm_classifier():
    return {
        "classifier": [SVC()],
        "classifier__C": stepped_values(0.5, 2, 0.5),
        "classifier__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "classifier__degree": [2, 3, 4],
        "classifier__gamma": ["scale", "auto"] + stepped_values(1.0, 5.0, 1),
        "classifier__random_state": [1]
    }
