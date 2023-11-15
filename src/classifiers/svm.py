from sklearn.svm import SVC

from src.utils import stepped_values


def svm_classifier():
    return {
        "classifier": [SVC()],
        "classifier__C": stepped_values(0.5, 1.5, 0.5),
        "classifier__kernel": ["poly"], #["poly", "rbf", "sigmoid", "precomputed"],
        "classifier__degree": [2],
        "classifier__gamma": ["scale"], # ["scale", "auto"], # + stepped_values(3.0, 4.0, 1),
        "classifier__random_state": [1],
    }
