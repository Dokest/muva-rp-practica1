from sklearn.svm import SVC

from src.utils import stepped_values


def svm_classifier():
    return {
        "classifier": [SVC()],
        "classifier__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "classifier__degree": [2, 3, 4],
        "classifier__gamma": ["scale", "auto"] + stepped_values(1.0, 5.0, 1),
    }