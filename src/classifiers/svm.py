from sklearn.svm import SVC

from src.utils.utils import stepped_values


def svm_classifier(C: [float] = None, kernel: [str] = None, degree: [int] = None, gamma: [str] = None, random_state: [int] = None):
    """
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

    We parametrize this method because it is the one we have been experimenting more with.
    """
    return {
        "classifier": [SVC()],
        "classifier__C": C if C else stepped_values(4.5, 5.5, 0.5),
        "classifier__kernel": kernel if kernel else ["poly", "rbf", "sigmoid", "precomputed"],
        "classifier__degree": degree if degree else [2],
        "classifier__gamma": gamma if gamma else ["scale", "auto"],  # + stepped_values(3.0, 4.0, 1),
        "classifier__random_state": random_state if random_state else [1],
    }
