from sklearn.svm import SVC


def svm_classifier():
    """Docs: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC"""
    return {
        "classifier": [SVC()],
        "classifier__C": [8.4],  # stepped_values(8.4, 8.5, 0.01),
        "classifier__kernel": ["rbf"],  # ["poly", "rbf", "sigmoid", "precomputed"],
        "classifier__degree": [2],
        "classifier__gamma": ["scale"],  # ["scale", "auto"], # + stepped_values(3.0, 4.0, 1),
        "classifier__random_state": [1],
    }
