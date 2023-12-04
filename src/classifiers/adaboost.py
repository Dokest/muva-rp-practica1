from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


def adaboost_classifier():
    """
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
    """
    return {
        "classifier": [AdaBoostClassifier()],
        "classifier__estimator": [SVC(C=2, gamma="scale", probability=True, kernel="linear", random_state=1)],
        "classifier__n_estimators": [100],  # stepped_values(1, 200, 10),
        "classifier__learning_rate": [1e-3],  # stepped_values(0.01, 2, 0.10),
        "classifier__algorithm": ["SAMME.R"],
        "classifier__random_state": [1],
    }

