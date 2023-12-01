from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


def adaboost_classifier():
    """Docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier"""
    return {
        "classifier": [AdaBoostClassifier()],
        "classifier__estimator": [SVC(C=2, degree=2, random_state=1)],
        "classifier__n_estimators": [50], # stepped_values(1, 200, 10),
        "classifier__learning_rate": [0.01], # stepped_values(0.01, 2, 0.10),
        "classifier__algorithm": ["SAMME"],
        "classifier__random_state": [1],
    }

