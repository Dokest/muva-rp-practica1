from sklearn.ensemble import RandomForestClassifier


def random_forest_classifier():
    """Docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier"""
    return {
        "classifier": [RandomForestClassifier()],
        "classifier__n_estimators": [5500, 6000], # stepped_values(4000, 4500, 50),
        "classifier__random_state": [1],
        "classifier__n_jobs": [-1],
    }
