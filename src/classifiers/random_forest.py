from sklearn.ensemble import RandomForestClassifier

from src.utils import stepped_values


def random_forest_classifier():
    return {
        "classifier": [RandomForestClassifier()],
        "classifier__n_estimators": [5500, 6000], # stepped_values(4000, 4500, 50),
        "classifier__random_state": [1],
        "classifier__n_jobs": [-1],
    }
