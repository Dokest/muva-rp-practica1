from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.utils import stepped_values


def random_forest_classifier():
    return {
        "classifier": [RandomForestClassifier()],
        "classifier__n_estimators": stepped_values(50, 300, 25),
        "classifier__random_state": [1],
    }

