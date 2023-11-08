from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.utils import stepped_values


def classifier_random_forest():
    """
    'f1': 0.4413145539906103,
    'precision': 0.6143790849673203,
    'recall': 0.3443223443223443,
    """
    return Pipeline([
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=1)),
    ])


def random_forest_classifier():
    return {
        "classifier": [RandomForestClassifier()],
        "classifier__n_estimators": stepped_values(50, 300, 25),
        "classifier__random_state": [1],
    }

