from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

from src.utils import stepped_values


def bayes_classifier():
    return {
        "classifier": [GaussianNB()]
    }